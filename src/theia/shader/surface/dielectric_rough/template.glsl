#include "math.glsl"
#include "util/sample.glsl"
#include "surface/fresnel.glsl"

/*
Shared implementation of all rough dielectric surface models. The only
model-specific parts are `prepare_microfacet`, `sample_microfacet_normal` and
`microfacet_accept_prob`, which are provided by one of the files in
`microfacet_models/` (see api.glsl there).

The Geant4 UNIFIED model, which splits the reflection into several lobes and
walks across the micro structure instead of rejecting facets, lives in
surface/unified/ instead.
*/

#define SURFACE_MODEL_SPECULAR

struct SurfaceProperties {
    float reflectance;      ///< Fresnel reflectance at the accepted micro-facet
    float weight;           ///< accumulated weight of the facet loop
    vec3 dirReflected;
    vec3 dirTransmitted;
    bool doReflect;
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch refractive indices
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);

    //fetch material flags
    bool isDetector = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    bool canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0;
    bool transmitHit = (hit.flags & MATERIAL_TRANSMIT_HIT_BIT) != 0;
    //detector implies no transmit unless requested
    canTransmit = (!isDetector && canTransmit) || transmitHit;

    SurfaceProperties props;
    props.reflectance = 0.0;
    props.weight = 1.0;
    props.dirReflected = vec3(0.0);
    props.dirTransmitted = vec3(0.0);
    props.doReflect = false;

    /*
    For reflect-and-transmit ("RT") we use the following rejection sampling:
    sample microfacet -> check if outgoing direction is valid -> either use
    that direction or resample

    The underlying assumption is that invalid directions correspond to multi-
    scattering, and that the distribution of the twice scattered rays is identical
    to single scattered rays (the following paper argued that this is roughly 
    correct: https://api.semanticscholar.org/CorpusID:221737278).

    We reproduce ("RT") behaviour for each flag combination. Let R be the reflectance,
    a_R the acceptance probability for reflected rays and a_T for transmitted rays.
    Per facet the loop accepts a micro-facet as a reflection with probability 
    q_R = R*a_R and as a transmission with q_T = (1-R)*a_T. If one of the two channels 
    is forbidden ("R" or "T"), we use:

        weight *= 1 - q_forbidden          accept with q_allowed/(1 - q_forbidden)

    which leaves both the emitted mass q_allowed and the continuation mass
    1 - q_R - q_T of the loop untouched.

    Particles cannot carry a weight, so they always take the plain
    reflect-and-transmit decision; sampleSurfaceInteraction() then absorbs the
    outcomes the flags forbid.
    */

#ifdef RAY_PARTICLE
    bool weightReflect = false;
    bool weightTransmit = false;
#else
    bool weightReflect = canReflect && !canTransmit;
    bool weightTransmit = !canReflect && canTransmit;
#endif
    //check if we have to compute both reflection and transmission
    bool bothChannels = weightReflect || weightTransmit;

    //nothing may leave the surface at all
    if (!canReflect && !canTransmit) {
        props.weight = 0.0;
        return props;
    }

    //compute the facet-independent micro-facet parameters once
    MicrofacetParams mfParams = prepare_microfacet(ray.direction, hit);
    float eta = n_i / n_o;

    //Sample micro-facets until one is accepted; in most cases the first one is.
    //The extra iteration is the fall-back after 20 failed attempts: an unrotated
    //facet (the surface normal), accepted as long as any channel can take it.
    bool valid = false;
    vec3 acceptedNormal = hit.rayNrm;
    for (uint i = 0; i <= 20; i++) {
        bool last = (i == 20);
        vec3 microfacetNormal = hit.rayNrm;
        if (!last)
            microfacetNormal = sample_microfacet_normal(mfParams, ray.direction, hit, idx, dim);
        acceptedNormal = microfacetNormal;

        float cos_i = abs(dot(ray.direction, microfacetNormal));
        props.reflectance = fresnelReflectance(cos_i, n_i, n_o);
        float u = random(idx, dim);

        bool accepted;
        if (!bothChannels) {
            //Plain reflect-and-transmit: pick the channel by the Fresnel coin,
            //then test only that one.
            props.doReflect = u < props.reflectance;
            vec3 dir;
            bool rightSide;
            float v;
            if (props.doReflect) {
                dir = reflect(ray.direction, microfacetNormal);
                props.dirReflected = dir;
                //can reuse same random number for acceptance decision
                v = u / props.reflectance;
                rightSide = dot(dir, hit.rayNrm) > 0.0;
            }
            else {
                dir = refract(ray.direction, microfacetNormal, eta);
                props.dirTransmitted = dir;
                //can reuse same random number for acceptance decision
                v = (u - props.reflectance) / (1 - props.reflectance);
                rightSide = dot(dir, hit.rayNrm) < 0.0;
            }
            //acceptance probability
            float a = rightSide
                ? microfacet_accept_prob(mfParams, dir, microfacetNormal, hit, ray.direction)
                : 0.0;

            accepted = last ? (a > 0.0) : (v < a);
        }

        else {
            //the forbidden channel's probability is also needed for the weight
            props.dirReflected = reflect(ray.direction, microfacetNormal);
            props.dirTransmitted = refract(ray.direction, microfacetNormal, eta);
            //acceptance probability of reflection
            float aR = dot(props.dirReflected, hit.rayNrm) > 0.0
                ? microfacet_accept_prob(
                    mfParams, props.dirReflected, microfacetNormal, hit, ray.direction)
                : 0.0;
            //acceptance probability of transmission
            float aT = dot(props.dirTransmitted, hit.rayNrm) < 0.0
                ? microfacet_accept_prob(
                    mfParams, props.dirTransmitted, microfacetNormal, hit, ray.direction)
                : 0.0;
            float qR = props.reflectance * aR;
            float qT = (1.0 - props.reflectance) * aT;

            if (weightReflect) {
                float survive = 1.0 - qT;
                props.weight *= survive;
                props.doReflect = true;
                accepted = props.weight <= 0.0 || u * survive < qR
                    || (last && qR > 0.0);
            }
            else {
                float survive = 1.0 - qR;
                props.weight *= survive;
                props.doReflect = false;
                accepted = props.weight <= 0.0 || u * survive < qT
                    || (last && qT > 0.0);
            }
        }

        if (accepted) {
            valid = true;
            break;
        }
    }

    //can only happen if the unrotated facet accepts no channel at all
    if (!valid)
        props.weight = 0.0;

    //Only the picked channel's direction was computed, but a transmit-hit
    //detector reads the transmitted one regardless of what the ray does.
    if (transmitHit && props.doReflect && !bothChannels)
        props.dirTransmitted = refract(ray.direction, acceptedNormal, eta);

    return props;
}

bool processSurfaceTargetHit(
    RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    //if requested, transmit ray before detecting
    //(this does not change ray contribution)
    bool transmitHit = (hit.flags & MATERIAL_TRANSMIT_HIT_BIT) != 0;
    if (transmitHit) {
        transmitRay(ray, hit, props.dirTransmitted);
    }

    #ifdef RAY_PARTICLE

    //we can only detect whole particles -> ignore if we sampled reflection earlier
    if (!props.doReflect) {
        item = createHit(
            ray,
            hit.objPos,
            hit.objNrm,
            objectId,
            hit.worldToObj
        );
    }
    return !props.doReflect;

    #else

    //We have a local copy of the ray. Attenuate before detecting: what is not
    //reflected back is what enters the detector. sampleSurfaceInteraction()
    //passes props.weight on to the reflected ray, so the complement is exactly
    //what the surface does not send back.
    ray.lin_contrib *= (1.0 - (props.doReflect ? props.weight : 0.0));
    item = createHit(
        ray,
        hit.objPos,
        hit.objNrm,
        objectId,
        hit.worldToObj
    );
    return true;

    #endif
}

ResultCode sampleSurfaceInteraction(
    inout RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    uint idx, inout uint dim
) {
    //fetch material flags
    bool isDetector = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    bool canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0;
    //detector implies no transmit
    canTransmit = !isDetector && canTransmit;

    if (props.weight <= 0.0)
        return RESULT_CODE_RAY_ABSORBED;

    ResultCode result;
    if (props.doReflect) {
        if (!canReflect)
            return RESULT_CODE_RAY_ABSORBED;
        #ifndef RAY_PARTICLE
        ray.lin_contrib *= props.weight;
        #endif
        result = reflectRay(ray, hit, props.dirReflected);
    }
    else {
        //due to finite numerical precision we might run into total internal
        //reflection -> mark as absorbed
        if (!canTransmit || props.dirTransmitted == vec3(0.0))
            return RESULT_CODE_RAY_ABSORBED;
        #ifndef RAY_PARTICLE
        ray.lin_contrib *= props.weight;
        #endif
        result = transmitRay(ray, hit, props.dirTransmitted);
    }
    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;
}
