#include "math.glsl"
#include "util/sample.glsl"
#include "surface/fresnel.glsl"
#include "surface/lobes.glsl"

/*
Shared implementation of all rough dielectric surface models. The only
model-specific parts are `sample_microfacet_normal` and `check_microfacet`,
which are provided by one of the files in `microfacet_models/`.

The reflected component is decomposed into a specular spike, specular lobe,
diffuse lobe and backscattering. The split is controlled by the optional material
properties `prob_backscatter`, `prob_specularspike`, `prob_specularlobe` and
`prob_diffuselobe`. If these are not provided (or all of backscatter/spike/diffuse
are zero) the surface falls back to a pure specular lobe.

The reflectance is either taken from the optional material property
`reflectivity` or, if not given, computed from the Fresnel equations evaluated
at the sampled micro-facet.
*/

#define SURFACE_MODEL_SPECULAR

struct SurfaceProperties {
    float reflectance;
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

    //sample which reflection lobe to use (specular spike/lobe, diffuse, backscatter)
    ReflectionLobes lobes = sampleReflectionLobes(hit, idx, dim);

    //fetch optional reflectivity. If not provided, use the Fresnel reflectance.
    bool hasReflectivity = false;
    float reflectivity = 0.0;
    #ifdef MATERIAL_SLOT_REFLECTIVITY
        reflectivity = lookUpMaterialTable1D(REFLECTIVITY, hit.materialIdx, ray.wavelength, -1.0);
        //lookUp returns -1.0 if this material has no reflectivity table
        hasReflectivity = (reflectivity >= 0.0);
    #endif

    //pre-sample the diffuse hemisphere direction once (only used by the diffuse lobe)
    vec3 diffuseDir = vec3(0.0);
    if (lobes.doDiffuseLobe)
        diffuseDir = createLocalCOSY(hit.rayNrm) * sampleHemisphereCosine(random2D(idx, dim));

    //we only need a micro-facet normal if the specular lobe direction is needed,
    //if we have to compute the Fresnel reflectance, or if the ray may transmit
    bool needMicrofacet = lobes.doSpecularLobe || !hasReflectivity || canTransmit;

    float r = hasReflectivity ? reflectivity : 0.0;
    vec3 dirReflected = vec3(0.0);
    vec3 dirTransmitted = vec3(0.0);
    bool doReflect = false;

    if (needMicrofacet) {
        //compute the facet-independent micro-facet parameters once (roughness,
        //tangent basis, ...) and reuse them across all retries
        MicrofacetParams mfParams = prepare_microfacet(ray.direction, hit);

        //sample micro-facets until a valid one is found (in most cases the first is valid)
        bool valid = false;
        for (int i = 0; i < 20; i++) {
            vec3 microfacetNormal = sample_microfacet_normal(mfParams, ray.direction, hit, idx, dim);

            //Fresnel reflectance at the micro-facet (unless a reflectivity is given)
            float cos_i = abs(dot(ray.direction, microfacetNormal));
            if (!hasReflectivity)
                r = fresnelReflectance(cos_i, n_i, n_o);

            doReflect = random(idx, dim) < r;

            bool cond1 = true;
            bool cond2 = true;

            if ((canReflect && doReflect) || (canReflect && !canTransmit)) {
                //determine outgoing direction of the sampled reflection lobe
                dirReflected = lobeReflectedDir(lobes, ray.direction, hit.rayNrm, microfacetNormal, diffuseDir);
                //check that reflected ray returns to the original medium
                cond1 = (dot(dirReflected, hit.rayNrm) > 0);
                //model-dependent validity check
                vec3 dirReflectedMicrofacet = lobes.doSpecularLobe ?
                    dirReflected : reflect(ray.direction, microfacetNormal);
                cond2 = check_microfacet(mfParams, dirReflectedMicrofacet, microfacetNormal, hit, ray.direction, idx, dim);

            }
            else if (r < 1.0) {
                dirTransmitted = refract(ray.direction, microfacetNormal, n_i / n_o);
                //check that transmitted ray goes towards new medium, or that total reflection occurs
                cond1 = (dot(dirTransmitted, hit.rayNrm) < 0) || (dirTransmitted == vec3(0.0));
                //model-dependent validity check
                cond2 = check_microfacet(mfParams, dirTransmitted, microfacetNormal, hit, ray.direction, idx, dim);
            }

            if (cond1 && cond2) {
                valid = true;
                break;
            }
        }

        //no valid micro-facet after 20 attempts -> fall back to an unrotated facet (the surface normal)
        if (!valid) {
            float cos_i2 = abs(dot(ray.direction, hit.rayNrm));
            if (!hasReflectivity)
                r = fresnelReflectance(cos_i2, n_i, n_o);
            dirReflected = lobeReflectedDir(lobes, ray.direction, hit.rayNrm, hit.rayNrm, diffuseDir);
            dirTransmitted = refract(ray.direction, hit.rayNrm, n_i / n_o);
            doReflect = random(idx, dim) < r;
        }
    }
    else {
        //reflectivity given and no micro-facet needed (reflection-only, no specular lobe)
        doReflect = random(idx, dim) < r;
        dirReflected = lobeReflectedDir(lobes, ray.direction, hit.rayNrm, hit.rayNrm, diffuseDir);
    }

    //return properties
    return SurfaceProperties(r, dirReflected, dirTransmitted, doReflect);
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

    //we have a local copy of the ray. Attenuate by reflectance before detecting
    ray.lin_contrib *= (1.0 - props.reflectance);
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

    //handle surface reflection/transmission
    #ifdef RAY_PARTICLE

    //importance sample what to do
    ResultCode result;
    if (props.doReflect && canReflect) {
        result = reflectRay(ray, hit, props.dirReflected);
    }
    else if (!props.doReflect && canTransmit) {
        //due to finite numerical precision, or if a custom reflectivity is given, 
        //we might run into total internal reflection -> mark as absorbed
        if (props.dirTransmitted == vec3(0.0))
            return RESULT_CODE_RAY_ABSORBED;
        result = transmitRay(ray, hit, props.dirTransmitted);
    }
    else {
        //sampled decision is forbidden -> abort tracing
        return RESULT_CODE_RAY_ABSORBED;
    }
    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #else

    ResultCode result;
    if (canReflect && canTransmit) {
        if (props.doReflect) {
            result = reflectRay(ray, hit, props.dirReflected);
        }
        else {
            //due to finite numerical precision, or if a custom reflectivity is given, 
            //we might run into total internal reflection -> mark as absorbed
            if (props.dirTransmitted == vec3(0.0))
                return RESULT_CODE_RAY_ABSORBED;
            result = transmitRay(ray, hit, props.dirTransmitted);
        }
    }
    else if (canReflect && props.reflectance > 0.0) {
        //only reflection allowed -> deterministically reflect
        ray.lin_contrib *= props.reflectance;
        result = reflectRay(ray, hit, props.dirReflected);
    }
    else if (canTransmit && props.reflectance < 1.0) {
        //due to finite numerical precision, or if a custom reflectivity is given, 
        //we might run into total internal reflection -> mark as absorbed
        if (props.dirTransmitted == vec3(0.0))
            return RESULT_CODE_RAY_ABSORBED;
        //only transmission allowed -> deterministically transmit
        ray.lin_contrib *= (1.0 - props.reflectance);
        result = transmitRay(ray, hit, props.dirTransmitted);
    }
    else {
        return RESULT_CODE_RAY_ABSORBED;
    }
    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #endif
}
