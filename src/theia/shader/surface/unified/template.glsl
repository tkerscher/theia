#include "math.glsl"
#include "util/sample.glsl"
#include "surface/fresnel.glsl"
#include "surface/lobes.glsl"

/*
Implementation of the Geant4 UNIFIED surface model for a rough dielectric
interface. It differs from the other rough surface models in two ways, both taken
from Geant4:

 - The reflection is decomposed into a specular spike, specular lobe, diffuse
   lobe and backscattering, weighted by the material properties `prob_*`
   (see surface/lobes.glsl).
 - A facet whose outgoing direction does not leave the micro surface does not
   restart the interaction. The walk continues with the direction that facet
   produced, which is Geant4's stand-in for multiple scattering on the micro
   structure. The walk also follows the ray across the interface: a transmission
   that fails to leave puts the ray on the far side, where it may cross back and
   end up reflected after all.

The facets always come from the Gaussian slope-angle distribution. The walk
relies on it: only because that distribution is isotropic around the macroscopic
normal and independent of the incoming direction do the micro-facet parameters
stay valid across the whole walk, and a facet on the far side is nothing but a
negated sample.
*/

#define SURFACE_MODEL_SPECULAR

/*
Number of micro-facet interactions after which the walk is forced to end. The
extra iteration uses the macroscopic normal, which always sends the ray out of
the micro surface, so the walk cannot run past it. Geant4 instead loops until it
succeeds (with a warning after 100 boundary actions).
*/
#define UNIFIED_MAX_BOUNCES 20

struct SurfaceProperties {
    vec3 dirOut;        ///< direction in which the walk left the micro surface
    bool doReflect;     ///< true if the walk left on the side it came from
    bool valid;         ///< false if the walk failed and the ray is absorbed
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch refractive indices of both sides
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);

    SurfaceProperties props;
    props.dirOut = vec3(0.0);
    props.doReflect = false;
    props.valid = false;

    /*
    Which reflection lobe to use is drawn at every reflection event, as Geant4
    does at a dielectric interface, where ChooseReflection() sits inside the
    boundary loop.

    The diffuse direction is pre-sampled once: every lobe but the specular one
    leaves the micro surface immediately, so it is used at most once per walk.
    */
    vec3 diffuseDir = createLocalCOSY(hit.rayNrm) * sampleHemisphereCosine(random2D(idx, dim));

    //facet-independent micro-facet parameters (roughness, tangent basis), valid
    //for the whole walk
    MicrofacetParams mfParams = prepare_microfacet(ray.direction, hit);

    //direction the ray currently travels in and the side of the interface it is on
    vec3 dir = ray.direction;
    bool farSide = false;
    for (int i = 0; i <= UNIFIED_MAX_BOUNCES; i++) {
        //macroscopic normal and refractive indices as seen from the current side
        vec3 nrm = farSide ? -hit.rayNrm : hit.rayNrm;
        float nIn = farSide ? n_o : n_i;
        float nOut = farSide ? n_i : n_o;

        //Sample a facet, mirrored to the current side. The last iteration is the
        //fall-back and uses the macroscopic normal instead.
        vec3 h = nrm;
        if (i < UNIFIED_MAX_BOUNCES) {
            h = sample_microfacet_normal(mfParams, dir, hit, idx, dim);
            if (farSide) h = -h;
            //Geant4's GetFacetNormal() redraws until the facet faces the photon.
            //That leaves the direction untouched, so unlike the walk below this
            //really is a plain facet rejection.
            if (dot(dir, h) >= 0.0) continue;
        }
        float cos_i = -dot(dir, h);

        //returns 1.0 in total internal reflection
        float F = fresnelReflectance(cos_i, nIn, nOut);
        bool doReflect = random(idx, dim) < F;
        ReflectionLobes lobes = sampleReflectionLobes(hit, idx, dim);

        if (doReflect) {
            //the diffuse lobe scatters around the macroscopic normal of the
            //current side, so the pre-sampled direction has to be mirrored too
            dir = lobeReflectedDir(
                lobes, dir, nrm, h, farSide ? -diffuseDir : diffuseDir);
        }
        else {
            vec3 dirRefracted = refract(dir, h, nIn / nOut);
            //A zero direction marks total internal reflection, which the Fresnel
            //coin above should have excluded. Due to finite numerical precision
            //that edge is a bit fuzzy -> mark as absorbed.
            if (dirRefracted == vec3(0.0))
                return props;
            //the ray went through the facet and is now on the other side
            dir = dirRefracted;
            farSide = !farSide;
        }

        //the walk ends as soon as the direction leaves the micro surface
        if (dot(dir, farSide ? -hit.rayNrm : hit.rayNrm) > 0.0) {
            props.valid = true;
            break;
        }
    }

    props.dirOut = dir;
    //the ray left on the far side exactly if it crossed the interface an odd
    //number of times
    props.doReflect = !farSide;
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
    /*
    A walk that continues as a reflection never reached the detector, everything
    else did: a transmission, or an outcome the flags absorb.

    Unlike the other rough models there is nothing to attenuate with here. The
    walk mixes reflection and transmission, so there is no analytic transmitted
    fraction - the walk itself decides, the same way Geant4 does.
    */
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    if (props.valid && props.doReflect && canReflect)
        return false;

    //if requested, transmit ray before detecting
    //(this does not change ray contribution)
    bool transmitHit = (hit.flags & MATERIAL_TRANSMIT_HIT_BIT) != 0;
    if (transmitHit && props.valid && !props.doReflect) {
        transmitRay(ray, hit, props.dirOut);
    }
    item = createHit(
        ray,
        hit.objPos,
        hit.objNrm,
        objectId,
        hit.worldToObj
    );
    return true;
}

ResultCode sampleSurfaceInteraction(
    inout RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    uint idx, inout uint dim
) {
    if (!props.valid)
        return RESULT_CODE_RAY_ABSORBED;

    //fetch material flags
    bool isDetector = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    bool canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0;
    //detector implies no transmit
    canTransmit = !isDetector && canTransmit;

    /*
    The walk does not depend on the flags at all, so every flag combination shows
    the exact same physics: forbidding a channel only absorbs what the surface may
    no longer emit. Weighting instead of absorbing is not an option here - the walk
    can reach the reflected outcome through the transmitting channel and back, so
    there is no per-facet probability to replace by its expectation.
    */
    ResultCode result;
    if (props.doReflect) {
        if (!canReflect)
            return RESULT_CODE_RAY_ABSORBED;
        result = reflectRay(ray, hit, props.dirOut);
    }
    else {
        if (!canTransmit)
            return RESULT_CODE_RAY_ABSORBED;
        result = transmitRay(ray, hit, props.dirOut);
    }

    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;
}
