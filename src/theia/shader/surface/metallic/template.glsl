#include "complex.glsl"

//only specular scattering
#define SURFACE_MODEL_SPECULAR

struct SurfaceProperties {
    float reflectance;
    bool doReflect;
};
//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //reflectance might be set directly, if not returns -1.0
    float R = lookUpMaterialTable1D(REFLECTIVITY, hit.materialIdx, ray.wavelength, -1.0);

    //if not, fall back to Fresnel equations ()
    if (!(R >= 0.0)) { //this also includes NaN
        //fetch optical properties
        float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
        float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);
        float k_o = lookUpMediaTable1D(IMAG_REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 0.0);
        //metalls require complex refractive indices
        complex N0 = complex(n_i, 0.0);
        complex N1 = complex(n_o, k_o);

        //apply Snell's law
        float cos0 = abs(dot(ray.direction, hit.rayNrm));
        float sin0 = sqrt(max((1.0 - cos0) * (1.0 + cos0), 0.0));
        complex cos1 = cpyth(cdiv(N0 * sin0, N1));
        //There's a mathematical branch cut along the negative axis.
        //Floats can distinguish the branches via signed zeros (+/- 0.0).
        //Due to finite precision we might end up at either branch near zero.
        //Unfortunately, this sign can propagate causing evanescent waves to explode
        //in magnitude instead of decaying.
        //-> ensure the imaginary parts are always non-negative
        cimag(cos1) = abs(cimag(cos1));

        //calculate Fresnel coefficients
        complex rs = cdiv(2.0 * N0 * cos0, N0 * cos0 + cmul(N1, cos1)) - complex(1.0, 0.0);
        complex rp = cdiv(2.0 * N1 * cos0, N1 * cos0 + n_i * cos1) - complex(1.0, 0.0);
        //convert to power coefficients
        float Rs = cnorm(rs);
        float Rp = cnorm(rp);

        //for unpolarized light, it's the average
        R = 0.5 * (Rs + Rp);
    }

    //especially for particles it is important to make the same decision in
    //processSurfaceTargetHit() and sampleSurfaceInteraction() to ensure we
    //neither lose nor duplicate a particle
    // -> make decision here
    bool doReflect = random(idx, dim) < R;

    //return properties
    return SurfaceProperties(R, doReflect);
}

bool processSurfaceTargetHit(
    RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    int objectId,
    out HitItem item,
    uint idx, inout uint dim
) {
    //metalls never transmit, so we ignore this flag
    // bool transmitHit = (hit.flags & MATERIAL_TRANSMIT_HIT_BIT) != 0;

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
    float A = 1.0 - props.reflectance;
    if (A > 0.0) {
        ray.lin_contrib *= A;
        item = createHit(
            ray,
            hit.objPos,
            hit.objNrm,
            objectId,
            hit.worldToObj
        );
    }
    return A > 0.0;

    #endif
}

ResultCode sampleSurfaceInteraction(
    inout RAY ray,
    const SurfaceHit hit,
    const SurfaceProperties props,
    uint idx, inout uint dim
) {
    //fetch material flags
    // bool isDetector = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    // bool canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0;
    //detector implies no transmit
    // canTransmit = !isDetector && canTransmit;

    //metalls should never transmit (we assume light in metals are rapidly absorbed)
    //so the only choices are between reflection and absorption

    #if !defined(RAY_PARTICLE) && !defined(SURFACE_ABSORB_RAY)

    //never absorb -> always reflect (if there's still some light left afterwards)
    ResultCode result = RESULT_CODE_RAY_ABSORBED;
    if (props.reflectance > 0.0 && canReflect) {
        ray.lin_contrib *= props.reflectance;
        result = reflectRay(ray, hit);
    }
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #else

    //either particle or we should IS absorption anyway
    ResultCode result = RESULT_CODE_RAY_ABSORBED;
    if (props.doReflect && canReflect) {
        result = reflectRay(ray, hit);
    }
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #endif
}
