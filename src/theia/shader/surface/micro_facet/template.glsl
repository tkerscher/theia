#define SURFACE_MODEL_MICRO_FACET

#include "math.glsl"

struct SurfaceProperties {
    float reflectance;
    vec3 dirRefelected;
    vec3 dirTransmitted;
    bool doReflect;
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

// Inverse CDF (Percent Point Function) der Standardnormalverteilung
// Acklam-Approximation
// Genauigkeit typischerweise ~1e-9 in float64,
// in GLSL float natürlich geringer, aber sehr brauchbar.

float normalPPF(float p)
{
    // Schutz gegen log(0)
    p = clamp(p, 1e-7, 1.0 - 1e-7);

    // Koeffizienten
    const float a1 = -3.969683028665376e+01;
    const float a2 =  2.209460984245205e+02;
    const float a3 = -2.759285104469687e+02;
    const float a4 =  1.383577518672690e+02;
    const float a5 = -3.066479806614716e+01;
    const float a6 =  2.506628277459239e+00;

    const float b1 = -5.447609879822406e+01;
    const float b2 =  1.615858368580409e+02;
    const float b3 = -1.556989798598866e+02;
    const float b4 =  6.680131188771972e+01;
    const float b5 = -1.328068155288572e+01;

    const float c1 = -7.784894002430293e-03;
    const float c2 = -3.223964580411365e-01;
    const float c3 = -2.400758277161838e+00;
    const float c4 = -2.549732539343734e+00;
    const float c5 =  4.374664141464968e+00;
    const float c6 =  2.938163982698783e+00;

    const float d1 =  7.784695709041462e-03;
    const float d2 =  3.224671290700398e-01;
    const float d3 =  2.445134137142996e+00;
    const float d4 =  3.754408661907416e+00;

    const float plow  = 0.02425;
    const float phigh = 1.0 - plow;

    float q, r;

    // Unterer Bereich
    if (p < plow)
    {
        q = sqrt(-2.0 * log(p));

        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    // Oberer Bereich
    if (phigh < p)
    {
        q = sqrt(-2.0 * log(1.0 - p));

        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    // Zentralbereich
    q = p - 0.5;
    r = q * q;

    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch refractive indices and sigma_alpha
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);

    //initialize all variables that have to leave the for-loop
    float r = 0.0;
    vec3 dirRefelected = vec3(0,0,0);
    vec3 dirTransmitted = vec3(0,0,0);

    //sample micro-facets until a valid micro-facet is found (in most cases, the first micro-facet will be valid)
    for(int i = 0; i < 10; i++){

        //sample micro-facet orientation
        float alpha = clamp(lookUpMaterialTable1D(PPF, hit.materialIdx, random(idx,dim), 0.0), -PI, PI);
        //float alpha = clamp(0.05*normalPPF(random(idx,dim)),-PI,PI);
        float phi = PI * random(idx,dim);

        //rotate surface normal
        vec3 planeVector1 = perpendicularTo(hit.rayNrm);
        vec3 planeVector2 = perpendicularTo(hit.rayNrm, planeVector1);
        vec3 microfacetNormal = cos(alpha) * hit.rayNrm + sin(alpha) * cos(phi) * planeVector1 + sin(alpha) * sin(phi) * planeVector2;

        //calculate outgoing angle (Snell's law)
        float cos_i = abs(dot(ray.direction, microfacetNormal));
        float sin_i = sqrt(max(1.0 - cos_i*cos_i, 0.0));
        float sin_o = sin_i * n_i / n_o;
        //by clamping cos_o to 0.0 we accurately handle total internal reflection
        float cos_o = sqrt(max(1.0 - sin_o*sin_o, 0.0));

        //evaluate Fresnel terms for reflectance
        float r_s = (n_i * cos_i - n_o * cos_o) / (n_i * cos_i + n_o * cos_o);
        float r_p = (n_o * cos_i - n_i * cos_o) / (n_o * cos_i + n_i * cos_o);
        r = 0.5 * (r_s*r_s + r_p*r_p);

        //calculate reflected and transmitted direction
        dirRefelected = reflect(ray.direction, microfacetNormal);
        if(1-r > 1.0e-5){
            dirTransmitted = refract(ray.direction, microfacetNormal, n_i / n_o);
        }

        //check if a valid micro-facet was sampled
        bool cond1 = (dot(ray.direction, microfacetNormal) < 0); //check that ray hits micro-facet surface from the front
        bool cond2 = (dot(dirRefelected, hit.rayNrm) > 0); //check that reflected ray returns to the original medium
        bool cond3 = (1-r < 1.0e-5) || (dot(dirTransmitted, hit.rayNrm) < 0); //check that transmitted ray goes towards new medium, or that total reflection occurs

        if (cond1 && cond2 && cond3){
            break;
        }

        //after 10 invalid micro-facets have been sampled, an unrotated facet is chosen to continue
        if (i == 9){
            //calculate outgoing angle (Snell's law)
            float cos_i = abs(dot(ray.direction, hit.rayNrm));
            float sin_i = sqrt(max(1.0 - cos_i*cos_i, 0.0));
            float sin_o = sin_i * n_i / n_o;
            //by clamping cos_o to 0.0 we accurately handle total internal reflection
            float cos_o = sqrt(max(1.0 - sin_o*sin_o, 0.0));

            //evaluate Fresnel terms for reflectance
            float r_s = (n_i * cos_i - n_o * cos_o) / (n_i * cos_i + n_o * cos_o);
            float r_p = (n_o * cos_i - n_i * cos_o) / (n_o * cos_i + n_i * cos_o);
            r = 0.5 * (r_s*r_s + r_p*r_p);

            //calculate reflected and transmitted direction
            dirRefelected = reflect(ray.direction, hit.rayNrm);
            if(1-r > 1.0e-5){
                dirTransmitted = refract(ray.direction, hit.rayNrm, n_i / n_o);
        }
        }

    }

    //especially for particles it is important to make the same decision in
    //processSurfaceTargetHit() and sampleSurfaceInteraction() to ensure we
    //neither lose nor duplicate a particle
    // -> make decision here
    bool doReflect = random(idx, dim) < r;

    //return properties
    return SurfaceProperties(r, dirRefelected, dirTransmitted, doReflect);
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
        result = reflectRay(ray, hit, props.dirRefelected);
    }
    else if (!props.doReflect && canTransmit) {
        result = transmitRay(ray, hit, props.dirTransmitted);
        //due to finite numerical precision, we might run into total internal
        //reflection earlier than expected -> mark as absorbed
        if (result == ERROR_CODE_TOTAL_INTERNAL_REFLECTION)
            result = RESULT_CODE_RAY_ABSORBED;
    }
    else {
        //sampled decision is forbidden -> abort tracing
        //TODO: Use a better code here?
        return RESULT_CODE_RAY_ABSORBED;
    }
    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #else

    ResultCode result;
    if (canReflect && canTransmit) {
        if (props.doReflect) {
            result = reflectRay(ray, hit, props.dirRefelected);
        }
        else {
            result = transmitRay(ray, hit, props.dirTransmitted);
        }
    }
    else if(canReflect && props.reflectance > 0.0) {
        //only reflection allowed -> deterministically reflect
        ray.lin_contrib *= props.reflectance;
        result = reflectRay(ray, hit, props.dirRefelected);
    }
    else if(canTransmit && 1 - props.reflectance > 1.0e-5) {
        //only transmission allowed -> deterministically transmit
        ray.lin_contrib *= (1.0 - props.reflectance);
        result = transmitRay(ray, hit, props.dirTransmitted);
        //due to finite numerical precision, we might run into total internal
        //reflection earlier than expected -> mark as absorbed
        if (result == ERROR_CODE_TOTAL_INTERNAL_REFLECTION)
            result = RESULT_CODE_RAY_ABSORBED;
    }
    else {
        return RESULT_CODE_RAY_ABSORBED;
    }
    //success
    return result >= 0 ? RESULT_CODE_RAY_HIT : result;

    #endif
}
