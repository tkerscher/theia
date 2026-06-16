#define SURFACE_MODEL_DIELECTRIC_ROUGH_UNIFIED

#include "math.glsl"
#include "util/sample.glsl"

//sampling of micro-facets according to the Beckmann distribution
vec3 sample_microfacet_normal(const RAY ray, const SurfaceHit hit, uint idx, inout uint dim){

    //sample angles (theta, phi)
    vec2 rdm = random2D(idx, dim);
    float theta= lookUpMaterialTable1D(PPF, hit.materialIdx, rdm.x, 0.0);
    float phi = TWO_PI * rdm.y;

    //rotate surface normal
    vec3 tangentialVector1 = perpendicularTo(hit.rayNrm);
    vec3 tangentialVector2 = perpendicularTo(hit.rayNrm, tangentialVector1);
    return cos(theta) * hit.rayNrm + sin(theta) * cos(phi) * tangentialVector1 + sin(theta) * sin(phi) * tangentialVector2;
}

//check that ray hits micro-facet from the front
bool check_microfacet(vec3 dirOut, vec3 microfacetNormal, const SurfaceHit hit, const RAY ray, uint idx, inout uint dim){
    return (dot(ray.direction, microfacetNormal) < 0.0);
}

struct SurfaceProperties {
    float reflectance;
    vec3 dirReflectedMicrofacet;
    vec3 dirTransmitted;
    bool doReflect;
    bool doBackScatter;
    bool doSpecularSpike;
    bool doSpecularLobe;
    bool doDiffuseLobe;
};

//tell tracer we want to do some prep work
#define SurfaceProperties SurfaceProperties

SurfaceProperties prepareSurface(
    const RAY ray,
    const SurfaceHit hit,
    uint idx, inout uint dim
) {
    //fetch refractive indices and alpha
    float n_i = lookUpMediaTable1D(REFRACTIVE_INDEX, ray.mediumIdx, ray.wavelength, 1.0);
    float n_o = lookUpMediaTable1D(REFRACTIVE_INDEX, hit.otherMediumIdx, ray.wavelength, 1.0);

    //fetch material flags
    bool isDetector = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    bool canReflect = (hit.flags & NO_REFLECT_BIT) == 0;
    bool canTransmit = (hit.flags & NO_TRANSMIT_BIT) == 0;
    bool transmitHit = (hit.flags & MATERIAL_TRANSMIT_HIT_BIT) != 0;
    //detector implies no transmit unless requested
    canTransmit = (!isDetector && canTransmit) || transmitHit;

    //initialize all variables that have to leave the for-loop
    float r = 0.0;
    vec3 dirReflectedMicrofacet = vec3(0,0,0);
    vec3 dirTransmitted = vec3(0,0,0);
    bool doReflect;
    bool doBackScatter = false;
    bool doSpecularSpike = false;
    bool doSpecularLobe = false;
    bool doDiffuseLobe = false;

    //sample micro-facets until a valid micro-facet is found (in most cases, the first micro-facet will be valid)
    for(int i = 0; i < 8; i++){

        //sample micro-facet
        vec3 microfacetNormal = sample_microfacet_normal(ray, hit, idx, dim);

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

        doReflect = random(idx, dim) < r;

        bool cond1 = true;
        bool cond2 = true;

        if((canReflect && doReflect) || (canReflect && !canTransmit)){

            //sample reflection type
            float probBackScatter = loadMaterialSlot_vec2(PROB_BACKSCATTER, hit.materialIdx).y;
            float probSpecularSpike = loadMaterialSlot_vec2(PROB_SPECULARSPIKE, hit.materialIdx).y;
            float probSpecularLobe = loadMaterialSlot_vec2(PROB_SPECULARLOBE, hit.materialIdx).y;
            float probDiffuseLobe = loadMaterialSlot_vec2(PROB_DIFFUSELOBE, hit.materialIdx).y;

            float u = random(idx, dim);
            doBackScatter = (u < probBackScatter);
            doSpecularSpike = (!doBackScatter && u < probBackScatter + probSpecularSpike);
            doSpecularLobe = (!doBackScatter && !doSpecularSpike && u < probBackScatter + probSpecularSpike + probSpecularLobe);
            doDiffuseLobe = (!doBackScatter && !doSpecularSpike && !doSpecularLobe);

            if(doSpecularLobe){
                dirReflectedMicrofacet = reflect(ray.direction, microfacetNormal);
                //check that reflected ray returns to the original medium
                cond1 = (dot(dirReflectedMicrofacet, hit.rayNrm) > 0);
                //model-dependent validity check
                cond2 = check_microfacet(dirReflectedMicrofacet, microfacetNormal, hit, ray, idx, dim);
            }
            
        }
        else if(r < 1.0){
            dirTransmitted = refract(ray.direction, microfacetNormal, n_i / n_o);
            //check that transmitted ray goes towards new medium, or that total reflection occurs
            cond1 = (dot(dirTransmitted, hit.rayNrm) < 0) || (dirTransmitted == vec3(0.0));
            //model-dependent validity check
            cond2 = check_microfacet(dirTransmitted, microfacetNormal, hit, ray, idx, dim);
        }
        
        if (cond1 && cond2){
            break;
        }

        //after 8 invalid micro-facets have been sampled, an unrotated facet is chosen to continue
        if (i == 7){
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
            dirReflectedMicrofacet = reflect(ray.direction, hit.rayNrm);
            dirTransmitted = refract(ray.direction, hit.rayNrm, n_i / n_o);

            //sample reflection type
            float probBackScatter = loadMaterialSlot_vec2(PROB_BACKSCATTER, hit.materialIdx).y;
            float probSpecularSpike = loadMaterialSlot_vec2(PROB_SPECULARSPIKE, hit.materialIdx).y;
            float probSpecularLobe = loadMaterialSlot_vec2(PROB_SPECULARLOBE, hit.materialIdx).y;
            float probDiffuseLobe = loadMaterialSlot_vec2(PROB_DIFFUSELOBE, hit.materialIdx).y;

            float u = random(idx, dim);
            doBackScatter = (u < probBackScatter);
            doSpecularSpike = (!doBackScatter && u < probBackScatter + probSpecularSpike);
            doSpecularLobe = (!doBackScatter && !doSpecularSpike && u < probBackScatter + probSpecularSpike + probSpecularLobe);
            doDiffuseLobe = (!doBackScatter && !doSpecularSpike && !doSpecularLobe && u < probBackScatter + probSpecularSpike + probSpecularLobe + probDiffuseLobe);

            doReflect = random(idx, dim) < r;
        }

    }

    //return properties
    return SurfaceProperties(r, dirReflectedMicrofacet, dirTransmitted, doReflect, 
                doBackScatter, doSpecularSpike, doSpecularLobe, doDiffuseLobe);
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
    if (props.doReflect && props.doBackScatter) {
        vec3 dirReflected = -ray.direction;
        result = reflectRay(ray, hit, dirReflected);
    }
    else if (props.doReflect && props.doSpecularSpike) {
        result = reflectRay(ray, hit);
    }
    else if (props.doReflect && props.doSpecularLobe) {
        result = reflectRay(ray, hit, props.dirReflectedMicrofacet);
    }
    else if (props.doReflect && props.doDiffuseLobe) {
        vec3 localDir = sampleHemisphereCosine(random2D(idx, dim));
        vec3 dirReflected = createLocalCOSY(hit.rayNrm) * localDir;
        result = reflectRay(ray, hit, dirReflected);
    }
    else if (!props.doReflect && canTransmit) {
        //due to finite numerical precision, we might run into total internal
        //reflection earlier than expected -> mark as absorbed
        if (props.dirTransmitted == vec3(0.0))
            return RESULT_CODE_RAY_ABSORBED;
        result = transmitRay(ray, hit, props.dirTransmitted);
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
        if (props.doReflect && props.doBackScatter) {
            vec3 dirReflected = -ray.direction;
            result = reflectRay(ray, hit, dirReflected);
        }
        else if (props.doReflect && props.doSpecularSpike) {
            result = reflectRay(ray, hit);
        }
        else if (props.doReflect && props.doSpecularLobe) {
            result = reflectRay(ray, hit, props.dirReflectedMicrofacet);
        }
        else if (props.doReflect && props.doDiffuseLobe) {
            vec3 localDir = sampleHemisphereCosine(random2D(idx, dim));
            vec3 dirReflected = createLocalCOSY(hit.rayNrm) * localDir;
            result = reflectRay(ray, hit, dirReflected);
        }
        else {
            //due to finite numerical precision, we might run into total internal
            //reflection earlier than expected -> mark as absorbed
            if (props.dirTransmitted == vec3(0.0))
                return RESULT_CODE_RAY_ABSORBED;
            result = transmitRay(ray, hit, props.dirTransmitted);
        }
    }
    else if(canReflect && props.reflectance > 0.0) {
        //only reflection allowed -> deterministically reflect
        ray.lin_contrib *= props.reflectance;
        if (props.doBackScatter) {
            vec3 dirReflected = -ray.direction;
            result = reflectRay(ray, hit, dirReflected);
        }
        else if (props.doSpecularSpike) {
            result = reflectRay(ray, hit);
        }
        else if (props.doSpecularLobe) {
            result = reflectRay(ray, hit, props.dirReflectedMicrofacet);
        }
        else if (props.doDiffuseLobe) {
            vec3 localDir = sampleHemisphereCosine(random2D(idx, dim));
            vec3 dirReflected = createLocalCOSY(hit.rayNrm) * localDir;
            result = reflectRay(ray, hit, dirReflected);
    }
    }
    else if(canTransmit && props.reflectance < 1.0) {
        //due to finite numerical precision, we might run into total internal
        //reflection earlier than expected -> mark as absorbed
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




