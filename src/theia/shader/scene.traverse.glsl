#ifndef _INCLUDE_SCENE_TRAVERSE
#define _INCLUDE_SCENE_TRAVERSE

#include "material.glsl"
#include "ray.propagate.glsl"
#include "result.glsl"
#include "scatter.surface.glsl"
#include "scene.intersect.glsl"

#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "response.glsl"

//pushes ray across border:
// - offsets ray to prevent self intersection on correct side
//   (assumes ray's current position is on interface)
// - updates ray medium
void crossBorder(inout Ray ray, Material mat, vec3 geomNormal, bool inward) {
    //put ray on other side to prevent self intersection
    ray.position = offsetRay(ray.position, -geomNormal);
    //update medium & constants in sample
    Medium medium = inward ? mat.inside : mat.outside;
    ray.medium = uvec2(medium);
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        ray.samples[i].constants = lookUpMedium(medium, ray.samples[i].wavelength);
    }
}

//reflects ray and updates contribution taking IS and reflectance into account
void reflectRay(
    inout Ray ray,
    vec3 worldNrm, vec3 geomNrm,
    float r[N_LAMBDA], //reflectance
    bool canTransmit //needed for correction factor caused by importance sampling  
) {
    //update ray
    ray.position = offsetRay(ray.position, geomNrm); //offset to prevent self intersection
    ray.direction = normalize(reflect(ray.direction, worldNrm));

    //update samples
    float p = canTransmit ? r[0] : 1.0; //IS prob
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        ray.samples[i].lin_contrib *= r[i] / p;
    }
}

//transmits ray; updates ray, medium and contribution:
// - crosses border into new medium
// - refracts direction
//   -> due to dispersion cancels all but the first sample
// - takes transmission into account
// - takes IS into account
void transmitRay(
    inout Ray ray,
    Material mat,
    vec3 worldNrm, vec3 geomNormal,
    float r, //reflectance
    bool inward, bool canReflect //needed for IS factor
) {
    //since we expect dispersion, the refracted direction would be different
    //for each sample -> only transmit first sample
    [[unroll]] for (uint i = 1; i < N_LAMBDA; ++i) {
        ray.samples[i].lin_contrib = 0.0;
    }

    //cross border, but remember both refractive indices
    float n_i = ray.samples[0].constants.n;
    crossBorder(ray, mat, geomNormal, inward);
    float n_o = ray.samples[0].constants.n;

    //calculate new direction
    float eta = n_i / n_o;
    ray.direction = normalize(refract(ray.direction, worldNrm, eta));

    //update sample contribution:
    //the factor (1-r) cancels with importance sampling the reflection leaving
    //only a factor of eta^-2 for the radiance due to rays getting denser
    float inv_eta = n_o / n_i;
    ray.samples[0].lin_contrib *= inv_eta * inv_eta;
    if (!canReflect) {
        //no random choice happen -> need to also account for transmission factor
        ray.samples[0].lin_contrib *= (1.0 - r);
    }
}

//creates a response using the provided hit data
void createResponse(
    const Ray ray, Material mat,
    float maxTime,
    vec3 objPos, vec3 objDir, vec3 objNrm, vec3 worldNrm,
    bool absorb, float weight
) {
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {        
        //skip out of bounds samples
        if (ray.samples[i].time > maxTime)
            continue;
        
        //calculate contribution
        Sample s = ray.samples[i];
        float contrib = weight * s.lin_contrib * exp(s.log_contrib);
        //attenuate by transmission if not absorbing
        if (!absorb) {
            contrib *= (1.0 - reflectance(
                mat, s.wavelength, s.constants.n, ray.direction, worldNrm
            ));
        }

        //create response
        response(HitItem(
            objPos, objDir, objNrm,
            s.wavelength,
            s.time,
            contrib
        ));
    }
}

//processes a scene intersection:
// - transmit/reflection/volume border
//   -> updates ray accordingly
// - optional detector response (via allowResponse)
// - result code indicates tracing abortion (e.g. absorbed)
//afterwards ray can be again traced against the scene to create next event
ResultCode processHit(
    inout Ray ray,
    rayQueryEXT rayQuery,
    uint targetId,
    float maxTime,
    float u,            //random number for transmit/reflect decision
    bool allowResponse  //whether detector hits generate responses
) {
    //process ray query
    vec3 objPos, objNrm, objDir, worldPos, worldNrm, geomNormal;
    Material mat; bool inward;
    ResultCode result = processRayQuery(
        ray, rayQuery,
        mat, inward,
        objPos, objNrm, objDir,
        worldPos, worldNrm,
        geomNormal
    );
    float dist = distance(worldPos, ray.position);
    //update ray position with more accurate value
    ray.position = worldPos;
    //check result
    if (result < 0)
        return result;
    
    //correction factor for worldNrm/geomNormal mismatch:
    //we importance sampled according to geomNormal (jacobian)
    //but lambert uses interpolated worldNrm
    float cosWorld = abs(dot(ray.direction, worldNrm));
    float cosGeom = abs(dot(ray.direction, geomNormal));
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        ray.samples[i].lin_contrib *= cosWorld / cosGeom;
    }
    
    //process flags
    uint flags = inward ? mat.flagsInwards : mat.flagsOutwards;
    bool isDet = (flags & MATERIAL_DETECTOR_BIT) != 0;
    bool black = (flags & MATERIAL_BLACK_BODY_BIT) != 0;
    bool canTransmit = (flags & MATERIAL_NO_TRANSMIT_BIT) == 0;
    bool canReflect = (flags & MATERIAL_NO_REFLECT_BIT) == 0;

    //calculate reflectance
    float r[N_LAMBDA];
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        r[i] = reflectance(
            mat,
            ray.samples[i].wavelength,
            ray.samples[i].constants.n,
            ray.direction,
            worldNrm
        );
    }

    //check if we hit target
    int customId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    bool hitTarget = customId == targetId && isDet;
    result = RESULT_CODE_RAY_HIT;
    //create response if we're allowed to
    if (hitTarget && allowResponse) {
        createResponse(
            ray, mat,
            maxTime,
            objPos, objDir, objNrm, worldNrm,
            black, 1.0
        );
        //dont return early as the ray is allowed to carry on
        //(e.g. partially reflect)
        result = RESULT_CODE_RAY_DETECTED;
    }

    //stop tracing if ray absorbed
    if (black) return RESULT_CODE_RAY_ABSORBED;

#ifndef DISABLE_VOLUME_BORDER
    //check for volume border
    if ((flags & MATERIAL_VOLUME_BORDER_BIT) != 0) {
        crossBorder(ray, mat, geomNormal, inward);
        return RESULT_CODE_VOLUME_HIT;
    }
#endif

    //world normal indicates on which side of the geometry we are
    //GLSL reflect()/refract() require the normal to be on the same side
    // -> flip if necessary
    //note: geomNormal already ensures that
    worldNrm *= 2.0 * float(inward) - 1.0; //inward ? 1.0 : -1.0

#ifndef DISABLE_TRANSMISSION
    //handle reflection/transmission
    if (canReflect && (!canTransmit || u < r[0])) {
        reflectRay(ray, worldNrm, geomNormal, r, canTransmit);
    }
    else if (canTransmit) {
        transmitRay(ray, mat, worldNrm, geomNormal, r[0], inward, canReflect);
    }
    else {
        //no way to proceed -> abort
        result = RESULT_CODE_RAY_ABSORBED;
    }
#else
    if (canReflect) {
        reflectRay(ray, worldNrm, geomNormal, r, false);
    }
    else {
        //no way to proceed -> abort
        result = RESULT_CODE_RAY_ABSORBED;
    }
#endif

    //done
    return result;
}

#endif
