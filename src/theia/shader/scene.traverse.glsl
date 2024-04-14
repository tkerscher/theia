#ifndef _INCLUDE_SCENE_TRAVERSE
#define _INCLUDE_SCENE_TRAVERSE

#include "material.glsl"
#include "ray.propagate.glsl"
#include "result.glsl"
#include "scatter.surface.glsl"
#include "scatter.volume.glsl"
#include "scene.intersect.glsl"

#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "response.glsl"

uniform accelerationStructureEXT tlas;

#ifndef SCENE_TRAVERSE_DISABLE_MIS

#include "sphere.intersect.glsl"
#include "tracer.mis.glsl"

layout(scalar) readonly buffer Targets {
    Sphere targets[];
};

#endif

//pushes ray across border:
// - offsets ray to prevent self intersection on correct side
//   (assumes ray's current position is on interface)
// - updates ray medium
void crossBorder(inout Ray ray, Material mat, vec3 geomNormal, bool inward) {
    //put ray on other side to prevent self intersection
    ray.position = offsetRay(ray.position, -geomNormal);
    //update medium & constants
    Medium medium = inward ? mat.inside : mat.outside;
    ray.medium = uvec2(medium);
    ray.constants = lookUpMedium(medium, ray.wavelength);
}

//reflects ray and updates contribution taking IS and reflectance into account
void reflectRay(
    inout Ray ray,
    vec3 worldNrm, vec3 geomNrm,
    float r, //reflectance
    bool canTransmit //needed for correction factor caused by importance sampling  
) {
    //update ray
    ray.position = offsetRay(ray.position, geomNrm); //offset to prevent self intersection
    ray.direction = normalize(reflect(ray.direction, worldNrm));

    float p = canTransmit ? r : 1.0; //IS prob
    ray.lin_contrib *= r / p;
}

//transmits ray; updates ray, medium and contribution:
// - crosses border into new medium
// - refracts direction
// - takes transmission into account
// - takes IS into account
void transmitRay(
    inout Ray ray,
    Material mat,
    vec3 worldNrm, vec3 geomNormal,
    float r, //reflectance
    bool inward, bool canReflect //needed for IS factor
) {
    //cross border, but remember both refractive indices
    float n_i = ray.constants.n;
    crossBorder(ray, mat, geomNormal, inward);
    float n_o = ray.constants.n;

    //calculate new direction
    float eta = n_i / n_o;
    ray.direction = normalize(refract(ray.direction, worldNrm, eta));

    //update sample contribution:
    //the factor (1-r) cancels with importance sampling the reflection leaving
    //only a factor of eta^-2 for the radiance due to rays getting denser
    //TODO: the factor eta^-2 only appears for camera rays
    //float inv_eta = n_o / n_i;
    //ray.lin_contrib *= inv_eta * inv_eta;
    if (!canReflect) {
        //no random choice happen -> need to also account for transmission factor
        ray.lin_contrib *= (1.0 - r);
    }
}

//creates a response using the provided hit data
void createResponse(
    const Ray ray, Material mat,
    float maxTime,
    vec3 objPos, vec3 objDir, vec3 objNrm, vec3 worldNrm,
    bool absorb, float weight
) {
    float contrib = weight * ray.lin_contrib * exp(ray.log_contrib);
    if (!absorb) {
        contrib *= (1.0 - reflectance(
            mat, ray.wavelength, ray.constants.n, ray.direction, worldNrm
        ));
    }

    //ignore zero contrib samples
    if (contrib != 0.0 && ray.time <= maxTime) {
        //create response
        response(HitItem(
            objPos, objDir, objNrm,
            ray.wavelength,
            ray.time,
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
    //TODO: This is not symmetric, i.e. different for light and camera paths
    float cosWorld = abs(dot(ray.direction, worldNrm));
    float cosGeom = abs(dot(ray.direction, geomNormal));
    ray.lin_contrib *= cosWorld * cosGeom;
    
    //process flags
    uint flags = inward ? mat.flagsInwards : mat.flagsOutwards;
    bool isDet = (flags & MATERIAL_DETECTOR_BIT) != 0;
    bool black = (flags & MATERIAL_BLACK_BODY_BIT) != 0;
    bool canTransmit = (flags & MATERIAL_NO_TRANSMIT_BIT) == 0;
    bool canReflect = (flags & MATERIAL_NO_REFLECT_BIT) == 0;

    //calculate reflectance
    float r = reflectance(
        mat,
        ray.wavelength,
        ray.constants.n,
        ray.direction,
        worldNrm
    );

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
    if (canReflect && (!canTransmit || u < r)) {
        reflectRay(ray, worldNrm, geomNormal, r, canTransmit);
    }
    else if (canTransmit) {
        transmitRay(ray, mat, worldNrm, geomNormal, r, inward, canReflect);
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

//traces a shadow ray and generates a response if target is hit
// - checks if target is in dir direction
// - dist should be a safe lower bound on the distance to check
void traceShadowRay(
    Ray ray,
    vec3 dir, float dist,
    uint targetId,
    PropagateParams params,
    float weight
) {
    //trace ray against scene
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit everything
        ray.position,
        0.0, dir, dist);
    rayQueryProceedEXT(rayQuery);

    //check if we hit anything
    bool hit = rayQueryGetIntersectionTypeEXT(rayQuery, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT;
    if (!hit) return;
    //process hit
    vec3 objPos, objNrm, objDir, worldPos, worldNrm, geomNormal;
    Material mat;
    bool inward;
    ResultCode result = processRayQuery(
        ray, rayQuery,
        mat, inward,
        objPos, objNrm, objDir,
        worldPos, worldNrm,
        geomNormal
    );
    dist = distance(worldPos, ray.position);
    uint flags = inward ? mat.flagsInwards : mat.flagsOutwards;
    if (result < 0) return;

    //check if we hit the target
    int customId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    if (customId != targetId || (flags & MATERIAL_DETECTOR_BIT) != 0)
        return;
    
    //propagate ray
    result = propagateRay(ray, dist, params, false);
    if (result < 0) return;

    //create response
    bool black = (flags & MATERIAL_BLACK_BODY_BIT) != 0;
    createResponse(
        ray, mat,
        params.maxTime,
        objPos, objDir, objNrm, worldNrm,
        black, weight
    );
}

//processes a volume scatter event produced by a trace
// - samples new ray direction
// - MIS shadow rays if not disabled via SCENE_TRAVERSE_DISABLE_MIS
//   (shadow rays can generate responses)
//NOTE: in case MIS is enabled, it expects a global accessible array
//Sphere targets[] where it can fetch the target sphere for MIS
void processScatter(
    inout Ray ray,
    uint targetId,
    uint idx, uint dim,
    PropagateParams params
) {
    //apply scatter coefficient
    ray.lin_contrib *= ray.constants.mu_s;

#ifndef SCENE_TRAVERSE_DISABLE_MIS
    //MIS detector:
    //we both sample the scattering phase function as well as the detector
    //sphere for a possible hit direction
    float wTarget, wPhase;
    vec3 dirTarget, dirPhase;
    vec2 uTarget = random2D(idx, dim), uPhase = random2D(idx, dim + 2); dim += 4;
    Sphere target = targets[targetId]; //NOTE: external dependency !!!
    sampleTargetMIS(
        Medium(ray.medium),
        ray.position, ray.direction, target,
        uTarget, uPhase,
        wTarget, dirTarget,
        wPhase, dirPhase
    );
    //check if phase sampling even has a chance of hitting the detector
    //TODO: check if this test actually reduce pressure on ray tracing hardware
    //      otherwise this only introduce divergence overhead
    bool phaseMiss = isinf(intersectSphere(target, ray.position, dirPhase));
    //trace shadow ray & create hit if visible
    //(dist is max distance checked for hit: dist to back of target sphere)
    float dist = distance(ray.position, target.position) + target.radius;
    traceShadowRay(ray, dirTarget, dist, targetId, params, wTarget);
    if (!phaseMiss)
        traceShadowRay(ray, dirPhase, dist, targetId, params, wPhase);
#endif

    //sample phase function for new ray direction
    float _ignore;
    vec2 u = random2D(idx, dim); dim += 2;
    ray.direction = scatter(Medium(ray.medium), ray.direction, u, _ignore);
}

//tracing rng stride
#ifndef SCENE_TRAVERSE_DISABLE_MIS
#define SCENE_TRAVERSE_TRACE_RNG_STRIDE 8
#else
#define SCENE_TRAVERSE_TRACE_RNG_STRIDE 4
#endif

//checks whether ray contains any nan or inf
bool isRayBad(const Ray ray) {
    return 
        any(isnan(ray.position)) ||
        any(isinf(ray.position)) ||
        any(isnan(ray.direction)) ||
        any(isinf(ray.direction)) ||
        length(ray.direction) <= 0.0;
}

//traces the given ray by propagating it and preparing the next trace:
// - samples a travel distance
// - intersects scene & handles interactions (e.g. reflection, detector hit)
// - prepares next ray: either volume scatter or sample scene interaction
// - detector response can be toggled via allowResponse
// - no MIS if last=true
// - return code indicates whether tracing can proceed
ResultCode trace(
    inout Ray ray,
    uint targetId,
    uint idx, uint dim,
    PropagateParams params,
    bool allowResponse, bool last
) {
    //health check ray
    if (isRayBad(ray))
        return ERROR_CODE_RAY_BAD;

    //sample distance
    float u = random(idx, dim); dim++;
    float dist = sampleScatterLength(ray, params, u);

    //trace ray against scene
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit everything
        ray.position,
        0.0, //t_min; self-intersections handled via offsets
        normalize(ray.direction),
        dist);
    rayQueryProceedEXT(rayQuery);
    //check if we hit anything
    bool hit = rayQueryGetIntersectionTypeEXT(rayQuery, true) ==
        gl_RayQueryCommittedIntersectionTriangleEXT;
    
    //fetch actual travelled distance if we hit anything
    if (hit) {
        dist = rayQueryGetIntersectionTEXT(rayQuery, true);
    }
    //propagate ray (ray, dist, params, scatter)
    ResultCode result = propagateRay(ray, dist, params, false);
    updateRayIS(ray, dist, params, hit);
    //check if propagation was sucessfull
    if (result < 0)
        return result; //abort tracing
    
    //handle either intersection or scattering
    if (hit) {
        u = random(idx, dim); dim++;
        result = processHit(
            ray, rayQuery,
            targetId,
            params.maxTime,
            u, allowResponse
        );
    }
    else {
        //dont bother scattering on the last iteration (we wont hit anything)
        //this also prevents MIS to sample paths one longer than PATH_LENGTH
        if (!last)
            processScatter(ray, targetId, idx, dim + 1, params);
        result = RESULT_CODE_RAY_SCATTERED;
    }

    //done
    return result;
}

#endif
