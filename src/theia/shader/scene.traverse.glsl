#ifndef _INCLUDE_SCENE_TRAVERSE
#define _INCLUDE_SCENE_TRAVERSE

#include "material.glsl"
#include "ray.propagate.glsl"
#include "result.glsl"
#include "scatter.surface.glsl"
#include "scatter.volume.glsl"
#include "scene.intersect.glsl"
#ifdef POLARIZATION
#include "polarization.glsl"
#endif

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
void crossBorder(inout Ray ray, Material mat, vec3 rayNrm, bool inward) {
    //put ray on other side to prevent self intersection
    ray.position = offsetRay(ray.position, -rayNrm);
    //update medium & constants
    Medium medium = inward ? mat.inside : mat.outside;
    ray.medium = uvec2(medium);
    ray.constants = lookUpMedium(medium, ray.wavelength);
}

//reflects ray and updates contribution taking IS and reflectance into account
//worldNrm MUST be point in oppsoite direction than ray.direction
void reflectRay(
    inout Ray ray,
    vec3 rayNrm,
    float r, //reflectance
    bool canTransmit //needed for correction factor caused by importance sampling  
) {
    //update ray
    ray.position = offsetRay(ray.position, rayNrm); //offset to prevent self intersection
    ray.direction = normalize(reflect(ray.direction, rayNrm));

    //update contribution
    // - attenuate contrib by reflection
    // - if canTransmit is true, reflection was IS -> cancels attenuation
    if (!canTransmit) {
        ray.lin_contrib *= r;
    }
}

//transmits ray; updates ray, medium and contribution:
// - crosses border into new medium
// - refracts direction
// - takes transmission into account
// - takes IS into account
//worldNrm MUST be point in oppsoite direction than ray.direction
void transmitRay(
    inout Ray ray,
    Material mat,
    vec3 rayNrm,
    float r, //reflectance
    bool inward, //which direction to cross interface
    bool canReflect, //needed for IS factor
    bool forward //true if we transport photons, false for importance particles
) {
    //cross border, but remember both refractive indices
    float n_i = ray.constants.n;
    crossBorder(ray, mat, rayNrm, inward);
    float n_o = ray.constants.n;

    //calculate new direction
    float eta = n_i / n_o;
    ray.direction = normalize(refract(ray.direction, rayNrm, eta));

    //attenuate contribution by transmission
    //if canReflect is true, transmission was IS -> cancels attenuation
    if (!canReflect) {
        ray.lin_contrib *= (1.0 - r);
    }

    //if transporting importance particles, we need to apply an extra factor
    //eta^-2. See PBRT or Veach thesis
    if (!forward) {
        float inv_eta = n_o / n_i;
        ray.lin_contrib *= inv_eta * inv_eta;
    }
}

//creates a response using the provided hit data
void createResponse(
    const Ray ray, const SurfaceHit hit,
    float maxTime,
    bool absorb, float weight
) {
    #ifdef POLARIZATION
    vec4 stokes = ray.stokes;
    #endif

    float contrib = weight * ray.lin_contrib * exp(ray.log_contrib);
    if (!absorb) {
        //calculate reflectance
        float r_s, r_p, n_t;
        fresnelReflect(
            hit.material,
            ray.wavelength, ray.constants.n,
            ray.direction, hit.worldNrm,
            r_s, r_p, n_t
        );
        float r = 0.5 * (r_s*r_s + r_p*r_p);
        //attenuate by transmission
        contrib *= 1.0 - r;

        #ifdef POLARIZATION
        //apply transmission matrix
        float t_s = r_s + 1.0;
        float t_p = (r_p + 1.0) * ray.constants.n / n_t;
        stokes = polarizerMatrix(t_p, t_s) * stokes;
        #endif
    }

    #ifdef POLARIZATION
    //normalize stokes
    contrib *= stokes.x;
    stokes /= stokes.x;

    //create polarization reference frame in local space
    //we just need to match the plane of incidence, since stokes vector are
    //symmetric under rotation of 180deg this is already enough
    vec3 hitPolRef = crosser(hit.objDir, hit.objNrm);
    //degenerate case: dir || normal
    // -> just choose any perpendicular to objDir
    float len = length(hitPolRef);
    if (len > 1e-5) {
        hitPolRef /= len;
    }
    else {
        //first two columns of local cosy trafo are vectors perpendicular to hitDir
        hitPolRef = createLocalCOSY(hit.objDir)[0];
    }
    #endif

    //ignore zero contrib samples
    if (contrib != 0.0 && ray.time <= maxTime) {
        //create response
        response(HitItem(
            hit.objPos, hit.objDir, hit.objNrm,
            #ifdef POLARIZATION
            stokes, hitPolRef,
            #endif
            ray.wavelength,
            ray.time,
            contrib
        ));
    }
}

/**
 * Processes a surface hit.
 *  - Handles transmission / Reflection / Volume Borders
 *    -> Nodges the ray's position to prevent self intersection
 *  - Optionally creates response if allowResponse is true
 *    (disabled if MIS is enabled as that should never happen in that case)
 *  - Result code indicates wether tracing should be aborted
 * Afterwards the ray is ready to be further traced
 *
 * Note: No ray propagation happens here, i.e. things like updating elapsed time 
*/
ResultCode processHit(
    inout Ray ray,          ///< Ray to update
    const SurfaceHit hit,   ///< Hit to process
    uint targetId,          ///< CustomId of target
    float maxTime,          ///< max elapsed time allowed for response
    float u,                ///< random number for transmit/reflect decision
    #ifdef EXPORT_MUELLER
    out mat4 mueller,       ///< Mueller matrix describing surface interaction
    #endif
    bool allowResponse,     ///< Whether detector hits create responses
    bool forward            ///< Whether photons (true) or importance (false) is transported
) {
    //Optional polarization stuff
    #ifndef EXPORT_MUELLER
    mat4 mueller;
    #endif
    #ifdef POLARIZATION
    //init mueller to unity
    mueller = mat4(1.0);
    //init space for fresnel mueller matrix (applied after branch reconverge)
    mat4 muellerFresnel = mat4(1.0);
    #endif

    //default result
    ResultCode result = RESULT_CODE_RAY_HIT;
    
    //update ray position with more accurate value
    ray.position = hit.worldPos;

    //process flags
    bool isAbs = (hit.flags & MATERIAL_BLACK_BODY_BIT) != 0;
    bool isDet = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    bool volBorder = (hit.flags & MATERIAL_VOLUME_BORDER_BIT) != 0;
    bool canTransmit = (hit.flags & MATERIAL_NO_TRANSMIT_BIT) == 0;
    bool canReflect = (hit.flags & MATERIAL_NO_REFLECT_BIT) == 0;
    
    #ifdef POLARIZATION
    //rotate polarization reference to be perpendicular to plane of incidence
    //eager apply mueller matrix for any responses
    vec3 polRef;
    mueller = rotatePolRef(ray.direction, ray.polRef, hit.rayNrm, polRef);
    ray.polRef = polRef;
    if (forward) {
        //we only care about the stokes vector when we actually carry light
        ray.stokes = mueller * ray.stokes;
    }
    #endif

    //create response if allowed
    if (allowResponse && isDet && hit.customId == targetId) {
        createResponse(
            ray, hit,
            maxTime,
            isAbs,
            1.0
        );
        //dont return early as the ray is allowed to carry on
        //(e.g. partially reflect)
        result = RESULT_CODE_RAY_DETECTED;
    }

    //stop tracing if ray absorbed
    if (isAbs) return RESULT_CODE_RAY_ABSORBED;

    #ifndef DISABLE_VOLUME_BORDER
    //check for volume border
    if (volBorder) {
        crossBorder(ray, hit.material, hit.rayNrm, hit.inward);
        return RESULT_CODE_VOLUME_HIT;
    }
    #endif

    //calculate reflectance
    float r_s, r_p, n_t;
    fresnelReflect(
        hit.material,
        ray.wavelength, ray.constants.n,
        ray.direction, hit.worldNrm,
        r_s, r_p, n_t
    );
    float r = 0.5 * (r_s*r_s + r_p*r_p);

    #ifndef DISABLE_TRANSMISSION
    //handle reflection/transmission
    if (canReflect && (!canTransmit || u < r)) {
        #ifdef POLARIZATION
        muellerFresnel = polarizerMatrix(r_p, r_s);
        #endif
        reflectRay(ray, hit.rayNrm, r, canTransmit);
    }
    else if (canTransmit) {
        #ifdef POLARIZATION
        float t_s = r_s + 1.0;
        float t_p = (r_p + 1.0) * ray.constants.n / n_t;
        muellerFresnel = polarizerMatrix(t_p, t_s);
        #endif

        transmitRay(ray, hit.material, hit.rayNrm, r, hit.inward, canReflect, forward);
    }
    else {
        //no way to proceed -> abort
        result = RESULT_CODE_RAY_ABSORBED;
    }

    #else //ifndef DISABLE_TRANSMISSION

    if (canReflect) {
        #ifdef POLARIZATION
        muellerFresnel = polarizerMatrix(r_p, r_s);
        #endif
        reflectRay(ray, hit.rayNrm, r, false);
    }
    else {
        //no way to proceed -> abort
        result = RESULT_CODE_RAY_ABSORBED;
    }

    #endif //ifndef DISABLE_TRANSMISSION

    #ifdef POLARIZATION
    //chain polarizer matrix to rotation matrix
    if (forward) {
        mueller = muellerFresnel * mueller;
        //apply polarizer matrix (rotation already applied)
        ray.stokes = muellerFresnel * ray.stokes;
    }
    else {
        //backwards, we also need to assemble the mueller matrix backwards,
        //plus rotating in the opposite direction (sampled new->old)
        //since rotating matrix is orthogonal we can use the transpose
        mueller = transpose(mueller) * muellerFresnel;
        //Ignore stokes in backwards mode
    }
    #endif

    //done
    return result;
}

//traces a shadow ray and generates a response if target is hit
// - checks if target is in dir direction
// - dist should be a safe lower bound on the distance to check
void processShadowRay(
    Ray ray, vec3 dir,
    const SurfaceHit hit,
    uint targetId,
    PropagateParams params,
    float weight
) {
    //check if we hit the target
    bool isDet = (hit.flags & MATERIAL_DETECTOR_BIT) != 0;
    if (!hit.valid || hit.customId != targetId || !isDet)
        return;
    
    //update ray; dont use propagate, as ray.direction is wrong
    //also scatter=false, as mu_s was already applied
    float dist = distance(hit.worldPos, ray.position);
    ResultCode result = updateRay(ray, dist, params, false);
    if (result < 0) return;

    #ifdef POLARIZATION
    //update polarization locally (ray is a copy anyway)
    // rotate to scatter plane -> phase matrix -> rotate to plane of incidence
    vec3 polRef;
    //rotate to scatter plane
    ray.stokes = rotatePolRef(ray.direction, ray.polRef, dir, polRef) * ray.stokes;
    ray.polRef = polRef;
    //phase matrix
    Medium medium = Medium(ray.medium);
    float cos_theta = dot(ray.direction, dir);
    ray.stokes = lookUpPhaseMatrix(medium, cos_theta) * ray.stokes;
    //rotate to plane of incidence
    ray.stokes = rotatePolRef(dir, ray.polRef, hit.rayNrm, polRef) * ray.stokes;
    ray.polRef = polRef;
    #endif

    //create response
    bool black = (hit.flags & MATERIAL_BLACK_BODY_BIT) != 0;
    createResponse(
        ray, hit,
        params.maxTime,
        black, weight
    );
}
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

    //process hit
    SurfaceHit hit;
    vec3 tmp = ray.direction;
    ray.direction = dir;    // temporarly set actual direction for ray query process
    ResultCode result = processRayQuery(ray, rayQuery, hit);
    if (result < 0) return; //successfull? (any hits?)
    ray.direction = tmp;    //Restore old state for correct shadow ray processing

    //process shadow ray
    processShadowRay(ray, dir, hit, targetId, params, weight);
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
    #ifdef EXPORT_MUELLER
    out mat4 mueller,
    #endif
    PropagateParams params,
    bool forward
) {
    Medium medium = Medium(ray.medium);
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
        medium,
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
    #endif //ifndef SCENE_TRAVERSE_DISABLE_MIS

    //sample phase function for new ray direction
    float cos_theta, phi;
    vec2 u = random2D(idx, dim); dim += 2;
    sampleScatterDir(medium, ray.direction, u, cos_theta, phi);
    vec3 scatteredDir = scatterDir(ray.direction, cos_theta, phi);

    //update polarization
    #ifndef EXPORT_MUELLER
    mat4 mueller;
    #endif
    #ifdef POLARIZATION
    //assemble mueller matrix
    mat4 phaseMatrix = lookUpPhaseMatrix(medium, cos_theta);
    vec3 polRef;
    // mat4 rotate = rotatePolRef(phi);
    mat4 rotate = rotatePolRef(ray.direction, ray.polRef, scatteredDir, polRef);
    //update polarization reference
    ray.polRef = polRef;
    //apply mueller matrix
    if (forward) {
        mueller = phaseMatrix * rotate;
        ray.stokes = mueller * ray.stokes;
    }
    else {
        //backwards, we also need to assemble the mueller matrix backwards,
        //plus rotating in the opposite direction (sampled new->old)
        //since rotating matrix is orthogonal we can use the transpose
        mueller = transpose(rotate) * phaseMatrix;
        //ignore stokes vector in backwards mode
    }
    #endif //ifdef POLARIZATION

    //update ray
    ray.direction = scatteredDir;
}

//tracing rng stride
#ifndef SCENE_TRAVERSE_DISABLE_MIS
#define SCENE_TRAVERSE_TRACE_RNG_STRIDE 8
#else
#define SCENE_TRAVERSE_TRACE_RNG_STRIDE 4
#endif
#define SCENE_TRAVERSE_TRACE_OFFSET 1

//checks whether ray contains any nan or inf
bool isRayBad(const Ray ray) {
    return 
        any(isnan(ray.position)) ||
        any(isinf(ray.position)) ||
        any(isnan(ray.direction)) ||
        any(isinf(ray.direction)) ||
        length(ray.direction) <= 0.0;
}

/*
 * Traces the given ray and propagates it. Creates a SurfaceHit describing if
 * a surface was hit and holds additional information about the hit.
 *
 * If MIS is enabled and allowResponse=true, tries to extend the ray to the
 * detector to create a response.
*/
ResultCode trace(
    inout Ray ray,
    out SurfaceHit hit,
    uint targetId,
    uint idx, uint dim,
    PropagateParams params,
    bool allowResponse
) {
    //health check ray
    if (isRayBad(ray))
        return ERROR_CODE_RAY_BAD;
    vec3 dir = normalize(ray.direction);

    //sample distance
    float u = random(idx, dim); dim++;
    float dist = sampleScatterLength(ray, params, u);

    //next event estimate target by extending ray if possible
    //only if allowResponse is true
    #ifndef SCENE_TRAVERSE_DISABLE_MIS
    float sampledDist = dist;
    Sphere target = targets[targetId];
    //check if we have a chance on hitting the detector by extending ray
    bool mis_target = allowResponse && !isinf(intersectSphere(target, ray.position, dir));
    if (mis_target) {
        //extend ray to detector
        dist = distance(ray.position, target.position) + target.radius;
        //check if we actually extended the ray
        //disable shadow ray if not
        mis_target = dist > sampledDist;
        dist = max(dist, sampledDist);
    }
    #endif

    //trace ray against scene
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit everything
        ray.position,
        0.0, //t_min; self-intersections handled via offsets
        dir,
        dist);
    rayQueryProceedEXT(rayQuery);
    ResultCode result = processRayQuery(ray, rayQuery, hit);
    if (result <= ERROR_CODE_MAX_VALUE)
        return result;

    //fetch actual travelled distance if we hit anything
    if (hit.valid) {
        dist = distance(ray.position, hit.worldPos);
    }

    #ifndef SCENE_TRAVERSE_DISABLE_MIS
    //Check if hit was actually our shadow ray
    if (mis_target && hit.valid && dist > sampledDist) {
        //create response
        processShadowRay(ray, dir, hit, targetId, params, 1.0);
        //act like we didn't actually hit anything
        hit.valid = false;
        dist = sampledDist;
    }
    #endif
    
    //propagate ray (ray, dist, params, scatter)
    result = propagateRay(ray, dist, params, false);
    updateRayIS(ray, dist, params, hit.valid);

    //done
    return result;
}

/**
 * Processes the interaction indicated by hit, i.e. either surface hit or
 * volume scatter.
 *  - Samples surface interaction, i.e. transmission/reflection.
 *  - Detector response enabled via allowResponse.
 *  - If MIS enabled and allowResponse, samples detector to create shadow rays 
 *  - forward toggles between photon (true) and importance particle (false) tracing
*/
ResultCode processInteraction(
    inout Ray ray,
    const SurfaceHit hit,
    uint targetId,
    uint idx, uint dim,
    PropagateParams params,
    #ifdef EXPORT_MUELLER
    out mat4 mueller,
    #endif
    bool allowResponse,
    bool forward,
    bool last
) {
    #ifdef EXPORT_MUELLER
    //default mueller matrix to identity
    mueller = mat4(1.0);
    #endif
    
    ResultCode result;
    //handle either intersection or scattering
    if (hit.valid) {
        float u = random(idx, dim); dim++;
        result = processHit(
            ray, hit,
            targetId,
            params.maxTime,
            u,
            #ifdef EXPORT_MUELLER
            mueller,
            #endif
            allowResponse,
            forward
        );
    }
    else {
        //dont bother scattering on the last iteration (we wont hit anything)
        //this also prevents MIS to sample paths one longer than PATH_LENGTH
        if (!last)
            processScatter(
                ray,
                targetId,
                idx, dim + 1,
                #ifdef EXPORT_MUELLER
                mueller,
                #endif                
                params,
                forward
            );
        result = RESULT_CODE_RAY_SCATTERED;
    }

    return result;
}

#endif
