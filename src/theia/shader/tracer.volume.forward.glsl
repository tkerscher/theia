//Check expected macros
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "math.glsl"
#include "ray.propagate.glsl"

layout(scalar) uniform DispatchParams {
    uint batchSize;
};

layout(scalar) uniform TraceParams {
    uvec2 medium;
    int objectId;

    PropagateParams propagation;
} params;

//define global medium
#define USE_GLOBAL_MEDIUM
Medium getMedium() {
    return Medium(params.medium);
}
#include "ray.medium.glsl"

#include "ray.response.glsl"
#include "ray.scatter.glsl"
#include "result.glsl"
#include "util.sample.glsl"

#include "wavelengthsource.common.glsl"
#include "lightsource.common.glsl"
#include "response.common.glsl"
#include "target.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "source.glsl"
#include "photon.glsl"
#include "response.glsl"
#include "target.glsl"

#include "callback.util.glsl"

void createResponse(
    ForwardRay ray,
    const TargetSample hit,
    vec3 dir,
    float weight,
    bool scattered,
    uint idx, inout uint dim
) {
    //check if hit
    if (!hit.valid) return;

    //scatter ray if needed
    if (scattered) {
        scatterRayIS(ray, dir);
    }

    //propagate ray to hit
    ResultCode code = propagateRayToHit(ray, hit.position, hit.normal, params.propagation);
    if (code < 0) return; //out-of-bounds

    //create weighted response
    ray.state.lin_contrib *= weight;
    HitItem item = createHit(
        ray,
        hit.objPosition,
        hit.objNormal,
        params.objectId,
        hit.worldToObj
    );
    if (item.contrib > 0.0)
        response(item, idx, dim);
}

#ifndef DISABLE_MIS

//MIS is a sampling method that combines multiple distributions using weights
//to minimize variance increase. Allows to use specialized distributions (here
//sampling the target sphere) to increase performance. Distributions need to
//cover the variable space only jointly, i.e. they are allowed to assign zero
//probability to a valid value as long as there is at least one that can sample
//it

//MIS: sample both scattering phase function & detector
//also include factors of phase function and sample propability:
//             p_XX^2        p_PX   <- scattering phase function
// w_X = ---------------- * ------
//        p_XX^2 + p_YX^2    p_XX   <- importance sampling
//       \-------V------/
//          MIS weight
//to improve precision, we already reduce the fraction where possible

//calculates the factor converting area prob to solid angle prob
float jacobian_dAdW(vec3 obs, vec3 pos, vec3 nrm) {
    vec3 dir = pos - obs;
    float d2 = dot(dir, dir);
    dir = normalize(dir);
    
    float factor = d2 / abs(dot(dir, nrm));
    //for dot(dir, nrm) near zero we might get inf as factor
    //-> mark as invalid (set to zero)
    if (isinf(factor)) factor = 0.0;

    return factor;
}

void sampleTargetMIS(ForwardRay ray, uint idx, inout uint dim) {
    //Here we'll use the following naming scheme: pXY, where:
    // X: prob, evaluated distribution
    // Y: sampled distribution
    // T: target, P: phase
    //e.g. pTP: p_target(dir ~ phase)

    //shorthand notation
    Medium med = Medium(params.medium);
    vec3 obs = ray.state.position;
    vec3 dir = ray.state.direction;

    //sample phase function
    float pPP;
    vec3 dirPhase = scatter(med, dir, random2D(idx, dim), pPP);
    TargetSample phaseHit = intersectTarget(obs, dirPhase);

    //sample target
    TargetSample targetHit = sampleTarget(obs, idx, dim);
    vec3 dirTarget = normalize(targetHit.position - obs);
    float pTT = targetHit.prob * jacobian_dAdW(obs, targetHit.position, targetHit.normal);

    //calculate cross probabilities
    float pPT = scatterProb(med, dir, dirTarget);
    float pTP = phaseHit.prob * jacobian_dAdW(obs, phaseHit.position, phaseHit.normal);

    //calculate MIS weights
    float wTarget = pTT * pPT / (pTT*pTT + pPT*pPT);
    float wPhase = pPP * pPP / (pPP*pPP + pTP*pTP);

    //create hits
    createResponse(ray, phaseHit, dirPhase, wPhase, true, idx, dim);
    createResponse(ray, targetHit, dirTarget, wTarget, true, idx, dim);
}

#endif

ResultCode trace(
    inout ForwardRay ray,
    uint idx, inout uint dim,
    bool first, bool allowResponse
) {
    //sample distance
    float u = random(idx, dim);
    float dist = sampleScatterLength(ray, params.propagation, u);

    //trace target
    TargetSample hit = intersectTarget(ray.state.position, ray.state.direction);
    //check if we hit
    bool hitValid = hit.valid && (hit.dist <= dist);

    //update ray
    dist = min(hit.dist, dist);
    ResultCode result = propagateRay(ray, dist, params.propagation);
    updateRayIS(ray, dist, params.propagation, hitValid);
    //check if ray is still in bounds
    if (result < 0)
        return result; //abort tracing

    //if we hit the sphere we assume the ray got absorbed -> abort tracing
    //we won't create a hit, as we already created a path of that length
    //in the previous step via MIS.
    //Only exception: first trace step, as we didn't MIS from the light source
    // can be disabled to use a direct light integrator instead
    // (partition of the path space)
    if (hitValid && allowResponse) {
        //Align hit, i.e. rotate polRef to plane of incidence
        alignRayToHit(ray, hit.normal);
        HitItem item = createHit(
            ray,
            hit.objPosition,
            hit.objNormal,
            params.objectId,
            hit.worldToObj
        );
        if (item.contrib > 0.0)
            response(item, idx, dim);

        return RESULT_CODE_RAY_DETECTED;
    }
    else if (hitValid) {
        //hit target, but response for this length was already sampled
        //-> abort without creating a response
        return RESULT_CODE_RAY_ABSORBED;
    }

    #ifndef DISABLE_MIS
    sampleTargetMIS(ray, idx, dim);
    #endif

    //no hit -> scatter
    return RESULT_CODE_RAY_SCATTERED;
}

//process macro flags
#if defined(DISABLE_MIS) && !defined(DISABLE_DIRECT_LIGHTING)
#define DIRECT_LIGHTING true
#else
#define DIRECT_LIGHTING false
#endif

#ifndef DISABLE_MIS
#define ALLOW_RESPONSE false
#else
#define ALLOW_RESPONSE true
#endif

ForwardRay sampleRay(uint idx, inout uint dim) {
    WavelengthSample photon = sampleWavelength(idx, dim);
    Medium medium = Medium(params.medium);
    MediumConstants constants = lookUpMedium(medium, photon.wavelength);
    SourceRay lightRay = sampleLight(photon.wavelength, constants, idx, dim);
    return createRay(lightRay, medium, constants, photon);
}

void traceMain() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= batchSize)
        return;

    //sample ray
    ForwardRay ray = sampleRay(idx, dim);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
    //discard ray if inside target
    if (isOccludedByTarget(ray.state.position)) {
        onEvent(ray, ERROR_CODE_TRACE_ABORT, idx, 0);
        return;
    }

    //try to extend first ray to sphere if direct lighting and MIS is enabled
    #if !defined(DISABLE_DIRECT_LIGHTING) && !defined(DISABLE_MIS)
    TargetSample directHit = intersectTarget(ray.state.position, ray.state.direction);
    createResponse(ray, directHit, ray.state.direction, 1.0, false, idx, dim);
    #endif

    //trace loop: first iteration
    // special as it is allowed to create a direct hit (there was no MIS yet)
    ResultCode result = trace(ray, idx, dim, true, DIRECT_LIGHTING);
    onEvent(ray, result, idx, 1);
    if (result < 0)
        return;
    
    //trace loop: rest
    [[unroll]] for (uint i = 0; i < PATH_LENGTH; ++i) {
        //scatter ray
        vec2 u = random2D(idx, dim);
        float cos_theta, phi;
        scatterRay(ray, u);

        //trace ray
        ResultCode result = trace(ray, idx, dim, false, ALLOW_RESPONSE);
        onEvent(ray, result, idx, i + 2);

        //stop codes are negative
        if (result < 0)
            return;
    }

    //finished trace loop, but could go further
    onEvent(ray, RESULT_CODE_MAX_ITER, idx, PATH_LENGTH + 2);
}

void main() {
    initResponse();
    traceMain();
    finalizeResponse();
}
