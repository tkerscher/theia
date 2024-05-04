#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require

//Check expected macros
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef PATH_LENGTH
#error "PATH_LENGTH not defined"
#endif
#ifndef DIM_OFFSET
#error "DIM_OFFSET not defined"
#endif
//#samples per iteration
#ifndef DISABLE_MIS
#define DIM_STRIDE 7
#else
#define DIM_STRIDE 3
#endif
//alter ray layout
#define USE_GLOBAL_MEDIUM

layout(local_size_x = BLOCK_SIZE) in;

#include "material.glsl"
#include "sphere.intersect.glsl"
#include "ray.propagate.glsl"
#include "ray.sample.glsl"
#include "result.glsl"
#include "tracer.mis.glsl"
#include "util.branch.glsl"

#ifdef POLARIZATION
#include "polarization.glsl"
#endif

#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "source.glsl"
#include "response.glsl"

layout(scalar) uniform TraceParams {
    Sphere target;
    uvec2 medium;

    PropagateParams propagation;
} params;

void createHit(Ray ray, vec3 dir, float weight, bool scattered) {
    //try to hit target: dist == inf if no hit
    float dist = intersectSphere(params.target, ray.position, dir);
    if (isinf(dist))
        return; // missed target -> no response

    #ifdef POLARIZATION
    //update polarization if needed
    if (scattered) {
        vec3 polRef;
        float cos_theta = dot(ray.direction, dir);
        Medium medium = Medium(params.medium);
        ray.stokes = lookUpPhaseMatrix(medium, cos_theta)
            * rotatePolRef(ray.direction, ray.polRef, dir, polRef) * ray.stokes;
        ray.polRef = polRef;
    }
    #endif

    //update ray (local copy)
    ray.direction = dir;
    ResultCode code = propagateRay(ray, dist, params.propagation, true);
    if (code < 0)
        return; //lost ray or time ran out -> create no response

    //calculate hit coordinates
    vec3 normal = normalize(ray.position - params.target.position);
    vec3 hitPos = normal * params.target.radius;
    //create hit
    if (ray.time <= params.propagation.maxTime) {
            #ifdef POLARIZATION
            //rotate polRef to plane of incidence
            vec3 polRef;
            vec4 stokes = rotatePolRef(dir, ray.polRef, normal, polRef) * ray.stokes;
            //normalize stokes
            float contrib = ray.lin_contrib * exp(ray.log_contrib);
            contrib *= stokes.x;
            stokes /= ray.stokes.x;
            response(HitItem(
                hitPos, dir, normal,
                //since we only translated the target, polRef are same in world and object space
                stokes, polRef,
                ray.wavelength,
                ray.time,
                contrib
            ));
            #else
            response(HitItem(
                hitPos, dir, normal,
                ray.wavelength,
                ray.time,
                ray.lin_contrib * exp(ray.log_contrib)
            ));
            #endif
    }
}

ResultCode trace(inout Ray ray, uint idx, uint dim, bool first, bool allowResponse) {
    //sample distance
    float u = random(idx, dim); dim++;
    float dist = sampleScatterLength(ray, params.propagation, u);

    //trace sphere
    float t = intersectSphere(params.target, ray.position, ray.direction);
    //check if we hit
    bool hit = t <= dist;

    //update ray (ray, dist, params, scatter)
    //only on the first trace the ray has not scattered
    dist = min(t, dist);
    ResultCode result = propagateRay(ray, dist, params.propagation, !first);
    updateRayIS(ray, dist, params.propagation, hit);
    //check if ray is still in bounds
    if (result < 0)
        return result; //abort tracing

    //if we hit the sphere we assume the ray got absorbed -> abort tracing
    //we won't create a hit, as we already created a path of that length
    //in the previous step via MIS.
    //Only exception: first trace step, as we didn't MIS from the light source
    // can be disabled to use a direct light integrator instead
    // (partition of the path space)
    if (hit && allowResponse) {
        //calculate local coordinates (dir is the same)
        vec3 hitNormal = normalize(ray.position - params.target.position);
        vec3 hitPos = hitNormal * params.target.radius;
        //create response
        if (ray.time <= params.propagation.maxTime) {
            #ifdef POLARIZATION
            //rotate polRef to plane of incidence
            vec3 polRef;
            vec4 stokes = rotatePolRef(ray.direction, ray.polRef, hitNormal, polRef) * ray.stokes;
            //normalize stokes
            float contrib = ray.lin_contrib * exp(ray.log_contrib);
            contrib *= stokes.x;
            stokes /= ray.stokes.x;
            response(HitItem(
                hitPos, ray.direction, hitNormal,
                stokes, polRef,
                ray.wavelength,
                ray.time,
                contrib
            ));
            #else
            response(HitItem(
                hitPos, ray.direction, hitNormal,
                ray.wavelength,
                ray.time,
                ray.lin_contrib * exp(ray.log_contrib)
            ));
            #endif
        }
        return RESULT_CODE_RAY_DETECTED;
    }
    if (hit) {
        return RESULT_CODE_RAY_ABSORBED;
    }

    //apply scattering coefficient
    ray.lin_contrib *= ray.constants.mu_s;

    #ifndef DISABLE_MIS
    //MIS detector:
    //we both sample the scattering phase function as well as the detector
    //sphere for a possible hit direction
    float wTarget, wPhase;
    vec3 dirTarget, dirPhase;
    vec2 uTarget = random2D(idx, dim), uPhase = random2D(idx, dim + 2); dim += 4;
    sampleTargetMIS(
        Medium(params.medium),
        ray.position, ray.direction, params.target,
        uTarget, uPhase,
        wTarget, dirTarget,
        wPhase, dirPhase
    );
    //create hits
    createHit(ray, dirTarget, wTarget, true);
    createHit(ray, dirPhase, wPhase, true);
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

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;

    //sample ray
    Medium medium = Medium(params.medium);
    Ray ray = createRay(sampleLight(idx), medium);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
    //discard ray if inside target
    if (distance(ray.position, params.target.position) <= params.target.radius) {
        onEvent(ray, ERROR_CODE_TRACE_ABORT, idx, 0);
        return;
    }

    //advance rng by amount used by the source
    uint dim = DIM_OFFSET;

    //try to extend first ray to sphere if direct lighting and MIS is enabled
    #if !defined(DISABLE_DIRECT_LIGHTING) && !defined(DISABLE_MIS)
    createHit(ray, ray.direction, 1.0, false);
    #endif

    //trace loop: first iteration
    // special as it is allowed to create a direct hit (there was no MIS yet)
    ResultCode result = trace(ray, idx, dim, true, DIRECT_LIGHTING);
    onEvent(ray, result, idx, 1);
    if (result < 0)
        return;
    
    //trace loop: rest
    [[unroll]] for (uint i = 0; i < PATH_LENGTH; ++i, dim += DIM_STRIDE) {
        //scatter ray: Importance sample phase function -> no change in contrib
        vec2 u = random2D(idx, dim);
        float cos_theta, phi;
        sampleScatterDir(medium, ray.direction, u, cos_theta, phi);
        vec3 newDir = scatterDir(ray.direction, cos_theta, phi);
        #ifdef POLARIZATION
        //update polarization
        vec3 polRef;
        ray.stokes = lookUpPhaseMatrix(medium, cos_theta)
            * rotatePolRef(ray.direction, ray.polRef, newDir, polRef)
            * ray.stokes;
        ray.polRef = polRef;
        // ray.stokes = lookUpPhaseMatrix(medium, cos_theta) * rotatePolRef(cos(phi)) * ray.stokes;
        #endif
        ray.direction = newDir;

        //trace ray
        ResultCode result = trace(ray, idx, dim + 2, false, ALLOW_RESPONSE);
        onEvent(ray, result, idx, i + 2);
        
        //stop codes are negative
        if (result < 0)
            return;
    }
}
