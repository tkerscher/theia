#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require

//check expected macros
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef N_LAMBDA
#error "N_LAMBDA not defined"
#endif
#ifndef LIGHT_PATH_LENGTH
#error "LIGHT_PATH_LENGTH not defined"
#endif
#ifndef CAMERA_PATH_LENGTH
#error "CAMERA_PATH_LENGTH not defined"
#endif
#ifndef DIM_OFFSET_LIGHT
#error "DIM_OFFSET_LIGHT not defined"
#endif
#ifndef DIM_OFFSET_CAMERA
#error "DIM_OFFSET_CAMERA not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

//disable MIS in tracer
#define SCENE_TRAVERSE_DISABLE_MIS 1

#include "material.glsl"
#include "ray.propagate.glsl"
#include "ray.sample.glsl"
#include "result.glsl"
#include "scene.traverse.glsl"
#include "util.branch.glsl"

#include "lightsource.common.glsl"
#include "response.common.glsl"
#include "camera.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "source.glsl"
#include "camera.glsl"

layout(scalar) uniform TraceParams {
    uint targetId;
    uvec2 sceneMedium;
    uvec2 cameraMedium;

    PropagateParams propagation;
} params;

struct PathVertexSample {
    float time;
    float contrib;
};
struct PathVertex {
    bool connectable; //i.e medium scatter

    vec3 position;
    vec3 direction;
    uvec2 medium;

    //TODO: MIS connection strategy

    PathVertexSample samples[N_LAMBDA];
};

PathVertex createVertex(Ray ray, vec3 dir, bool connectable) {
    PathVertexSample samples[N_LAMBDA];
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        samples[i] = PathVertexSample(
            ray.samples[i].time,
            ray.samples[i].lin_contrib * exp(ray.samples[i].log_contrib)
        );
    }
    return PathVertex(
        connectable,
        ray.position,
        dir,
        ray.medium,
        samples
    );
}

//toggle light path response
#ifndef DISABLE_LIGHT_PATH_RESPONSE
#define LIGHT_PATH_ALLOW_RESPONSE true
#else
#define LIGHT_PATH_ALLOW_RESPONSE false
#endif
#ifndef DISABLE_DIRECT_LIGHTING
#define LIGHT_PATH_ALLOW_DIRECT true
#else
#define LIGHT_PATH_ALLOW_DIRECT false
#endif

PathVertex lightPath[LIGHT_PATH_LENGTH];
uint nLight;
void createLightSubPath(Ray ray, uint idx, uint dim) {
    bool allow = LIGHT_PATH_ALLOW_DIRECT;
    nLight = 0;
    [[unroll]] for (
        uint i = 1;
        i <= LIGHT_PATH_LENGTH;
        ++i, dim += SCENE_TRAVERSE_TRACE_RNG_STRIDE
    ) {
        //remember ray's current direction; trace() will already sample a new one
        vec3 dir = ray.direction;
        //trace ray
        ResultCode result = trace(
            ray, params.targetId,
            idx, dim,
            params.propagation,
            false/*LIGHT_PATH_ALLOW_RESPONSE && allow*/, false);
        onEvent(ray, result, idx, i);
        //stop codes are negative
        if (result < 0)
            break;
        
        //store vertex
        bool connectable = (result == RESULT_CODE_RAY_SCATTERED);
        lightPath[nLight] = createVertex(ray, dir, connectable);
        nLight++;

        //only needed for first iteration
        allow = true;
    }
}

//updates given ray to be the new camera ray
//uses the already samples wavelengths
void createCameraRay(inout Ray ray, CameraRay camera, uint idx) {
    Medium medium = Medium(params.cameraMedium);
    //update ray
    ray.position = camera.position;
    ray.direction = camera.direction;
    ray.medium = params.cameraMedium;
    //update samples
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        //wavelength should still be valid
        ray.samples[i].time = camera.timeDelta;
        ray.samples[i].log_contrib = 1.0;
        ray.samples[i].lin_contrib = camera.contrib;
        ray.samples[i].constants = lookUpMedium(medium, ray.samples[i].wavelength);
    }
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, nLight + 1);
}

//checks whether two positions in a volume are visible frome each other
bool isVisible(vec3 obs, vec3 dir, float d) {
    //trace shadow ray
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsTerminateOnFirstHitEXT, //only interested if there's anything
        0xFF, //mask -> hit anything
        obs, 0.0, dir, d);
    rayQueryProceedEXT(rayQuery);
    return rayQueryGetIntersectionTypeEXT(rayQuery, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT;
}

//calculates the extra factor we have to apply to account for the different
//amount of paths we could sample for each path length
float normalizeConnection(uint pathLength) {
    //calculate number of connections for given path length
    uint nLight = min(LIGHT_PATH_LENGTH, pathLength);
    uint nCam = min(CAMERA_PATH_LENGTH, pathLength);
    uint nCon = nLight + nCam - pathLength + 1;
    //take contributions from light path into account
#ifndef DISABLE_LIGHT_PATH_RESPONSE
    nCon += 1;
#endif
    //norm factor is 1/nCon
    return 1.0 / float(nCon);
}

void connectVertex(Ray ray, CameraRay camera, PathVertex vertex, uint pathLength) {
    //create connection segment
    vec3 con = vertex.position - ray.position;
    float d = length(con);
    con = normalize(con);
    //visibility check
    if (!isVisible(ray.position, con, d))
        return;

    //evaluate scattering phase function
    Medium medium = Medium(ray.medium);
    float attenuation = scatterProb(medium, ray.direction, con);
    attenuation *= scatterProb(medium, vertex.direction, -con);
    //apply normalization factor
    attenuation *= normalizeConnection(pathLength);

    //propagate connection segment to update samples
    //(ray is a local copy so we're fine to do whatever)
    //note that the tracing algorithm of both sub paths already applied a factor
    //mu_s to prepare the next tracing step, so we must not do that again here
    ray.direction = con;
    propagateRay(ray, d, params.propagation, false);

    //create responses
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        //determine final time and contribution
        float time = ray.samples[i].time + vertex.samples[i].time;
        float contrib = ray.samples[i].lin_contrib * exp(ray.samples[i].log_contrib);
        contrib *= attenuation * vertex.samples[i].contrib;
        //again: the two factor mu_s were already applied by trace()
        
        //skip out of bound and zero contribution samples
        if (time > params.propagation.maxTime || contrib == 0.0)
            continue;
        
        //create response
        response(HitItem(
            camera.hitPosition,
            camera.hitDirection,
            camera.hitNormal,
            ray.samples[i].wavelength,
            time, contrib
        ));
    }
}

//tries all connections with the light sub path and create response if successfull
void connectCameraRay(const Ray ray, CameraRay camera, uint nCamera) {
    //iterate over all vertices
    for (uint i = 0; i < nLight; ++i) {
        if (lightPath[i].connectable && lightPath[i].medium == ray.medium)
           connectVertex(ray, camera, lightPath[i], nCamera + i + 2);
    }
}

//traces camera ray and creates responses from all possible connections
void simulateCamera(Ray ray, CameraRay camera, uint idx, uint dim) {
    //change light to camera, effectively passing the wavelengths
    createCameraRay(ray, camera, idx);

    [[unroll]] for (uint i = 0; i < CAMERA_PATH_LENGTH; ++i) {
        //remember ray's current direction; trace() will already sample a new one
        vec3 oldDir = ray.direction;
        //trace ray
        ResultCode result = trace(
            ray, params.targetId,
            idx, dim,
            params.propagation,
            false, false);
        onEvent(ray, result, idx, nLight + i + 1);
        //stop codes are negative
        if (result < 0)
            break;
        
        //temporarly restore old ray direction for connection algorithm
        vec3 tmp = ray.direction;
        ray.direction = oldDir;
        //create connections if possible
        if (result == RESULT_CODE_RAY_SCATTERED)
            connectCameraRay(ray, camera, i);
        //restore new ray direction for tracing
        ray.direction = tmp;
    }
}

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;

    //sample light ray
    Ray ray = createRay(sampleLight(idx), Medium(params.sceneMedium));
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);

    //trace light path
    createLightSubPath(ray, idx, DIM_OFFSET_LIGHT);

    //create camera ray
    uint dim = DIM_OFFSET_LIGHT + LIGHT_PATH_LENGTH * SCENE_TRAVERSE_TRACE_RNG_STRIDE;
    CameraRay camera = sampleCameraRay(idx, dim);
    dim += DIM_OFFSET_CAMERA;

    //trace camera ray and connect eagerly
    //(reuse light ray for easy wavelengt passing)
    simulateCamera(ray, camera, idx, dim);
}
