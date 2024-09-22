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

//Enable ray tracing
#ifdef USE_SCENE
#extension GL_EXT_ray_query : require
#endif

//Check expected macros
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "ray.glsl"
#include "ray.propagate.glsl"

layout(scalar) uniform TraceParams {
    uvec2 medium;

    PropagateParams propagation;
} params;

//define global medium
#define USE_GLOBAL_MEDIUM
Medium getMedium() {
    return Medium(params.medium);
}
#include "ray.medium.glsl"

#include "ray.combine.glsl"
#include "ray.response.glsl"
#include "result.glsl"

#include "lightsource.common.glsl"
#include "camera.common.glsl"
#include "response.common.glsl"
#include "wavelengthsource.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "camera.glsl"
#include "source.glsl"
#include "response.glsl"
#include "photon.glsl"

#include "callback.util.glsl"

#ifdef USE_SCENE

//Top level acceleration structure containing the scene
uniform accelerationStructureEXT tlas;

bool isVisible(vec3 observer, vec3 target) {
    //Direction and length of shadow ray
    vec3 dir = target - observer;
    float dist = length(dir);
    dir /= dist;

    //create and trace ray query
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        observer,
        0.0, dir, dist
    );
    rayQueryProceedEXT(rayQuery);

    //points are mutable visible if no hit
    return rayQueryGetIntersectionTypeEXT(rayQuery, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT;
}

#else

//no scene -> all visibility tests are trivially true
bool isVisible(vec3 observer, vec3 target) {
    return true;
}

//One might expect here a test against a detector sphere.
//However, self shadowing is tested by comparing ray direction and detector
//surface normal.

#endif

ResultCode checkSamples(const CameraSample camSample, const SourceRay light) {
    //check light shines from the right side...
    if (dot(camSample.normal, light.direction) >= 0.0)
        return RESULT_CODE_RAY_MISSED;
    //...and is visible
    if (!isVisible(camSample.position, light.position))
        return RESULT_CODE_RAY_ABSORBED;
    
    //all good
    return RESULT_CODE_SUCCESS;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;
    
    //sample light path from wavelength, camera and lightsource
    WavelengthSample photon = sampleWavelength(idx, 0);
    CameraSample camSample = sampleCamera(
        photon.wavelength, idx, DIM_PHOTON_OFFSET);
    SourceRay light = sampleLight(
        camSample.position, camSample.normal,
        photon.wavelength, idx, DIM_CAM_OFFSET);
    ForwardRay ray = createRay(light, Medium(params.medium), photon);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
    
    //check if we can combine both rays
    ResultCode result = checkSamples(camSample, light);
    if (result >= 0) {
        //now that we know the direction the light hits the detector,
        //ask it to create a camera ray
        CameraRay camera = createCameraRay(camSample, light.direction, photon.wavelength);
        //Finally, combine source and camera ray to create the hit
        HitItem hit;
        result = combineRaysAligned(ray, camera, params.propagation, hit);
        if (result >= 0) {
            result = RESULT_CODE_RAY_DETECTED;
            response(hit);
        }
    }

    //tell callback
    onEvent(ray, result, idx, 1);
}
