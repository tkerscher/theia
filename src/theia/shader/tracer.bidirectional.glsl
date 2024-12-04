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
#ifndef LIGHT_PATH_LENGTH
#error "LIGHT_PATH_LENGTH not defined"
#endif
#ifndef CAMERA_PATH_LENGTH
#error "CAMERA_PATH_LENGTH not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

//disable shadow rays in scene traverse
#define SCENE_TRAVERSE_FORWARD_DISABLE_MIS 1
#define SCENE_TRAVERSE_BACKWARD_DISABLE_SHADOW_RAYS 1

#include "scene.traverse.forward.glsl"
#include "scene.traverse.backward.glsl"

#include "lightsource.scatter.glsl"
#include "ray.combine.glsl"

#include "lightsource.common.glsl"
#include "response.common.glsl"
#include "camera.common.glsl"
//user provided code
#include "rng.glsl"
#include "callback.glsl"
#include "light.glsl"
#include "camera.glsl"
#include "photon.glsl"

#include "callback.util.glsl"

layout(scalar) uniform TraceParams {
    uvec2 sceneMedium;
    uvec2 cameraMedium;

    PropagateParams propagation;
} params;

struct PathVertex {
    uvec2 medium;   // zero if not connectable; i.e. not a volume scatter
    SourceRay ray;
};
PathVertex lightPath[LIGHT_PATH_LENGTH];
uint nLight;

//calculates the number of possible light connections for a given path length
//used for normalizing the specific path integral
float normalizePath(uint pathLength) {
    int l = int(pathLength);    //convert uint -> int
    int n = l - 2;              //ideal case for inf long sub paths
    n -= max(l - 2 - CAMERA_PATH_LENGTH, 0);   //limit from camera sub path
    n -= max(l - 2 - LIGHT_PATH_LENGTH, 0);    //limit from light sub path
    //just to be sure
    n = max(n, 0);

    return 1.0 / float(n);
}

void storeVertex(const ForwardRay ray, const SurfaceHit hit, uint i) {
    uvec2 medium = uvec2(ray.state.medium);
    //set medium to zero if surface hit to mark as not connectable
    medium *= uint(!hit.valid); //hit.valid ? 0 : 1
    
    //create and store vertex (zero based index)
    lightPath[i - 1] = PathVertex(
        medium,
        SourceRay(
            ray.state.position,
            ray.state.direction,
            #ifdef POLARIZATION
            ray.stokes,
            ray.polRef,
            #endif
            ray.state.time,
            ray.state.lin_contrib * exp(ray.state.log_contrib)
        )
    );
    nLight = i;
}

void createLightSubPath(
    WavelengthSample photon,
    inout uint pathIdx,
    uint idx, inout uint dim
) {
    //sample ray
    ForwardRay ray = createRay(
        sampleLight(photon.wavelength, idx, dim),
        Medium(params.sceneMedium), photon);
    #ifndef DISABLE_LIGHT_CALLBACK
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, pathIdx++);
    #endif

    //trace loop
    [[unroll]] for (uint i = 1; i <= LIGHT_PATH_LENGTH; ++i) {
        //trace ray
        bool last = i == LIGHT_PATH_LENGTH;
        SurfaceHit hit;
        ResultCode result = trace(
            ray, hit,
            0,                  //targetIdx; ignored
            idx, dim,
            params.propagation,
            false               //allowResponse
        );
        //store vertex
        storeVertex(ray, hit, i);
        //handle interaction
        if (result >= 0) {
            result = processInteraction(
                ray, hit,
                0,                  //targetIdx; ignored
                idx, dim,
                params.propagation,
                false,              //allowResponse
                last
            );
        }
        #ifndef DISABLE_LIGHT_CALLBACK
        onEvent(ray, result, idx, pathIdx++);
        #endif
        //stop codes are negative
        if (result < 0)
            return;
    }
}

void connectVertex(BackwardRay ray, CameraRay cam, SourceRay light, uint pathLength) {
    //check if visible
    if (!isVisible(ray.state.position, light.position))
        return;

    //we need the distance later for the G term
    float d = distance(ray.state.position, light.position);

    //normalize path length
    light.contrib *= normalizePath(pathLength);
    //scatter source ray in right direction    
    Medium medium = Medium(ray.state.medium);    
    scatterSourceRay(
        light, medium,
        ray.state.constants.mu_s,
        normalize(ray.state.position - light.position)
    );

    //combine rays
    HitItem hit;
    ResultCode result = combineRays(ray, light, cam.hit, params.propagation, hit);
    //add missing G term from the path integral that would have cancelled with
    //sampling the next direction.
    //This is equivalent as treating the light vertex as new point source with
    //the phase function as emission profile.
    hit.contrib /= d*d;

    if (result >= 0 && hit.contrib > 0.0) {
        response(hit);
    }
}

void completePath(BackwardRay ray, CameraRay cam, uint camLength) {
    //fail safe: do not even try to connect in vacuum (no scatter)
    if (ray.state.medium == uvec2(0))
        return;

    //iterate over all light path vertices
    [[unroll]] for (uint i = 0; i < LIGHT_PATH_LENGTH; ++i) {       
        if (i >= nLight) return;

        if (ray.state.medium == lightPath[i].medium) {
            //i and nCamera are zero index
            //-> total path length is i + nCamera + 2(zero index) + 1(connection)
            uint pathLength = camLength + i + 3;
            //complete path
            connectVertex(ray, cam, lightPath[i].ray, pathLength);
        }
    }
}

void simulateCamera(
    float wavelength,
    inout uint pathIdx,
    uint idx, inout uint dim
) {
    //sample camera ray
    CameraRay cam = sampleCameraRay(wavelength, idx, dim);
    BackwardRay ray = createRay(cam, Medium(params.cameraMedium), wavelength);
    #ifndef DISABLE_CAMERA_CALLBACK
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, pathIdx++);
    #endif

    //trace loop
    [[unroll]] for (uint i = 0; i < CAMERA_PATH_LENGTH; ++i) {
        //trace ray
        SurfaceHit hit;
        ResultCode result = trace(
            ray, hit,
            idx, dim,
            params.propagation
        );
        //on sucess...
        if (result >= 0) {
            //connect camera ray to light vertices
            completePath(ray, cam, i);
            //handle interaction
            processInteraction(
                ray, hit,
                cam.hit,
                idx, dim,
                params.propagation
            );
        }
        #ifndef DISABLE_CAMERA_CALLBACK
        onEvent(ray, result, idx, pathIdx++);
        #endif
        //stop codes are negative
        if (result < 0)
            return;
    }

}

void traceMain() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    uint pathIdx = 0;
    if (idx >= BATCH_SIZE)
        return;
    
    //sample wavelength
    WavelengthSample photon = sampleWavelength(idx, dim);
    //create light path
    createLightSubPath(photon, pathIdx, idx, dim);
    //trace camera path
    simulateCamera(photon.wavelength, pathIdx, idx, dim);    
}

void main() {
    initResponse();
    traceMain();
    finalizeResponse();
}
