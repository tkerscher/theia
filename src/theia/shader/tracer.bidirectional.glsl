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
#ifndef DIM_OFFSET_LIGHT
#error "DIM_OFFSET_LIGHT not defined"
#endif
#ifndef DIM_OFFSET_CAMERA
#error "DIM_OFFSET_CAMERA not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

//disable MIS in tracer
#define SCENE_TRAVERSE_DISABLE_MIS 1
#ifdef POLARIZATION
#define EXPORT_MUELLER 1
#endif

#include "math.glsl"
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

struct PathVertex {
    bool connectable; //i.e medium scatter

    vec3 position;
    vec3 direction;
    uvec2 medium;

    #ifdef POLARIZATION
    vec4 stokes;
    vec3 polRef;
    #endif

    //TODO: MIS connection strategy

    float time;
    float contrib;
};

PathVertex createVertex(Ray ray, bool connectable) {
    return PathVertex(
        connectable,
        ray.position,
        ray.direction,
        ray.medium,
        #ifdef POLARIZATION
        ray.stokes, ray.polRef,
        #endif
        ray.time,
        ray.lin_contrib * exp(ray.log_contrib)
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
    #ifdef POLARIZATION
    mat4 mueller;
    #endif

    bool allow = LIGHT_PATH_ALLOW_DIRECT;
    nLight = 0;
    [[unroll]] for (
        uint i = 1;
        i <= LIGHT_PATH_LENGTH;
        ++i, dim += SCENE_TRAVERSE_TRACE_RNG_STRIDE
    ) {
        bool allowResponse = LIGHT_PATH_ALLOW_RESPONSE && allow;
        //trace ray
        SurfaceHit hit;
        ResultCode result = trace(
            ray, hit,
            params.targetId,
            idx, dim,
            params.propagation,
            allowResponse
        );
        //create vertex if successfull (connectable if nothing was hit)
        if (result < 0) break;
        lightPath[nLight] = createVertex(ray, !hit.valid);
        nLight++;
        //process interaction & prepare next trace
        if (result >= 0) {
            result = processInteraction(
                ray, hit, params.targetId,
                idx, dim + SCENE_TRAVERSE_TRACE_OFFSET,
                params.propagation,
                #ifdef POLARIZATION
                mueller,
                #endif
                allowResponse,
                true,  //forward, i.e. photons
                false  //last
            );
        }
        onEvent(ray, result, idx, i);
        //stop codes are negative
        if (result < 0)
            break;

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
    #ifdef POLARIZATION
    ray.polRef = camera.polRef;
    ray.stokes = vec4(1.0, 0.0, 0.0, 0.0); //not really needed
    #endif
    //wavelength should still be valid
    ray.time = camera.timeDelta;
    ray.log_contrib = 1.0;
    ray.lin_contrib = camera.contrib;
    ray.constants = lookUpMedium(medium, ray.wavelength);
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

//calculates the number of possible light connections for a given path length
//used for normalizing the specific path integral
uint nConnections(uint pathLength) {
    //calculate number of combinations between subpaths
    uint firstLight = max(pathLength - CAMERA_PATH_LENGTH - 1, 1);
    uint firstCam = max(min(CAMERA_PATH_LENGTH, pathLength - firstLight - 1), 0);
    uint n = min(nLight - firstLight + 1, firstCam);
    //take contributions from light path into account
    #ifndef DISABLE_DIRECT_LIGHTING
    //uint(bool) -> bool ? 1 : 0
    n += uint(pathLength > 0 && pathLength <= LIGHT_PATH_LENGTH + 1);
    #endif
    return n;
}

void connectVertex(
    Ray ray,
    CameraRay camera,
    PathVertex vertex,
    #ifdef POLARIZATION
    mat4 cameraMueller,
    vec3 hitPolRef,
    #endif
    uint pathLength
) {

    Medium medium = Medium(ray.medium);
    //create connection segment
    vec3 con = vertex.position - ray.position;
    float d = length(con);
    con = normalize(con);
    //visibility check
    if (!isVisible(ray.position, con, d))
        return;

    #ifdef POLARIZATION
    //connect chains of mueller matrices:
    vec3 polRef = vertex.polRef;
    vec4 stokes = vertex.stokes;
    //rotate light to first scatter plane
    stokes = rotatePolRef(vertex.direction, polRef, -con, polRef) * stokes;
    //apply phase matrix on first scatter
    stokes = lookUpPhaseMatrix(medium, dot(vertex.direction, -con)) * stokes;
    //rotate to second scatter plane
    stokes = rotatePolRef(-con, polRef, -ray.direction, polRef) * stokes;
    //apply phase matrix on second scatter
    stokes = lookUpPhaseMatrix(medium, dot(con, -ray.direction)) * stokes;
    //rotate to match reference plane on camera path
    stokes = matchPolRef(-ray.direction, polRef, ray.polRef) * stokes;
    //finally apply the accumulated mueller matrix from the camera path
    stokes = cameraMueller * stokes;
    #endif

    //evaluate scattering phase function
    float attenuation = scatterProb(medium, ray.direction, con) * ray.constants.mu_s;
    attenuation *= scatterProb(medium, vertex.direction, -con) * ray.constants.mu_s;
    //apply normalization factor
    attenuation /= float(nConnections(pathLength));

    //propagate connection segment to update samples
    //(ray is a local copy so we're fine to do whatever)
    ray.direction = con;
    propagateRay(ray, d, params.propagation, false);

    //create response
    //determine final time and contribution
    float time = ray.time + vertex.time;
    float contrib = ray.lin_contrib * exp(ray.log_contrib);
    contrib *= attenuation * vertex.contrib;

    #ifdef POLARIZATION
    //normalize stokes
    contrib *= stokes.x;
    stokes /= stokes.x;
    #endif
    
    //create response
    if (time <= params.propagation.maxTime && contrib != 0.0) {
        response(HitItem(
            camera.hitPosition,
            camera.hitDirection,
            camera.hitNormal,
            #ifdef POLARIZATION
            stokes, hitPolRef,
            #endif
            ray.wavelength,
            time, contrib
        ));
    }
}

//tries all connections with the light sub path and create response if successfull
/**
 * Connects the given camera ray to all recorded light vertices if possible
 *
 * ray: Ray containing propagation information
 * camera: Additional information about camera ray
 * camMueller: Mueller matrix describing the change in polarization along camera subpath
 * nCamera: Zero index of camera subpath vertex (camera is -1)
*/
void connectCameraRay(
    const Ray ray,
    CameraRay camera,
    #ifdef POLARIZATION
    mat4 camMueller,
    vec3 hitPolRef,
    #endif
    uint nCamera
) {
    //iterate over all vertices
    for (uint i = 0; i < nLight; ++i) {
        if (lightPath[i].connectable && lightPath[i].medium == ray.medium) {
            //i and nCamera are zero index
            //-> total path length is i + nCamera + 2(zero index) + 1(connection)
            #ifdef POLARIZATION
            connectVertex(ray, camera, lightPath[i], camMueller, hitPolRef, nCamera + i + 3);
            #else
            connectVertex(ray, camera, lightPath[i], nCamera + i + 3);
            #endif
        }
    }
}

//traces camera ray and creates responses from all possible connections
void simulateCamera(Ray ray, CameraRay camera, uint idx, uint dim) {
    //change light to camera, effectively passing the wavelengths
    createCameraRay(ray, camera, idx);

    #ifdef POLARIZATION
    mat4 mueller = mat4(1.0);       //last mueller matrix
    mat4 cumMueller = mat4(1.0);    //accumulated mueller matrix

    //create poleration reference in object space
    //we know it's in the plane of incidence on the normal
    //since stokes vector is the same after rotating 180deg, we can just use
    //the cross product and either is fine (+/-hitPolRef have the same stokes vector)
    vec3 hitPolRef = crosser(camera.hitDirection, camera.hitNormal);
    //degenerate case: dir || normal
    // -> just choose any perpendicular to objDir
    float len = length(hitPolRef);
    if (len > 1e-5) {
        hitPolRef /= len;
    }
    else {
        //first two columns of local cosy trafo are vectors perpendicular to hitDir
        hitPolRef = createLocalCOSY(camera.hitDirection)[0];
    }
    #endif

    [[unroll]] for (uint i = 0; i < CAMERA_PATH_LENGTH; ++i) {
        //trace ray
        SurfaceHit hit;
        ResultCode result = trace(
            ray, hit,
            params.targetId,
            idx, dim,
            params.propagation,
            false               //allowResponse
        );

        //create connections if possible
        if (result >= 0 && !hit.valid) {
            #ifdef POLARIZATION
            connectCameraRay(ray, camera, cumMueller, hitPolRef, i);
            #else
            connectCameraRay(ray, camera, i);
            #endif
        }

        //process interaction
        if (result >= 0) {
            result = processInteraction(
                ray, hit, params.targetId,
                idx, dim + SCENE_TRAVERSE_TRACE_OFFSET,
                params.propagation,
                #ifdef POLARIZATION
                mueller,
                #endif
                false,      //allowResponse
                false,      //backward, i.e. trace importance particles
                false       //not last
            );
        }
        onEvent(ray, result, idx, nLight + i + 1);
        //stop codes are negative
        if (result < 0)
            break;

        #ifdef POLARIZATION
        //cummulate mueller matrix
        cumMueller = cumMueller * mueller;
        #endif
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
