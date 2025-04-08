#ifndef _INCLUDE_DIRECT_TRACE_COMMON
#define _INCLUDE_DIRECT_TRACE_COMMON

#include "ray.combine.glsl"

#ifdef USE_SCENE

bool isDirectVisible(vec3 observer, vec3 target) {
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
bool isDirectVisible(vec3 observer, vec3 target) {
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
    if (!isDirectVisible(camSample.position, light.position))
        return RESULT_CODE_RAY_ABSORBED;
    
    //all good
    return RESULT_CODE_SUCCESS;
}

void sampleDirect(
    uint idx, inout uint dim,
    const Medium medium,
    const PropagateParams params    
) {
    //sample light path from wavelength, camera and lightsource
    WavelengthSample photon = sampleWavelength(idx, dim);
    CameraSample camSample = sampleCamera(
        photon.wavelength, idx, dim);
    MediumConstants consts = lookUpMedium(medium, photon.wavelength);
    SourceRay light = sampleLight(
        camSample.position, camSample.normal,
        photon.wavelength, consts, idx, dim);
    ForwardRay ray = createRay(light, medium, consts, photon);
    onEvent(ray, RESULT_CODE_RAY_CREATED, idx, 0);
    
    //check if we can combine both rays
    ResultCode result = checkSamples(camSample, light);
    if (result >= 0) {
        //now that we know the direction the light hits the detector,
        //ask it to create a camera ray
        CameraRay camera = createCameraRay(camSample, light.direction, photon.wavelength);
        //Finally, combine source and camera ray to create the hit
        HitItem hit;
        result = combineRaysAligned(ray, camera, params, hit);
        if (result >= 0 && hit.contrib > 0.0) {
            result = RESULT_CODE_RAY_DETECTED;
            response(hit);
        }
    }

    //tell callback
    onEvent(ray, result, idx, 1);
}

#endif
