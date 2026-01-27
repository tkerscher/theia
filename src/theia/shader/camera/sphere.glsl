#ifndef _INCLUDE_CAMERA_SPHERE
#define _INCLUDE_CAMERA_SPHERE

#include "math.glsl"
#include "util/sample.glsl"

uniform CameraParams {
    vec3 position;
    float radius;

    float timeDelta;

    uint mediumIdx;
    int objectId;

    //constant factor calculated on cpu
    float contrib;
    float contribDirect;
} cameraParams;

BackwardRay sampleCameraRay(
    float wavelength,
    out CameraHit hit,
    uint idx, inout uint dim
) {
    //sample normal
    vec3 normal = sampleUnitSphere(random2D(idx, dim));
    //derive ray pos from normal
    vec3 rayPos = cameraParams.radius * normal + cameraParams.position;

    //sample direction
    vec3 rayDir = sampleHemisphere(random2D(idx, dim));
    float cos_theta = rayDir.z;
    //rotate dir so that hemisphere coincides with normal
    mat3 cosy = createLocalCOSY(normal);
    rayDir = cosy * rayDir;
    //local dir is opposite (hits sphere)
    vec3 localDir = -rayDir;
    
    float contrib = cos_theta * cameraParams.contrib;

    //assemble hit
    hit = createCameraHit(
        normal,                 //hit pos on unit sphere
        localDir,               //local dir
        normal,                 //normal on unit sphere
        cameraParams.objectId
    );
    //assemble ray
    return createBackwardRay(
        rayPos,
        rayDir,
        wavelength,
        cameraParams.mediumIdx,
        cameraParams.timeDelta,
        contrib
    );
}

struct CameraSample {
    vec3 position;
    vec3 normal;
    float contrib;
};

CameraSample sampleCamera(float wavelength, uint idx, inout uint dim) {
    //sample normal
    vec3 normal = sampleUnitSphere(random2D(idx, dim));
    //derive ray pos from normal
    vec3 rayPos = cameraParams.radius * normal + cameraParams.position;
    float contrib = FOUR_PI * cameraParams.radius * cameraParams.radius;

    return CameraSample(rayPos, normal, contrib);
}

BackwardRay createCameraRay(
    const CameraSample cam,
    vec3 lightDir,
    float wavelength,
    out CameraHit hit
) {
    //calculate contribution
    float cos_theta = dot(lightDir, -cam.normal);
    float contrib = cameraParams.contribDirect * cos_theta;
    //check light comes from the right side
    contrib *= float(dot(cam.normal, lightDir) < 0.0);

    //assemble hit
    hit = createCameraHit(
        cam.normal,
        lightDir,
        cam.normal,
        cameraParams.objectId
    );
    return createBackwardRay(
        cam.position,
        -lightDir,
        wavelength,
        cameraParams.mediumIdx,
        cameraParams.timeDelta,
        contrib
    );
}

#endif
