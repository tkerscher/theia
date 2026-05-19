#ifndef _INCLUDE_CAMERA_DISK
#define _INCLUDE_CAMERA_DISK

#include "util/sample.glsl"

uniform CameraParams {
    vec3 position;
    float radius;

    int objectId;
    uint mediumIdx;

    mat3 view;
} cameraParams;

BackwardRay sampleCameraRay(
    float wavelength,
    out CameraHit hit,
    uint idx, inout uint dim
) {
    //sample position on disk
    vec3 localPos = cameraParams.radius * sampleUnitDisk(random2D(idx, dim));
    //transform to world space
    mat3 objToWorld = transpose(cameraParams.view); //transpose, since it's orthogonal
    vec3 rayPos = objToWorld * localPos + cameraParams.position;

    //sample direction
    vec3 localDir = sampleHemisphereCosine(random2D(idx, dim));
    vec3 rayDir = objToWorld * localDir;
    //flip local dir as it should hit the detector
    localDir *= -1.0;
    //calculate contribution
    float contrib = PI_SQUARE * cameraParams.radius * cameraParams.radius;

    //assemble camera hit
    hit = createCameraHit(
        localPos,
        localDir,
        vec3(0.0, 0.0, 1.0),
        cameraParams.objectId
    );
    //assemble backward ray
    return createBackwardRay(
        rayPos,
        rayDir,
        wavelength,
        cameraParams.mediumIdx,
        0.0,
        contrib
    );
}

struct CameraSample{
    vec3 position;
    vec3 normal;
    float contrib;
    uint mediumIdx;
};

CameraSample sampleCamera(
    float wavelength,
    uint idx, inout uint dim
) {
    //sample position on detector
    vec3 localPos = cameraParams.radius * sampleUnitDisk(random2D(idx, dim));
    //transform to world space
    mat3 objToWorld = transpose(cameraParams.view); //inverse, since it's orthogonal
    vec3 rayPos = objToWorld * localPos + cameraParams.position;
    vec3 rayNrm = objToWorld * vec3(0.0, 0.0, 1.0); //again, it's orthogonal
    //calculate contrib
    float contrib = PI * cameraParams.radius * cameraParams.radius;
    //return sample
    return CameraSample(rayPos, rayNrm, contrib, cameraParams.mediumIdx);
}

BackwardRay createCameraRay(
    const CameraSample cam,
    vec3 lightDir,
    float wavelength,
    out CameraHit hit
) {
    //get local coordinates
    vec3 localPos = cameraParams.view * (cam.position - cameraParams.position);
    vec3 localDir = cameraParams.view * lightDir;
    //calculate contributions
    float cos_theta = dot(cam.normal, lightDir);
    //set contrib to zero if light comes from the wrong side
    float contrib = cam.contrib * max(-cos_theta, 0.0);

    //assemble camera hit
    hit = createCameraHit(
        localPos,
        localDir,
        vec3(0.0, 0.0, 1.0),
        cameraParams.objectId
    );
    return createBackwardRay(
        cam.position,
        -lightDir,
        wavelength,
        cameraParams.mediumIdx,
        0.0,
        contrib
    );
}

#endif
