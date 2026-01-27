#ifndef _INCLUDE_CAMERA_FLAT
#define _INCLUDE_CAMERA_FLAT

#include "math.glsl"
#include "util/sample.glsl"

uniform CameraParams {
    float width;
    float height; //length

    int objectId;
    uint mediumIdx;

    vec3 offset;
    mat3 view;
} cameraParams;

BackwardRay sampleCameraRay(
    float wavelength,
    out CameraHit hit,
    uint idx, inout uint dim
) {
    mat3 objToWorld = transpose(cameraParams.view); // inverse, since it's orthogonal
    //sample position on detector
    vec2 u = random2D(idx, dim);
    float localX = cameraParams.width * (u.x - 0.5);
    float localY = cameraParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);
    //transform to screen coord space
    vec3 rayPos = objToWorld * localPos + cameraParams.offset;

    //sample direction
    vec3 localDir = sampleHemisphere(random2D(idx, dim));
    float cos_theta = localDir.z;
    vec3 rayDir = objToWorld * localDir;
    //flip local dir as it should hit the detector
    localDir *= -1.0;
    //calculate contribution
    float contrib = TWO_PI * cameraParams.width * cameraParams.height * cos_theta;

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

struct CameraSample {
    vec3 position;
    vec3 normal;
    float contrib;
};

CameraSample sampleCamera(float wavelength, uint idx, inout uint dim) {
    mat3 objToWorld = transpose(cameraParams.view); // inverse, since it's orthogonal
    //sample position on detector
    vec2 u = random2D(idx, dim);
    float localX = cameraParams.width * (u.x - 0.5);
    float localY = cameraParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);
    //transform to screen coord space
    vec3 rayPos = objToWorld * localPos + cameraParams.offset;
    vec3 rayNrm = transpose(cameraParams.view) * vec3(0.0, 0.0, 1.0);
    //calculate contribution
    float contrib = cameraParams.width * cameraParams.height;
    //return sample
    return CameraSample(rayPos, rayNrm, contrib);
}

BackwardRay createCameraRay(
    const CameraSample cam,
    vec3 lightDir,
    float wavelength,
    out CameraHit hit
) {
    //get local coordinates
    vec3 localPos = cameraParams.view * (cam.position - cameraParams.offset);
    vec3 localDir = cameraParams.view * lightDir;
    //calculate contribution
    float cos_theta = -localDir.z; //dot(-localDir, vec3(0.0, 0.0, 1.0));
    float contrib = cam.contrib * cos_theta;
    //check light comes from the right side
    contrib *= float(dot(cam.normal, lightDir) < 0.0);

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
