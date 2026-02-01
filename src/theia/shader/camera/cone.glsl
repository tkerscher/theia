#ifndef _INCLUDE_CAMERA_CONE
#define _INCLUDE_CAMERA_CONE

#include "math.glsl"
#include "util/sample.glsl"

uniform CameraParams {
    vec3 conePos;
    float cosOpeningAngle;
    vec3 coneDir;
    uint mediumIdx;
    int objectId;
} cameraParams;

BackwardRay sampleCameraRay(
    float wavelength,
    out CameraHit hit,
    uint idx, inout uint dim
) {
    //sample cone
    vec3 localDir = sampleDirectionCone(cameraParams.cosOpeningAngle, random2D(idx, dim));
    float contrib = TWO_PI * (1.0 - cameraParams.cosOpeningAngle);
    //convert to global space
    mat3 trafo = createLocalCOSY(cameraParams.coneDir);
    vec3 rayDir = trafo * localDir;
    //flip local dir as it should point towards the detector
    localDir *= -1.0;

    //assemble camera hit
    hit = createCameraHit(
        vec3(0.0),
        localDir,
        vec3(0.0, 0.0, 1.0),
        cameraParams.objectId
    );
    //assemble backward ray
    return createBackwardRay(
        cameraParams.conePos,
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
    uint mediumIdx;
};

CameraSample sampleCamera(float wavelength, uint idx, inout uint dim) {
    return CameraSample(
        cameraParams.conePos,
        cameraParams.coneDir,
        1.0,
        cameraParams.mediumIdx
    );
}

BackwardRay createCameraRay(
    const CameraSample cam,
    vec3 lightDir,
    float wavelength,
    out CameraHit hit    
) {
    //check if ray is within opening cone
    float cosAngle = dot(cameraParams.coneDir, -lightDir);
    float contrib = float(cosAngle >= 1.0 - cameraParams.cosOpeningAngle);

    //convert lightDir to local space
    mat3 trafo = createLocalCOSY(cameraParams.coneDir);
    vec3 hitDir = transpose(trafo) * lightDir;

    //assemble camera hit
    hit = createCameraHit(
        vec3(0.0),
        hitDir,
        vec3(0.0, 0.0, 1.0),
        cameraParams.objectId
    );
    //assemble backward ray
    return createBackwardRay(
        cameraParams.conePos,
        -lightDir,
        wavelength,
        cameraParams.mediumIdx,
        0.0,
        contrib
    );
}

#endif
