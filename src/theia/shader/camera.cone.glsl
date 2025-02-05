#ifndef _INCLUDE_CAMERARAYSOURCE_CONE
#define _INCLUDE_CAMERARAYSOURCE_CONE

#include "math.glsl"
#include "util.sample.glsl"

layout(scalar) uniform CameraParams {
    vec3 conePos;
    vec3 coneDir;
    float cosOpeningAngle;
} cameraParams;

CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim) {
    //sample cone
    vec3 localDir = sampleDirectionCone(cameraParams.cosOpeningAngle, random2D(idx, dim));
    //convert to global space
    mat3 trafo = createLocalCOSY(cameraParams.coneDir);
    vec3 rayDir = trafo * localDir;
    //flip local dir as it should point towards the detector
    localDir *= -1.0;

    //polarization
    vec3 hitPolRef, polRef;
    #ifdef POLARIZATION
    hitPolRef = perpendicularToZand(localDir);
    polRef = trafo * hitPolRef;
    #endif

    //assemble camera ray
    return createCameraRay(
        cameraParams.conePos,                    //ray position
        rayDir,                                     //ray direction
        polRef,                                     //ray polRef
        mat4(1.0),                                  //ray mueller matrix
        TWO_PI * (1.0 - cameraParams.cosOpeningAngle), //contrib
        0.0,                                        //time delta
        vec3(0.0, 0.0, 0.0),                        //hit position
        localDir,                                   //hit direction
        vec3(0.0, 0.0, 1.0),                        //hit normal
        hitPolRef                                   //hit polRef
    );
}

CameraSample sampleCamera(float wavelength, uint idx, uint dim) {
    return createCameraSample(
        cameraParams.conePos,
        cameraParams.coneDir,
        1.0
    );
}

CameraRay createCameraRay(CameraSample cam, vec3 lightDir, float wavelength) {
    //check if ray is within opening cone
    float cosAngle = dot(cameraParams.coneDir, -lightDir);
    float contrib = float(cosAngle >= 1.0 - cameraParams.cosOpeningAngle);

    //convert lightDir to local space
    mat3 trafo = createLocalCOSY(cameraParams.coneDir);
    vec3 hitDir = transpose(trafo) * lightDir;

    //create reference frame
    vec3 hitPolRef, polRef;
    #ifdef POLARIZATION
    hitPolRef = perpendicularToZand(hitDir);
    polRef = trafo * hitPolRef;
    #endif

    //assemble camera ray
    return createCameraRay(
        cameraParams.conePos,    //ray position
        -lightDir,                  //ray direction
        polRef,                     //polarization reference frame
        mat4(1.0),                  //mueller matrix
        contrib,                    //ray contribution
        0.0,                        //time delta
        vec3(0.0, 0.0, 0.0),        //hit position in object space
        hitDir,                     //hit direction in object space
        vec3(0.0, 0.0, 1.0),        //hit normal in object space
        hitPolRef                   //referenc frame in object space
    );
}

#endif
