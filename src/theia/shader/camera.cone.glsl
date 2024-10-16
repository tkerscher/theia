#ifndef _INCLUDE_CAMERARAYSOURCE_CONE
#define _INCLUDE_CAMERARAYSOURCE_CONE

#include "math.glsl"

layout(scalar) uniform CameraRayParams {
    vec3 conePos;
    vec3 coneDir;
    float cosOpeningAngle;
} cameraRayParams;

CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim) {
    //sample cone
    vec2 u = random2D(idx, dim);
    float phi = TWO_PI * u.x;
    float cos_theta = 1.0 - cameraRayParams.cosOpeningAngle * u.y;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    //construct local ray dir
    vec3 localDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    //convert to global space
    mat3 trafo = createLocalCOSY(cameraRayParams.coneDir);
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
        cameraRayParams.conePos,                    //ray position
        rayDir,                                     //ray direction
        polRef,                                     //ray polRef
        mat4(1.0),                                  //ray mueller matrix
        TWO_PI * cameraRayParams.cosOpeningAngle,   //contrib
        0.0,                                        //time delta
        vec3(0.0, 0.0, 0.0),                        //hit position
        localDir,                                   //hit direction
        vec3(0.0, 0.0, 1.0),                        //hit normal
        hitPolRef                                   //hit polRef
    );
}

CameraSample sampleCamera(float wavelength, uint idx, uint dim) {
    return createCameraSample(
        cameraRayParams.conePos,
        cameraRayParams.coneDir,
        1.0
    );
}

CameraRay createCameraRay(CameraSample cam, vec3 lightDir, float wavelength) {
    //check if ray is within opening cone
    float cosAngle = dot(cameraRayParams.coneDir, -lightDir);
    float contrib = float(cosAngle >= 1.0 - cameraRayParams.cosOpeningAngle);

    //convert lightDir to local space
    mat3 trafo = createLocalCOSY(cameraRayParams.coneDir);
    vec3 hitDir = transpose(trafo) * lightDir;

    //create reference frame
    vec3 hitPolRef, polRef;
    #ifdef POLARIZATION
    hitPolRef = perpendicularToZand(hitDir);
    polRef = trafo * hitPolRef;
    #endif

    //assemble camera ray
    return createCameraRay(
        cameraRayParams.conePos,    //ray position
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
