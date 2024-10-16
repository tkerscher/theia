#ifndef _INCLUDE_CAMERARAYSOURCE_FLAT
#define _INCLUDE_CAMERARAYSOURCE_FLAT

#include "math.glsl"

layout(scalar) uniform CameraRayParams {
    float width;
    float height; //length
    vec3 offset;
    mat3 view;
} cameraRayParams;

CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim) {
    mat3 objToWorld = transpose(cameraRayParams.view); // inverse, since it's orthogonal
    //sample position on detector
    vec2 u = random2D(idx, dim);
    float localX = cameraRayParams.width * (u.x - 0.5);
    float localY = cameraRayParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);
    //transform to sceen coord space
    vec3 rayPos = objToWorld * localPos + cameraRayParams.offset;

    //sample direction
    u = random2D(idx, dim);
    float cos_theta = 1.0 - u.x; //limit to upper hemisphere (exclude 0.0)
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * u.y;
    vec3 localDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
    vec3 rayDir = objToWorld * localDir;
    //flip local dir as it should hit the detector
    localDir *= -1.0;
    //calculate contribution
    float contrib = TWO_PI * cameraRayParams.width * cameraRayParams.height * cos_theta;

    //polarization
    vec3 hitPolRef, polRef;
    #ifdef POLARIZATION
    hitPolRef = perpendicularToZand(localDir);
    //NOTE: This only works because objToWorld is orthogonal
    polRef = objToWorld * hitPolRef;
    #endif

    //assemble ray
    return createCameraRay(
        rayPos,
        rayDir,
        polRef,
        mat4(1.0),
        contrib,
        0.0,
        localPos,
        localDir,
        vec3(0.0, 0.0, 1.0),
        hitPolRef
    );
}

CameraSample sampleCamera(float wavelength, uint idx, inout uint dim) {
    mat3 objToWorld = transpose(cameraRayParams.view); // inverse, since it's orthogonal
    //sample position on detector
    vec2 u = random2D(idx, dim);
    float localX = cameraRayParams.width * (u.x - 0.5);
    float localY = cameraRayParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);
    //transform to sceen coord space
    vec3 rayPos = objToWorld * localPos + cameraRayParams.offset;
    vec3 rayNrm = transpose(cameraRayParams.view) * vec3(0.0, 0.0, 1.0);
    //calculate contribution
    float contrib = cameraRayParams.width * cameraRayParams.height;
    //return sample
    return createCameraSample(rayPos, rayNrm, contrib);
}

CameraRay createCameraRay(CameraSample cam, vec3 lightDir, float wavelength) {
    //get local coordinates
    vec3 localPos = cameraRayParams.view * (cam.position - cameraRayParams.offset);
    vec3 localDir = cameraRayParams.view * lightDir;
    //calculate contribution
    float cos_theta = -localDir.z; //dot(-localDir, vec3(0.0, 0.0, 1.0));
    float contrib = cam.contrib * cos_theta;
    //check light comes from the right side
    contrib *= float(dot(cam.normal, lightDir) < 0.0);

    //polarization
    vec3 hitPolRef, polRef;
    #ifdef POLARIZATION
    hitPolRef = perpendicularToZand(localDir);
    polRef = transpose(cameraRayParams.view) * hitPolRef;
    #endif

    //assemble camera ray
    return createCameraRay(
        cam.position,
        -lightDir,
        polRef,
        mat4(1.0),
        contrib,
        0.0,
        localPos,
        localDir,
        vec3(0.0, 0.0, 1.0),
        hitPolRef
    );
}

#endif
