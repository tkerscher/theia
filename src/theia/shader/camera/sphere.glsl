#ifndef _INCLUDE_CAMERARAYSOURCE_SPHERE
#define _INCLUDE_CAMERARAYSOURCE_SPHERE

#include "math.glsl"
#include "util.sample.glsl"

uniform CameraParams {
    vec3 position;
    float radius;

    float timeDelta;

    //constant factor calculated on cpu
    float contrib;
    float contribDirect;
} cameraParams;

CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim) {
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

    //create polarization reference frame in plane of incidence
    vec3 polRef = perpendicularTo(localDir, normal);

    //assemble camera ray
    return createCameraRay(
        rayPos,                     //ray position
        rayDir,                     //ray direction
        polRef,                     //ray polRef
        mat4(1.0),                  //ray mueller matrix
        contrib,                    //contribution
        cameraParams.timeDelta,  //time delta
        normal,                     //hit pos on unit sphere
        localDir,                   //local dir (opposite dir than normal)
        normal,                     //normal on unit sphere
        polRef                      //hit polRef
    );
}

CameraSample sampleCamera(float wavelength, uint idx, inout uint dim) {
    //sample normal
    vec3 normal = sampleUnitSphere(random2D(idx, dim));
    //derive ray pos from normal
    vec3 rayPos = cameraParams.radius * normal + cameraParams.position;

    return createCameraSample(rayPos, normal, cameraParams.contribDirect);
}

CameraRay createCameraRay(CameraSample cam, vec3 lightDir, float wavelength) {
    //calculate contribution
    float cos_theta = dot(lightDir, -cam.normal);
    float contrib = cam.contrib * cos_theta;
    //check light comes from the right side
    contrib *= float(dot(cam.normal, lightDir) < 0.0);
    //polarization
    vec3 polRef = perpendicularTo(lightDir, cam.normal);

    //assemble camera ray
    return createCameraRay(
        cam.position,               //ray position
        -lightDir,                  //ray direction
        polRef,                     //ray polRef
        mat4(1.0),                  //ray mueller matrix
        contrib,                    //contrib
        cameraParams.timeDelta,  //time delta
        cam.normal,                 //hit position
        lightDir,                   //hit direction
        cam.normal,                 //hit normal
        polRef                      //hit polRef
    );
}

#endif
