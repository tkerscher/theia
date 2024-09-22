#ifndef _INCLUDE_CAMERARAYSOURCE_SPHERE
#define _INCLUDE_CAMERARAYSOURCE_SPHERE

#include "math.glsl"

layout(scalar) uniform CameraRayParams {
    vec3 position;
    float radius;

    float timeDelta;

    //constant factor calculated on cpu
    float contribFwd;
    float contribBwd;
} cameraRayParams;

CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim) {
    //sample normal
    vec2 u = random2D(idx, dim);
    float phi = TWO_PI * u.x;
    float cos_theta = 2.0 * u.y - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    vec3 normal = vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );
    //derive ray pos from normal
    vec3 rayPos = cameraRayParams.radius * normal + cameraRayParams.position;

    //sample direction
    u = random2D(idx, dim);
    phi = TWO_PI * u.x;
    cos_theta = 1.0 - u.y; //upper hemisphere (exclude 0.0)
    sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    vec3 rayDir = vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );
    //rotate dir so that hemisphere coincides with normal
    mat3 cosy = createLocalCOSY(normal);
    rayDir = cosy * rayDir;
    //local dir is opposite (hits sphere)
    vec3 localDir = -rayDir;
    
    float contrib = cos_theta * cameraRayParams.contribFwd;

    //create polarization reference frame in plane of incidence
    vec3 polRef = perpendicularTo(localDir, normal);

    //assemble camera ray
    return createCameraRay(
        rayPos,                     //ray position
        rayDir,                     //ray direction
        polRef,                     //ray polRef
        mat4(1.0),                  //ray mueller matrix
        contrib,                    //contribution
        cameraRayParams.timeDelta,  //time delta
        normal,                     //hit pos on unit sphere
        localDir,                   //local dir (opposite dir than normal)
        normal,                     //normal on unit sphere
        polRef                      //hit polRef
    );
}

CameraSample sampleCamera(float wavelength, uint idx, inout uint dim) {
    //sample normal
    vec2 u = random2D(idx, dim);
    float phi = TWO_PI * u.x;
    float cos_theta = 2.0 * u.y - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    vec3 normal = vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );
    //derive ray pos from normal
    vec3 rayPos = cameraRayParams.radius * normal + cameraRayParams.position;

    return CameraSample(rayPos, normal, cameraRayParams.contribBwd);
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
        cameraRayParams.timeDelta,  //time delta
        cam.normal,                 //hit position
        lightDir,                   //hit direction
        cam.normal,                 //hit normal
        polRef                      //hit polRef
    );
}

#endif
