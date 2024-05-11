#ifndef _INCLUDE_CAMERARAYSOURCE_SPHERE
#define _INCLUDE_CAMERARAYSOURCE_SPHERE

#include "math.glsl"

layout(scalar) uniform CameraRayParams {
    vec3 position;
    float radius;

    float timeDelta;

    float contrib; //constant factor calculated on cpu
} cameraRayParams;

CameraRay sampleCameraRay(uint idx, uint dim) {
    //sample normal
    vec2 u = random2D(idx, dim); dim += 2;
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
    u = random2D(idx, dim); dim += 2;
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

    //calculate contribution (cos/prob)
    // - cosine is from lamberts law
    // - prob = p(point) * p(direction)
    float r = cameraRayParams.radius;
    float contrib = cos_theta * cameraRayParams.contrib;

    #ifdef POLARIZATION
    //create polarization reference frame in plane of incidence
    vec3 polRef = crosser(localDir, normal);
    //degenerate case: localDir || normal
    float len = length(polRef);
    if (len < 1e-5) {
        //both of the first two vectors in the normal cosy is perpendicular
        polRef = cosy[0];
    }
    else {
        polRef /= len;
    }
    #endif

    //assemble camera ray
    return CameraRay(
        rayPos,
        rayDir,
        contrib,
        cameraRayParams.timeDelta,
        #ifdef POLARIZATION
        polRef,
        #endif
        normal,     // hit pos on unit sphere
        localDir,   // local dir (opposite dir than normal)
        normal      // normal on unit sphere
    );
}

#endif
