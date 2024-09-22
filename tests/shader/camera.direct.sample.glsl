#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 32) in;

#define LAM_MIN 400.0
#define LAM_MAX 800.0

#include "math.glsl"

#include "camera.common.glsl"

#include "rng.glsl"
#include "camera.glsl"

struct Result {
    float wavelength;
    vec3 lightDir;
    CameraSample cam;
    CameraRay ray;
};
layout(scalar) writeonly buffer ResultBuffer { Result r[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;

    //sample wavelength
    float lambda = mix(LAM_MIN, LAM_MAX, random(i, 0));
    //sample light direction
    float cos_theta = 2.0 * random(i, 1) - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * random(i, 2);
    vec3 lightDir = vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );

    //sample camera
    CameraSample cam = sampleCamera(lambda, i, 3);
    CameraRay ray = createCameraRay(cam, lightDir, lambda);

    //save result
    r[i] = Result(lambda, lightDir, cam, ray);
}
