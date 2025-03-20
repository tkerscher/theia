layout(local_size_x = 32) in;

#define LAM_MIN 400.0
#define LAM_MAX 800.0

#include "math.glsl"

#include "camera.common.glsl"

#include "rng.glsl"
#include "camera.glsl"
#include "util.sample.glsl"

struct Result {
    float wavelength;
    vec3 lightDir;
    CameraSample cam;
    CameraRay ray;
};
layout(scalar) writeonly buffer ResultBuffer { Result r[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint dim = 0;

    //sample wavelength
    float lambda = mix(LAM_MIN, LAM_MAX, random(i, dim));
    //sample light direction
    vec3 lightDir = sampleUnitSphere(random2D(i, dim));

    //sample camera
    CameraSample cam = sampleCamera(lambda, i, dim);
    CameraRay ray = createCameraRay(cam, lightDir, lambda);

    //save result
    r[i] = Result(lambda, lightDir, cam, ray);
}
