layout(local_size_x = 32) in;

#include "math.glsl"

#include "camera.common.glsl"
#include "wavelengthsource.common.glsl"

#include "rng.glsl"
#include "camera.glsl"
#include "photon.glsl"

#include "util.sample.glsl"

struct Result{
    float wavelength;
    vec3 lightDir;
    CameraSample cam;
    CameraRay ray;
};
writeonly buffer ResultBuffer { Result r[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx > BATCH_SIZE) return;

    uint dim = 0;
    //sample wavelength
    WavelengthSample photon = sampleWavelength(idx, dim);
    float lambda = photon.wavelength;
    //sample light direction
    vec3 lightDir = sampleUnitSphere(random2D(idx, dim));

    //sample camera
    CameraSample cam = sampleCamera(lambda, idx, dim);
    CameraRay ray = createCameraRay(cam, lightDir, lambda);

    //save result
    r[idx] = Result(lambda, lightDir, cam, ray);
}
