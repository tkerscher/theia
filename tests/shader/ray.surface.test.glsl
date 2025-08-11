layout(local_size_x = 32) in;

#include "ray.glsl"
#include "ray.propagate.glsl"
#include "ray.surface.glsl"
#include "scatter.surface.glsl"

#include "rng.glsl"

struct Result {
    //orig
    float wavelength;
    vec3 orig_dir;
    float contrib;
    
    //reflected
    vec3 refl_pos;
    vec3 refl_dir;
    float refl_contrib;
    
    //transmitted
    vec3 trans_pos;
    vec3 trans_dir;
    float trans_contrib;
};
writeonly buffer ResultBuffer { Result r[]; };

#ifdef POLARIZATION
struct PolRefResult {
    vec3 refl;
    vec3 trans;
};
writeonly buffer PolRefBuffer { PolRefResult polRef[]; };

#ifdef FORWARD
struct StokesResult {
    vec4 refl;
    vec4 trans;
};
writeonly buffer StokesBuffer { StokesResult stokes[]; };
#else
struct MuellerResult {
    mat4 refl;
    mat4 trans;
};
writeonly buffer MuellerBuffer { MuellerResult mueller[]; };
#endif

#endif //#ifdef POLARIZATION

layout(scalar, push_constant) uniform Push {
    Material mat;
    vec3 normal;

    float lam_min;
    float lam_max;
} push;

//We want to resuse the same code for both forward and backward rays
#ifdef FORWARD

#include "source.glsl"

#define Ray ForwardRay

Ray sampleRay(uint idx) {
    uint dim = 0;
    //sample wavelength
    float u = random(idx, dim);
    float lam = push.lam_min + (push.lam_max - push.lam_min) * u;

    //use wrong medium for light sampling as it does not depend on it
    MediumConstants consts = lookUpMedium(Medium(uvec2(0)), lam);
    SourceRay source = sampleLight(lam, consts, idx, dim);
    bool inward = dot(source.direction, push.normal) < 0.0;
    Medium med = inward ? push.mat.outside : push.mat.inside; //medium of ray
    consts = lookUpMedium(med, lam);
    
    return createRay(source, med, consts, lam);
}

#else //#ifdef FORWARD

#include "camera.glsl"

#define Ray BackwardRay

Ray sampleRay(uint idx) {
    uint dim = 0;
    //sample wavelength
    float u = random_s(idx, dim);
    float lam = push.lam_min + (push.lam_max - push.lam_min) * u;

    CameraRay cam = sampleCameraRay(lam, idx, dim);
    bool inward = dot(cam.direction, push.normal) < 0.0;
    Medium med = inward ? push.mat.outside : push.mat.inside; //medium of ray

    return createRay(cam, med, lam);
}

#endif

//use function for easy copy
Ray doReflect(Ray ray, const SurfaceHit hit, const Reflectance refl) {
    reflectRay(ray, hit, refl);
    return ray;
}
Ray doTransmit(Ray ray, const SurfaceHit hit, const Reflectance refl) {
    transmitRay(ray, hit, refl);
    return ray;
}

Result createResult(Ray original, Ray reflected, Ray transmitted) {
    return Result(
        original.state.wavelength,
        original.state.direction,
        original.state.lin_contrib * exp(original.state.log_contrib),

        reflected.state.position,
        reflected.state.direction,
        reflected.state.lin_contrib * exp(reflected.state.log_contrib),

        transmitted.state.position,
        transmitted.state.direction,
        transmitted.state.lin_contrib * exp(reflected.state.log_contrib)
    );
}

void main() {
    //sample ray
    uint i = gl_GlobalInvocationID.x;
    Ray ray = sampleRay(i);
    //align polRef (prerequisite)
    alignRayToHit(ray, push.normal);
    //to keep tests manageable reset mueller matrix
    #if defined(POLARIZATION) && !defined(FORWARD)
    ray.mueller = mat4(1.0);
    #endif

    //assemble dummy hit
    bool inward = dot(ray.state.direction, push.normal) <= 0.0;
    vec3 normal = inward ? push.normal : -push.normal;
    SurfaceHit hit = SurfaceHit(
        true, push.mat, inward,
        0, 0,
        ray.state.position, normal,
        ray.state.position, push.normal, ray.state.direction,
        mat3(1.0)
    );

    //calculate surface reflectance
    Reflectance surface = fresnelReflect(ray.state, hit);
    //reflect and transmit
    Ray refl = doReflect(ray, hit, surface);
    Ray trans = doTransmit(ray, hit, surface);

    //create result and save it
    r[i] = createResult(ray, refl, trans);

    //store polarization state
    #ifdef POLARIZATION
    polRef[i] = PolRefResult(refl.polRef, trans.polRef);
    #ifdef FORWARD
    stokes[i] = StokesResult(refl.stokes, trans.stokes);
    #else
    mueller[i] = MuellerResult(refl.mueller, trans.mueller);
    #endif
    #endif
}
