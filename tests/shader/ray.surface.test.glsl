#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

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
layout(scalar) writeonly buffer ResultBuffer { Result r[]; };

#ifdef POLARIZATION
struct PolRefResult {
    vec3 refl;
    vec3 trans;
};
layout(scalar) writeonly buffer PolRefBuffer { PolRefResult polRef[]; };

#ifdef FORWARD
struct StokesResult {
    vec4 refl;
    vec4 trans;
};
layout(scalar) writeonly buffer StokesBuffer { StokesResult stokes[]; };
#else
struct MuellerResult {
    mat4 refl;
    mat4 trans;
};
layout(scalar) writeonly buffer MuellerBuffer { MuellerResult mueller[]; };
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
    SourceRay source = sampleLight(idx, 0);
    bool inward = dot(source.direction, push.normal) < 0.0;
    Medium med = inward ? push.mat.outside : push.mat.inside; //medium of ray
    return createRay(source, med);
}

#else //#ifdef FORWARD

#include "camera.glsl"

#define Ray BackwardRay

Ray sampleRay(uint idx) {
    //sample wavelength
    float u = random(idx, 0);
    float lam = push.lam_min + (push.lam_max - push.lam_min) * u;

    CameraRay cam = sampleCameraRay(idx, 1);
    bool inward = dot(cam.direction, push.normal) < 0.0;
    Medium med = inward ? push.mat.outside : push.mat.inside; //medium of ray

    return createRay(cam, med, lam);
}

#endif

//use function for easy copy
Ray doReflect(Ray ray, const SurfaceReflectance surface) {
    reflectRay(ray, surface);
    return ray;
}
Ray doTransmit(Ray ray, const SurfaceReflectance surface) {
    transmitRay(ray, surface);
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

    //calculate surface reflectance
    SurfaceReflectance surface = fresnelReflect(
        push.mat,
        ray.state.wavelength,
        ray.state.constants.n,
        ray.state.direction,
        push.normal
    );
    //reflect and transmit
    Ray refl = doReflect(ray, surface);
    Ray trans = doTransmit(ray, surface);

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
