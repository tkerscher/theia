layout(local_size_x = 32) in;

#include "lightsource.particles.common.glsl"
//rng
#include "rng.glsl"

writeonly buffer Values { float x[]; };
writeonly buffer Contribs { float c[]; };

layout(scalar, push_constant) uniform PushConstant {
    float n;
    float a;
    float b;
    float cos_min;
    float cos_max;
} push;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint dim = 0;

    float contrib;
    float value = particle_sampleEmissionAngle(
        push.n,
        push.a, push.b,
        random(idx, dim),
        push.cos_min, push.cos_max,
        contrib
    );

    x[idx] = value;
    c[idx] = contrib;
}
