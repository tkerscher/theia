layout(local_size_x = 32) in;

#include "lightsource.particles.common.glsl"
//rng
#include "rng.glsl"

writeonly buffer Samples { float x[]; };

layout(scalar, push_constant) uniform PushConstant {
    float n;
    float a;
    float b;
} push;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint dim = 0;

    x[idx] = particle_sampleEmissionAngle(
        push.n, push.a, push.b, random(idx, dim)
    );
}
