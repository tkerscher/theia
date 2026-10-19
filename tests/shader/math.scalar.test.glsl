layout(local_size_x = 32) in;

#include "math.glsl"
#include "math/expf.glsl"
#include "math/expm1.glsl"
#include "math/hyperbolic.glsl"
#include "math/log1p.glsl"
#include "math/sincos.glsl"

struct Result{
    float expf;
    float expm1;
    float log1p;
    float sinhf;
    float coshf;
    vec2 sinhcoshf;
    vec2 sincos;
};

readonly buffer QueryBuffer{ float x[]; };
writeonly buffer ResultBuffer{ Result r[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    float a = x[i];
    r[i] = Result(
        expf(a),
        expm1(a),
        log1p(a),
        sinhf(a),
        coshf(a),
        sinhcoshf(a),
        sincos(a)
    );
}
