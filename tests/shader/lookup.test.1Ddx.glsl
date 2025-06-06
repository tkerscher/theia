#include "lookup.glsl"

layout(local_size_x = 32) in;

layout(scalar) buffer writeonly ValueOut{ float value[]; };
layout(scalar) buffer writeonly DerivOut{ float deriv[]; };

layout(push_constant) uniform Push{
    Table1D table;
    float normalization; //i.e. total invocations
} push;

void main() {
    uint i = gl_GlobalInvocationID.x;
    float u = float(i) / push.normalization;

    vec2 result = lookUpDx(push.table, u);

    value[i] = result.x;
    deriv[i] = result.y;
}
