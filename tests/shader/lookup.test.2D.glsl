#include "lookup.glsl"

layout(local_size_x = 4, local_size_y = 4) in;

// let's use a image since it has an easier api
layout(binding = 0, r32f) uniform writeonly image2D outputImage;

layout(push_constant) uniform PushConstant {
    Table2D table;
    float normalization; //i.e. total invocations per dim
} push;

void main() {
    float u = float(gl_GlobalInvocationID.x) / push.normalization;
    float v = float(gl_GlobalInvocationID.y) / push.normalization;

    float value = lookUp2D(push.table, u, v);
    imageStore(outputImage, ivec2(gl_GlobalInvocationID.xy), vec4(value));
}
