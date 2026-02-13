#include "result.glsl"
#include "material.glsl"

#include "ray.glsl"
#include "rng.glsl"
#include "volume.glsl"

struct LengthData{
    PROXY_RAY ray;
    uint dim;
    float dist;
};
layout(location = 0) callableDataInEXT LengthData lengthData;

void main() {
    lengthData.dist = sampleInteractionLength(lengthData.ray, gl_LaunchIDEXT.x, lengthData.dim);
}
