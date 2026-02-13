#include "result.glsl"
#include "material.glsl"

#include "ray.glsl"
#include "rng.glsl"
#include "volume.glsl"

#include "math.glsl"

struct ApplyData{
    PROXY_RAY ray;
    float dist;
    uint dim;
    ResultCode result;
};
layout(location = 1) callableDataInEXT ApplyData applyData;

void main() {
    //fetch hit flag from sign of dist
    bool hit = signBit(applyData.dist) < 0.0;
    float dist = abs(applyData.dist);

    //call model function
    applyData.result = applyVolumeSampled(
        applyData.ray,
        applyData.dist,
        hit,
        gl_LaunchIDEXT.x,
        applyData.dim
    );
}
