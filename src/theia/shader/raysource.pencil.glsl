#ifndef _INCLUDE_RAYSAMPLER_PENCIL
#define _INCLUDE_RAYSAMPLER_PENCIL

layout(scalar) uniform RayParams {
    vec3 position;
    vec3 direction;
} rayParams;

RaySample sampleRay(uint idx, uint dim) {
    //assemble ray
    return RaySample(
        rayParams.position,
        rayParams.direction,
        1.0
    );
}

#endif
