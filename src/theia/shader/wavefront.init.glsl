#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "wavefront.items.glsl"

layout(local_size_x = 32) in;

struct PhotonQuery{
    float wavelength;
    float log_radiance;
    float startTime;
    float probability;
};

struct RayQuery{
    vec3 position;
    vec3 direction;
    int targetIdx;
    PhotonQuery photons[N_PHOTONS];
};
layout(scalar) readonly buffer QueryInput{ RayQuery queries[]; };

layout(scalar) writeonly buffer RayQueue{
    uint rayCount;
    RayItem rayItems[];
};

layout(push_constant, scalar) uniform Push{
    uvec2 medium;
    uint count;
    uint rngStride;
} params;

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= params.count)
        return;
    RayQuery query = queries[idx];
    Medium medium = Medium(params.medium);

    //create photons
    Photon photons[N_PHOTONS];
    for (int i = 0; i < N_PHOTONS; ++i) {
        photons[i] = createPhoton(
            medium,
            query.photons[i].wavelength,
            query.photons[i].startTime,
            query.photons[i].log_radiance,
            query.photons[i].probability
        );
    }
    
    //create ray from query and place it on the queue
    // int n = atomicAdd(rayCount, 1);
    //we assume that we are filling an empty queue
    // -> no need for atomic counting
    Ray ray = Ray(
        query.position,
        query.direction,
        idx * params.rngStride, //rngIdx
        params.medium,
        photons
    );
    rayItems[idx] = RayItem(ray, query.targetIdx);
    //save the count only once (it would not matter though)
    if (idx == 0) {
        rayCount = params.count;
    }
}
