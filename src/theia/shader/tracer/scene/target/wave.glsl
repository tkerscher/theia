#ifndef _INCLUDE_TRACER_SCENE_TARGET_WAVE
#define _INCLUDE_TRACER_SCENE_TARGET_WAVE

/**
 * Per-dispatch state of the scene target tracer with wavefront compaction.
 */

#include "util/buffers.glsl"

uniform WaveParams {
    //Queue this dispatch reads its rays from. Unused by the first wave, which
    //samples the light source instead.
    uvec2 srcQueue;
    //Queue the survivors of this dispatch are appended to.
    uvec2 dstQueue;
    //Address of the counter of dstQueue. The three uints it points at are a
    //VkTraceRaysIndirectCommandKHR, i.e. (count, 1, 1): this shader only ever
    //touches the first of them via Counter(), while the host hands the same
    //address to traceRaysIndirect, whose command processor reads all three as
    //the launch dimensions of the next wave.
    uvec2 dstCounter;
    //Number of trace iterations this dispatch performs.
    uint iterations;
    //Number of slots per queue. The ray states are stored as a structure of
    //arrays, so this is also the stride between consecutive fields.
    uint capacity;
    //Non-zero for the wave that samples the light source, zero for the ones
    //continuing the paths out of srcQueue.
    uint firstWave;
} waveParams;

#endif
