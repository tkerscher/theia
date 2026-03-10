#ifndef _INCLUDE_TRACER_SCENE_TARGET_PROPAGATE
#define _INCLUDE_TRACER_SCENE_TARGET_PROPAGATE

#ifndef TRACE_DIRECTION
#error "No trace direction specified"
#endif

#if TRACE_DIRECTION == TRACE_DIRECTION_FORWARD

#include "tracer/propagate/forward.glsl"

#elif TRACE_DIRECTION == TRACE_DIRECTION_BACKWARD

#include "tracer/propagate/backward.glsl"

#else

#error "Invalid trace direction specified!"

#endif

#endif
