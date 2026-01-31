#ifndef _INCLUDE_TRACER_PROPAGATE_FORWARD
#define _INCLUDE_TRACER_PROPAGATE_FORWARD

#include "tracer/propagate/common.glsl"
#define RAY ForwardRay
#include "tracer/propagate/template.glsl"
#undef RAY

#endif
