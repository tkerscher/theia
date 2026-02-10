#ifndef _INCLUDE_TRACER_PROPAGATE_BACKWARD
#define _INCLUDE_TRACER_PROPAGATE_BACKWARD

#include "tracer/propagate/common.glsl"
#define RAY BackwardRay
#include "tracer/propagate/template.glsl"
#undef RAY

#endif
