#ifndef _INCLUDE_TRACER_SCENE_TARGET_CONFIG
#define _INCLUDE_TRACER_SCENE_TARGET_CONFIG

//Target tracer can go both forward (start at light source and hit detector meshes)
//and backwards (start at camera and hit "light source" meshes)
//This file sets the corresponding macros to configure the tracer

#define TRACE_DIRECTION_FORWARD 1
#define TRACE_DIRECTION_BACKWARD 2

#ifndef TRACE_DIRECTION
#error "No trace direction specified"
#endif

#if TRACE_DIRECTION == TRACE_DIRECTION_FORWARD

#define TRACE_RAY ForwardRay
#define MATERIAL_TARGET_BIT MATERIAL_DETECTOR_BIT

#elif TRACE_DIRECTION == TRACE_DIRECTION_BACKWARD

#define TRACE_RAY BackwardRay
#define MATERIAL_TARGET_BIT MATERIAL_LIGHT_SOURCE_BIT

#else

#error "Invalid trace direction specified!"

#endif

#endif
