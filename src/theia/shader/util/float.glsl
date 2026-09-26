#ifndef _INCLUDE_UTIL_FLOAT
#define _INCLUDE_UTIL_FLOAT

//machine epsilon constants
#define FLT_EPS 1.1920928955078125e-7
#define FLT_U   5.9604644775390625e-8

//Core Vulkan does not guarantee that fma only rounds one
//An optional extension allows us to use a dedicated SPIR-V instruction that
//guarantees it

#ifdef _FLT_FMA_32

spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
float fmaKHR(float a, float b, float c);
spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
vec2 fmaKHR(vec2 a, vec2 b, vec2 c);
spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
vec3 fmaKHR(vec3 a, vec3 b, vec3 c);
spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
vec4 fmaKHR(vec4 a, vec4 b, vec4 c);

#else
//not supported fallback to built-in without rounding guarantees
#define fmaKHR fma

#endif

#ifdef _FLT_FMA_64

spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
double fmaKHRd(double a, double b, double c);
spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
dvec2 fmaKHRd(dvec2 a, dvec2 b, dvec2 c);
spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
dvec3 fmaKHRd(dvec3 a, dvec3 b, dvec3 c);
spirv_instruction(extensions = ["SPV_KHR_fma"], capabilities = [6030], id = 4427)
dvec4 fmaKHRd(dvec4 a, dvec4 b, dvec4 c);

#else
//not supported fallback to built-in without rounding guarantees
#define fmaKHRd fma

#endif

#endif
