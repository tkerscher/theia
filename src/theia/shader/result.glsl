#ifndef _INCLUDE_RESULT
#define _INCLUDE_RESULT

//enumeration of result codes
//negative are codes/errors indicating tracing should abort

//GLSL has no enum so we go a bit more old fashioned
//TODO: we might want to switch to 16 or even 8 bit reduce ray tracing payload size
#define ResultCode int

const ResultCode RESULT_CODE_SUCCESS                    = 0;

const ResultCode RESULT_CODE_RAY_CREATED                = 1;
const ResultCode RESULT_CODE_RAY_SCATTERED              = 2;
const ResultCode RESULT_CODE_RAY_HIT                    = 3;
const ResultCode RESULT_CODE_RAY_DETECTED               = 4;
const ResultCode RESULT_CODE_VOLUME_HIT                 = 5;

const ResultCode RESULT_CODE_RAY_LOST                   = -1;
const ResultCode RESULT_CODE_RAY_DECAYED                = -2;
const ResultCode RESULT_CODE_RAY_ABSORBED               = -3;
const ResultCode RESULT_CODE_RAY_MISSED                 = -4;
const ResultCode RESULT_CODE_MAX_ITER                   = -5;

//abort codes are negative, i.e. that highest bit is set
//we choose error codes to also have the second highest bit set
#define ERR_MASK 0xC0000000

const ResultCode ERROR_CODE_MAX_VALUE                   = 0xC0000000;
const ResultCode ERROR_CODE_UNKNOWN                     = ERR_MASK | 1;
const ResultCode ERROR_CODE_MEDIA_MISMATCH              = ERR_MASK | 2;
const ResultCode ERROR_CODE_TRACE_ABORT                 = ERR_MASK | 3;
const ResultCode ERROR_CODE_RAY_BAD                     = ERR_MASK | 4;
const ResultCode ERROR_CODE_TOTAL_INTERNAL_REFLECTION   = ERR_MASK | 5;

#undef ERR_MASK

#endif
