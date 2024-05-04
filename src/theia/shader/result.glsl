#ifndef _INCLUDE_RESULT
#define _INCLUDE_RESULT

//enumeration of result codes
//negative are codes/errors indicating tracing should abort

//GLSL has no enum so we go a bit more old fashioned
#define ResultCode int

const ResultCode RESULT_CODE_SUCCESS        = 0;

const ResultCode RESULT_CODE_RAY_CREATED    = 1;
const ResultCode RESULT_CODE_RAY_SCATTERED  = 2;
const ResultCode RESULT_CODE_RAY_HIT        = 3;
const ResultCode RESULT_CODE_RAY_DETECTED   = 4;
const ResultCode RESULT_CODE_VOLUME_HIT     = 5;

const ResultCode RESULT_CODE_RAY_LOST       = -1;
const ResultCode RESULT_CODE_RAY_DECAYED    = -2;
const ResultCode RESULT_CODE_RAY_ABSORBED   = -3;
const ResultCode RESULT_CODE_RAY_MISSED     = -4;

const ResultCode ERROR_CODE_MAX_VALUE       = -10;
const ResultCode ERROR_CODE_UNKNOWN         = -10;
const ResultCode ERROR_CODE_MEDIA_MISSMATCH = -11;
const ResultCode ERROR_CODE_TRACE_ABORT     = -12;
const ResultCode ERROR_CODE_RAY_BAD         = -13;

#endif
