#ifndef _INCLUDE_CALLBACK_EMPTY
#define _INCLUDE_CALLBACK_EMPTY

#ifdef ForwardRay

void onEvent(const ForwardRay ray, ResultCode code, uint idx, uint i) {}

#endif

#ifdef BackwardRay

void onEvent(const BackwardRay ray, ResultCode code, uint idx, uint i) {}

#endif

#endif
