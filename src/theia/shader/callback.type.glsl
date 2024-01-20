#ifndef _INCLUDE_CALLBACK_TYPES
#define _INCLUDE_CALLBACK_TYPES

//enumeration of event types matching the values of ResultCode
//so we don't need to do arithmetic for conversion

#define EventType int

const EventType EVENT_TYPE_RAY_CREATED    = 1;
const EventType EVENT_TYPE_RAY_SCATTERED  = 2;
const EventType EVENT_TYPE_RAY_HIT        = 3;
const EventType EVENT_TYPE_RAY_DETECTED   = 4;
const EventType EVENT_TYPE_VOLUME_CHANGED = 5;
const EventType EVENT_TYPE_RAY_LOST       = -1;
const EventType EVENT_TYPE_RAY_DECAYED    = -2;
const EventType EVENT_TYPE_RAY_ABSORBED   = -3;

#endif
