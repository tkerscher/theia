#ifndef _INCLUDE_RAY_MODEL_UNPOLARIZED
#define _INCLUDE_RAY_MODEL_UNPOLARIZED

#include "util/buffers.glsl"

/*********************************** STRUCTs **********************************/

struct HitItem {
    vec3 position;
    vec3 direction;
    vec3 normal;

    int objectId;

    float wavelength;

    #ifdef RAY_TRANSIENT
    float time;
    #endif

    #ifndef RAY_PARTICLE
    float contrib;
    #endif
};

struct ForwardRay {
    vec3 position;
    vec3 direction;

    float wavelength;
    uint mediumIdx;

    #ifdef RAY_TRANSIENT
    float time;
    #endif

    #ifndef RAY_PARTICLE
    float lin_contrib;
    float log_contrib;
    #endif
};
//tell other code this ray exist
#define ForwardRay ForwardRay

//Backward rays require contrib and thus cannot be particles
#ifndef RAY_PARTICLE

struct BackwardRay {
    vec3 position;
    vec3 direction;

    float wavelength;
    uint mediumIdx;

    #ifdef RAY_TRANSIENT
    float time;
    #endif

    float lin_contrib;
    float log_contrib;
};
//tell other code this ray exists
#define BackwardRay BackwardRay

struct CameraHit {
    //Note: The following are defined in object space
    vec3 position;
    vec3 direction;
    vec3 normal;

    int objectId;
};

#endif

#ifndef RAY_STATIC

/************************************ INIT ************************************/

#ifndef RAY_PARTICLE

ForwardRay createForwardRay(
    vec3 position,
    vec3 direction,
    float wavelength,
    uint mediumIdx,
    float time,
    float contrib
) {
    return ForwardRay(
        position,
        direction,
        wavelength,
        mediumIdx,
        #ifdef RAY_TRANSIENT
        time,
        #endif
        contrib,
        0.0
    );
}

#endif

ForwardRay createForwardRay(
    vec3 position,
    vec3 direction,
    float wavelength,
    uint mediumIdx,
    float time
) {
    return ForwardRay(
        position,
        direction,
        wavelength,
        mediumIdx
        #ifdef RAY_TRANSIENT
        ,time
        #endif
        #ifndef RAY_PARTICLE
        ,1.0
        ,0.0
        #endif
    );
}

#ifndef RAY_PARTICLE

BackwardRay createBackwardRay(
    vec3 position,
    vec3 direction,
    float wavelength,
    uint mediumIdx,
    float time,
    float contrib
) {
    return BackwardRay(
        position,
        direction,
        wavelength,
        mediumIdx,
        #ifdef RAY_TRANSIENT
        time,
        #endif
        contrib,
        0.0
    );
}

CameraHit createCameraHit(
    vec3 position,
    vec3 direction,
    vec3 normal,
    int objectId
) {
    return CameraHit(
        position,
        direction,
        normal,
        objectId
    );
}

#endif

/********************************* PROPAGATION ********************************/

ResultCode propagateRay(
    inout ForwardRay ray,
    float dist
) {
    ray.position += dist * ray.direction;

    #ifdef RAY_TRANSIENT
    float lam = normalize_lambda(ray.mediumIdx, ray.wavelength);
    float vg = lookUpMediaTable1D(GROUP_VELOCITY, ray.mediumIdx, lam, SPEED_OF_LIGHT);
    ray.time += dist / vg;
    #endif

    return RESULT_CODE_SUCCESS;
}

void alignRayToHit(
    inout ForwardRay ray,
    vec3 normal
) {
    //nothing to do
}

void scatterRay(
    inout ForwardRay ray,
    vec3 newDir
) {
    ray.direction = newDir;
}

#ifndef RAY_PARTICLE

ResultCode propagateRay(
    inout BackwardRay ray,
    float dist
) {
    ray.position += dist * ray.direction;

    #ifdef RAY_TRANSIENT
    float lam = normalize_lambda(ray.mediumIdx, ray.wavelength);
    float vg = lookUpMediaTable1D(GROUP_VELOCITY, ray.mediumIdx, lam, SPEED_OF_LIGHT);
    ray.time += dist / vg;
    #endif

    return RESULT_CODE_SUCCESS;
}

void alignRayToHit(
    inout BackwardRay ray,
    vec3 normal
) {
    //nothing to do
}

void scatterRay(
    inout BackwardRay ray,
    vec3 newDir
) {
    ray.direction = newDir;
}

#endif

/************************************* HIT ************************************/

HitItem createHit(
    const ForwardRay ray,
    vec3 objHitPosition,
    vec3 objHitNormal,
    int objectId,
    mat3 worldToObj
) {
    //transform ray direction to object space
    vec3 objHitDir = normalize(worldToObj * ray.direction);
    return HitItem(
        objHitPosition,
        objHitDir,
        objHitNormal,
        objectId,
        ray.wavelength
        #ifdef RAY_TRANSIENT
        ,ray.time
        #endif
        #ifndef RAY_PARTICLE
        ,ray.lin_contrib * exp(ray.log_contrib)
        #endif
    );
}

#ifndef RAY_PARTICLE

HitItem combineRaysAligned(
    ForwardRay forward,
    BackwardRay backward,
    const CameraHit hit
) {
    float contrib = forward.lin_contrib * exp(forward.log_contrib) *
                    backward.lin_contrib * exp(backward.log_contrib);
    #ifdef RAY_TRANSIENT
    float time = forward.time + backward.time;
    #endif

    return HitItem(
        hit.position,
        hit.direction,
        hit.normal,
        hit.objectId,
        forward.wavelength,
        #ifdef RAY_TRANSIENT
        time,
        #endif
        contrib
    );
}

#endif

#endif //#ifdef RAY_STATIC

/***************************** QUEUE SERIALIZATION ****************************/

//helper macros for better readability
#define _loadInt ints.values[idx]; idx += queueSize
#define _loadUInt uints.values[idx]; idx += queueSize
#define _loadFloat floats.values[idx]; idx += queueSize
#define _loadVec3 vec3(floats.values[idx], floats.values[idx + queueSize], floats.values[idx + 2 * queueSize]); idx += 3 * queueSize
#define _saveInt(v) ints.values[idx] = (v); idx += queueSize
#define _saveUInt(v) uints.values[idx] = (v); idx += queueSize
#define _saveFloat(v) floats.values[idx] = (v); idx += queueSize
#define _saveVec3(v) _saveFloat(v.x); _saveFloat(v.y); _saveFloat(v.z)

void saveForwardRay(uvec2 queueAdr, uint queueSize, uint idx, const ForwardRay ray) {
    FloatBuffer floats = FloatBuffer(queueAdr);
    UIntBuffer uints = UIntBuffer(queueAdr);

    _saveVec3(ray.position);
    _saveVec3(ray.direction);
    _saveFloat(ray.wavelength);
    _saveUInt(ray.mediumIdx);
    
    #ifdef RAY_TRANSIENT
    _saveFloat(ray.time);
    #endif

    #ifndef RAY_PARTICLE
    float contrib = ray.lin_contrib * exp(ray.log_contrib);
    _saveFloat(contrib);
    #endif
}

ForwardRay loadForwardRay(uvec2 queueAdr, uint queueSize, uint idx) {
    FloatBuffer floats = FloatBuffer(queueAdr);
    UIntBuffer uints = UIntBuffer(queueAdr);

    vec3 position = _loadVec3;
    vec3 direction = _loadVec3;
    float wavelength = _loadFloat;
    uint mediumIdx = _loadUInt;
    #ifdef RAY_TRANSIENT
    float time = _loadFloat;
    #endif
    #ifndef RAY_PARTICLE
    float contrib = _loadFloat;
    #endif

    return ForwardRay(
        position,
        direction,
        wavelength,
        mediumIdx
        #ifdef RAY_TRANSIENT
        ,time
        #endif
        #ifndef RAY_PARTICLE
        ,contrib
        ,0.0
        #endif
    );
}

void saveHitItem(uvec2 queueAdr, uint queueSize, uint idx, const HitItem hit) {
    FloatBuffer floats = FloatBuffer(queueAdr);
    IntBuffer ints = IntBuffer(queueAdr);

    _saveVec3(hit.position);
    _saveVec3(hit.direction);
    _saveVec3(hit.normal);
    _saveInt(hit.objectId);
    _saveFloat(hit.wavelength);

    #ifdef RAY_TRANSIENT
    _saveFloat(hit.time);
    #endif

    #ifndef RAY_PARTICLE
    _saveFloat(hit.contrib);
    #endif
}

HitItem loadHitItem(uvec2 queueAdr, uint queueSize, uint idx) {
    FloatBuffer floats = FloatBuffer(queueAdr);
    IntBuffer ints = IntBuffer(queueAdr);

    vec3 position = _loadVec3;
    vec3 direction = _loadVec3;
    vec3 normal = _loadVec3;
    int objectId = _loadInt;
    float wavelength = _loadFloat;
    #ifdef RAY_TRANSIENT
    float time = _loadFloat;
    #endif
    #ifndef RAY_PARTICLE
    float contrib = _loadFloat;
    #endif

    return HitItem(
        position,
        direction,
        normal,
        objectId,
        wavelength
        #ifdef RAY_TRANSIENT
        ,time
        #endif
        #ifndef RAY_PARTICLE
        ,contrib
        #endif
    );
}

#ifndef RAY_PARTICLE

void saveBackwardRay(uvec2 queueAdr, uint queueSize, uint idx, const BackwardRay ray) {
    FloatBuffer floats = FloatBuffer(queueAdr);
    UIntBuffer uints = UIntBuffer(queueAdr);

    _saveVec3(ray.position);
    _saveVec3(ray.direction);
    _saveFloat(ray.wavelength);
    _saveUInt(ray.mediumIdx);
    #ifdef RAY_TRANSIENT
    _saveFloat(ray.time);
    #endif
    float contrib = ray.lin_contrib * exp(ray.log_contrib);
    _saveFloat(contrib);
}

BackwardRay loadBackwardRay(uvec2 queueAdr, uint queueSize, uint idx) {
    FloatBuffer floats = FloatBuffer(queueAdr);
    UIntBuffer uints = UIntBuffer(queueAdr);

    vec3 position = _loadVec3;
    vec3 direction = _loadVec3;
    float wavelength = _loadFloat;
    uint mediumIdx = _loadUInt;
    #ifdef RAY_TRANSIENT
    float time = _loadFloat;
    #endif
    float contrib = _loadFloat;

    return BackwardRay(
        position,
        direction,
        wavelength,
        mediumIdx,
        #ifdef RAY_TRANSIENT
        time,
        #endif
        contrib,
        0.0
    );
}

#ifdef RAY_TRANSIENT
#define BACKWARD_RAY_QUEUE_SLOTS 10
#else
#define BACKWARD_RAY_QUEUE_SLOTS 9
#endif

void saveBackwardRay(
    uvec2 queueAdr, uint queueSize, uint idx,
    const BackwardRay ray, const CameraHit hit
) {
    FloatBuffer floats = FloatBuffer(queueAdr);
    IntBuffer ints = IntBuffer(queueAdr);

    saveBackwardRay(queueAdr, queueSize, idx, ray);
    idx += BACKWARD_RAY_QUEUE_SLOTS * queueSize;

    _saveVec3(hit.position);
    _saveVec3(hit.direction);
    _saveVec3(hit.normal);
    _saveInt(hit.objectId);
}

BackwardRay loadBackwardRay(
    uvec2 queueAdr, uint queueSize, uint idx,
    out CameraHit hit
) {
    FloatBuffer floats = FloatBuffer(queueAdr);
    IntBuffer ints = IntBuffer(queueAdr);

    BackwardRay ray = loadBackwardRay(queueAdr, queueSize, idx);
    idx += BACKWARD_RAY_QUEUE_SLOTS * queueSize;

    vec3 position = _loadVec3;
    vec3 direction = _loadVec3;
    vec3 normal = _loadVec3;
    int objectId = _loadInt;

    hit = CameraHit(
        position,
        direction,
        normal,
        objectId
    );
    return ray;
}

#undef BACKWARD_RAY_QUEUE_SLOTS

#endif

//clean up macros
#undef _loadInt
#undef _loadUInt
#undef _loadFloat
#undef _loadVec3
#undef _saveInt
#undef _saveUInt
#undef _saveFloat
#undef _saveVec3

#endif
