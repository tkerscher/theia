#ifndef _INCLUDE_TARGET_COMMON
#define _INCLUDE_TARGET_COMMON

struct TargetSample {
    vec3 position;      //Hit position in world space
    vec3 normal;        //Normal at position in world space
    float dist;         //Distance from observer to sample

    vec3 objPosition;   //Hit position in object space
    vec3 objNormal;     //Hit normal in object space
    
    float prob;         //Sample probability over Area (dA)
    bool valid;         //True for valid hit, false if missed

    vec3 offset;        //Translation from world to object space
    mat3 worldToObj;    //Orthogonal trafo from world to object space
};

TargetSample createTargetSample(
    vec3 position,
    vec3 normal,
    float dist,
    float prob,
    vec3 offset,
    mat3 worldToObj
) {
    return TargetSample(
        position,
        normal,
        dist,
        worldToObj * position + offset,
        worldToObj * normal, //works since orthogonal
        prob,
        true,
        offset,
        worldToObj
    );
}

TargetSample createTargetSample(
    vec3 position,
    vec3 normal,
    float dist,
    float prob,
    vec3 offset
) {
    return TargetSample(
        position,
        normal,
        dist,
        position + offset,
        normal,
        prob,
        true,
        offset,
        mat3(1.0)
    );
}

TargetSample createTargetSample(
    vec3 position,
    vec3 normal,
    float dist,
    float prob
) {
    return TargetSample(
        position,
        normal,
        dist,
        position,
        normal,
        prob,
        true,
        vec3(0.0),
        mat3(1.0)
    );
}

TargetSample createTargetSampleMiss() {
    return TargetSample(
        vec3(0.0),
        vec3(0.0),
        1.0 / 0.0, //inf
        vec3(0.0),
        vec3(0.0),
        0.0,
        false,
        vec3(0.0),
        mat3(1.0)
    );
}

#endif
