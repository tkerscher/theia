#ifndef _INCLUDE_TARGET_DISK
#define _INCLUDE_TARGET_DISK

#include "util.sample.glsl"

uniform TargetParams {
    vec3 position;
    float radius;
    vec3 normal;

    float prob;
    mat3 objToWorld;
} targetParams;

TargetSample sampleTarget(vec3 observer, uint idx, inout uint dim) {
    //sample point on disk
    vec3 localPos = targetParams.radius * sampleUnitDisk(random2D(idx, dim));
    //transform to world coordinates
    vec3 pos = targetParams.objToWorld * localPos + targetParams.position;

    //check which side is visible
    // -> ensure normal opposes observer
    vec3 normal = targetParams.normal;
    float side = sign(dot(normal, observer - pos));
    normal *= side;

    //edge case: observer looks exactly at edge of flat -> normal = 0.0
    bool valid = dot(normal, normal) != 0.0;

    //return sample
    float dist = distance(observer, pos);
    mat3 worldToObj = transpose(targetParams.objToWorld);
    return TargetSample(
        pos, normal, dist,                              //hit in world space
        localPos, vec3(0.0, 0.0, side),                 //hit in obj space
        targetParams.prob, valid,                       //prob + valid
        -worldToObj*targetParams.position, worldToObj
    );
}

TargetSample intersectTarget(vec3 observer, vec3 direction) {
    //convert to local space
    mat3 worldToObj = transpose(targetParams.objToWorld); //works since it's orthogonal
    vec3 localObs = worldToObj * (observer - targetParams.position);
    vec3 localDir = worldToObj * direction;

    //get intersection ray/xy-plane
    float t = -localObs.z / localDir.z;
    vec3 localPos = localObs + t * localDir;
    //check if valid hit
    bool valid = t > 0.0 && length(localPos.xy) <= targetParams.radius;

    //convert to world space
    vec3 pos = targetParams.objToWorld * localPos + targetParams.position;
    //create normal
    vec3 normal = targetParams.normal;
    float side = signBit(localObs.z);
    normal *= side;

    //assemble sample
    return TargetSample(
        pos, normal, t,                                 //hit in world space
        localPos, vec3(0.0, 0.0, side),                 //hit in object space
        targetParams.prob * float(valid), valid,        //prob + valid
        -worldToObj*targetParams.position, worldToObj   //worldToObj trafo
    );
}

bool isOccludedByTarget(vec3 observer) {
    //never occlude
    return false;
}

#endif
