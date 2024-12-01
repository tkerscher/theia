#ifndef _INCLUDE_TARGET_FLAT
#define _INCLUDE_TARGET_FLAT

#include "math.glsl"

layout(scalar) uniform TargetParams {
    float width;
    float height; //length
    vec3 offset;
    vec3 normal;

    float prob;
    mat3 objToWorld;
} targetParams;

TargetSample sampleTarget(vec3 observer, uint idx, inout uint dim) {
    //sample point on flat
    vec2 u = random2D(idx, dim);
    float localX = targetParams.width * (u.x - 0.5);
    float localY = targetParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);
    //transform to world coordinates
    vec3 pos = targetParams.objToWorld * localPos + targetParams.offset;

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
        pos, normal, dist,                          //hit in world space
        localPos, vec3(0.0, 0.0, side),             //hit in object space
        targetParams.prob, valid,                   //prob + valid
        -worldToObj*targetParams.offset, worldToObj
    );
}

TargetSample intersectTarget(vec3 observer, vec3 direction) {
    //convert to local space
    mat3 worldToObj = transpose(targetParams.objToWorld); //works since it's orthogonal
    vec3 localObs = worldToObj * (observer - targetParams.offset);
    vec3 localDir = worldToObj * direction;

    //get intersection ray/xy-plane
    float t = -localObs.z / localDir.z;
    vec3 localPos = localObs + t * localDir;
    //check if valid hit
    bool valid = t > 0.0 &&
        2.0 * abs(localPos.x) <= targetParams.width &&
        2.0 * abs(localPos.y) <= targetParams.height;
    
    //convert to world space
    vec3 pos = targetParams.objToWorld * localPos + targetParams.offset;
    //create normal
    vec3 normal = targetParams.normal;
    float side = signBit(localObs.z);
    normal *= side;

    //assemble sample
    return TargetSample(
        pos, normal, t,                             //hit in world space
        localPos, vec3(0.0, 0.0, side),             //hit in object space
        targetParams.prob * float(valid), valid,    //prob + valid
        -worldToObj*targetParams.offset, worldToObj //worldToObj trafo
    );
}

bool isOccludedByTarget(vec3 observer) {
    //we never occlude
    return false;
}

#endif
