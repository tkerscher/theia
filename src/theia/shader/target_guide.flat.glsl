#ifndef _INCLUDE_TARGET_GUIDE_FLAT
#define _INCLUDE_TARGET_GUIDE_FLAT

#include "math.glsl"

uniform TargetGuideParams {
    float width;
    float height;
    vec3 position;
    vec3 normal;
    float prob;
    mat3 objToWorld;
} targetGuideParams;

TargetGuideSample sampleTargetGuide(vec3 observer, uint idx, inout uint dim) {
    //sample point on flat
    vec2 u = random2D(idx, dim);
    float localX = targetGuideParams.width * (u.x - 0.5);
    float localY = targetGuideParams.height * (u.y - 0.5);
    vec3 localPos = vec3(localX, localY, 0.0);

    //transform to world coordinates
    vec3 pos = targetGuideParams.objToWorld * localPos + targetGuideParams.position;

    //return sample
    return createTargetGuideSample(observer, pos, targetGuideParams.normal, targetGuideParams.prob);
}

TargetGuideSample evalTargetGuide(vec3 observer, vec3 direction) {
    //transform to local space
    mat3 worldToObj = transpose(targetGuideParams.objToWorld); //works since orthogonal
    vec3 localObs = worldToObj * (observer - targetGuideParams.position);
    vec3 localDir = worldToObj * direction;

    //get intersection ray/xy-plane
    float t = -localObs.z / localDir.z; //+/- inf or NaN if localDir.z = 0
    vec3 localPos = localObs + t * localDir;
    //check if valid hit
    bool valid = t > 0.0 &&
        !isinf(t) && !isnan(t) &&
        2.0 * abs(localPos.x) <= targetGuideParams.width &&
        2.0 * abs(localPos.y) <= targetGuideParams.height;
    //set prob to zero if invalid
    float prob = targetGuideParams.prob * float(valid);

    //return sample
    return createTargetGuideSample(direction, t, targetGuideParams.normal, prob);
}

#endif
