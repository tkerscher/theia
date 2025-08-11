#ifndef _INCLUDE_TARGET_GUIDE_DISK
#define _INCLUDE_TARGET_GUIDE_DISK

#include "util.sample.glsl"

uniform TargetGuideParams {
    vec3 position;
    float radius;
    vec3 normal;
    float prob;
    mat3 objToWorld;
} targetGuideParams;

TargetGuideSample sampleTargetGuide(vec3 observer, uint idx, inout uint dim) {
    //sample point on disk
    vec3 localPos = targetGuideParams.radius * sampleUnitDisk(random2D(idx, dim));
    //transform to world coordinates
    vec3 pos = targetGuideParams.objToWorld * localPos + targetGuideParams.position;

    //create sample
    return createTargetGuideSample(observer, pos, targetGuideParams.normal, targetGuideParams.prob);
}

TargetGuideSample evalTargetGuide(vec3 observer, vec3 direction) {
    //transform to local space
    mat3 worldToObj = transpose(targetGuideParams.objToWorld); //works since orthogonal
    vec3 localObs = worldToObj * (observer - targetGuideParams.position);
    vec3 localDir = worldToObj * direction;

    //get local hit
    float t = -localObs.z / localDir.z; //+/- inf or NaN if localDir.z = 0
    vec3 localHit = localObs + t * localDir;
    //get radius squared
    float r = length(localHit.xy); //ignore z
    bool valid =
        t > 0.0 &&  //ensure hit is in positive direction
        !isinf(t) && !isnan(t) && //ensure valid plane hit
        r <= targetGuideParams.radius; //within disk
    //set prob to 0.0 if invalid    
    float prob = targetGuideParams.prob * float(valid);

    //create sample
    return createTargetGuideSample(direction, t, targetGuideParams.normal, prob);
}

#endif
