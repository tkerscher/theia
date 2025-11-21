#ifndef _INCLUDE_TARGET_GUIDE_COMMON
#define _INCLUDE_TARGET_GUIDE_COMMON

struct TargetGuideSample {
    vec3 dir;       ///< Sampled direction
    float dist;     ///< Max dist to trace
    float prob;     ///< Prob of sampled direction
};

TargetGuideSample createTargetGuideSample(
    vec3 observer,      ///< observer position
    vec3 targetPos,     ///< sample target position
    vec3 targetNrm,     ///< normal at targetPos
    float targetProb    ///< prob of sample (dA)
) {
    vec3 dir = targetPos - observer;
    float d2 = dot(dir, dir);
    dir = normalize(dir);

    float cosNormal = dot(dir, targetNrm);
    //calculate prob of sampled direction
    // -> apply jacobian r^2/d2 to transform area to angle distribution
    float prob = targetProb * d2 / abs(cosNormal);
    //for cosNormal near zero, we might get inf prob
    // -> mark sample as invalid by setting prob to zero
    if (isinf(prob)) prob = 0.0;
    //set to zero if if observer is on the wrong side
    prob *= float(cosNormal < 0.0);

    float dist = distance(targetPos, observer);
    return TargetGuideSample(dir, dist, prob);
}

TargetGuideSample createTargetGuideSample(
    vec3 dir,           ///< sampled direction
    float dist,         ///< sampled distance
    vec3 targetNrm,     ///< normal at sampled position
    float targetProb    ///< prob of sample (dA)
) {
    float cosNormal = dot(dir, targetNrm);
    //calculate prob of sampled direction
    // -> apply jacobian r^2/cos to transform area to angle distribution
    float prob = targetProb * dist * dist / abs(cosNormal);
    //for cosNormal near zero, we might get inf prob
    // -> mark sample as invalid by setting prob to zero
    if (isinf(prob)) prob = 0.0;
    //set to zero if if observer is on the wrong side
    prob *= float(cosNormal < 0.0);

    return TargetGuideSample(dir, dist, prob);
}

#endif
