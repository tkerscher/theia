#ifndef _INCLUDE_TARGET_GUIDE_SPHERE
#define _INCLUDE_TARGET_GUIDE_SPHERE

#include "math.glsl"
#include "util.sample.glsl"

uniform TargetGuideParams {
    vec3 position;
    float radius;
} targetGuideParams;

//Here we loosely follow PBRT Ed. 4 by Pharr et al., Chapter 6.2.4 to sample
//sphere with the key difference, that we sample the backside of the sphere

TargetGuideSample sampleTargetGuide(vec3 observer, uint idx, inout uint dim) {
    //get opening angle of visible cone
    float d = distance(targetGuideParams.position, observer);
    vec3 viewDir = normalize(targetGuideParams.position - observer);
    float sinMax = targetGuideParams.radius / d;
    float sin2Max = sinMax * sinMax;
    float cosMin = 1.0 - sqrt(max(1.0 - sin2Max, 0.0));
    //for narrow cones we run out of numerical precision (around 1)
    //causing cosMin to always be zero
    // -> use taylor approximation instead
    if (sin2Max < 0.00068523f /* sin^2(1.5 deg) */) {
        //sqrt(1-x^2) ~= 1 - x^2/2 + O(x^4)
        // -> 1 - sqrt(1-x^2) ~= x^2/2
        cosMin = 0.5 * sin2Max;
    }
    float prob = 1.0 / (TWO_PI * cosMin);
    //mark sample as invalid (prob = 0) if we're inside the sphere
    prob *= float(d > targetGuideParams.radius);

    //sample cone
    float cosTheta = 1.0 - cosMin * random(idx, dim);
    float sinTheta = sqrt(max(1.0 - cosTheta*cosTheta, 0.0));
    float phi = TWO_PI * random(idx, dim);
    vec3 dir = createLocalCOSY(viewDir) * vec3(
        sinTheta * sin(phi),
        sinTheta * cos(phi),
        cosTheta
    );

    //we don't need the exact distance to the backside of the sphere
    //any larger value will suffice
    // -> use constant d + r
    float dist = d + targetGuideParams.radius;

    //return sample
    return TargetGuideSample(dir, dist, prob);
}

TargetGuideSample evalTargetGuide(vec3 observer, vec3 direction) {
    //get opening angle of visible cone
    float d = distance(targetGuideParams.position, observer);
    vec3 viewDir = normalize(targetGuideParams.position - observer);
    float sinMax = targetGuideParams.radius / d;
    float sin2Max = sinMax * sinMax;
    float cosMin = 1.0 - sqrt(max(1.0 - sin2Max, 0.0));
    //for narrow cones we run out of numerical precision (around 1)
    //causing cosMin to always be zero
    // -> use taylor approximation instead
    if (sin2Max < 0.00068523f /* sin^2(1.5 deg) */) {
        //sqrt(1-x^2) ~= 1 - x^2/2 + O(x^4)
        // -> 1 - sqrt(1-x^2) ~= x^2/2
        cosMin = 0.5 * sin2Max;
    }
    float prob = 1.0 / (TWO_PI * cosMin);
    //mark sample as invalid (prob = 0) if we're inside the sphere
    prob *= float(d > targetGuideParams.radius);

    //again as distance d + r suffice
    float dist = d + targetGuideParams.radius;

    //check if direction is within cone
    float cosDir = dot(viewDir, direction);
    prob *= float(cosMin >= 1.0 - cosDir);

    //return sample
    return TargetGuideSample(direction, dist, prob);
}

#endif
