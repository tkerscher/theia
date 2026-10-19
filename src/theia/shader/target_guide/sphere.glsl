#ifndef _INCLUDE_TARGET_GUIDE_SPHERE
#define _INCLUDE_TARGET_GUIDE_SPHERE

#include "math.glsl"
#include "util/sample.glsl"

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
    float cosMin = sinMax * sinMax / (1.0 + sqrt(max((1.0 - sinMax) * (1.0 + sinMax), 0.0)));
    float prob = 1.0 / (TWO_PI * cosMin);
    //mark sample as invalid (prob = 0) if we're inside the sphere
    prob *= float(d > targetGuideParams.radius);

    //sample cone
    vec2 u = random2D(idx, dim);
    float t = cosMin * u.x;
    float cosTheta = 1.0 - t;
    float sinTheta = sqrt(max(t * (2.0 - t), 0.0));
    float phi = TWO_PI * u.y;
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
    float cosMin = sinMax * sinMax / (1.0 + sqrt(max((1.0 - sinMax) * (1.0 + sinMax), 0.0)));
    float prob = 1.0 / (TWO_PI * cosMin);
    //mark sample as invalid (prob = 0) if we're inside the sphere
    prob = d > targetGuideParams.radius ? prob : 0.0;

    //again as distance d + r suffice
    float dist = d + targetGuideParams.radius;

    //check if direction is within cone
    vec3 diff = viewDir - direction;
    prob = dot(diff, diff) <= 2.0 * cosMin ? prob : 0.0;

    //return sample
    return TargetGuideSample(direction, dist, prob);
}

#endif
