#ifndef _INCLUDE_TARGET_SPHERE_INNER
#define _INCLUDE_TARGET_SPHERE_INNER

#include "math.glsl"
#include "math.sphere.glsl"
#include "util.sample.glsl"

layout(scalar) uniform TargetParams {
    vec3 position;
    float radius;

    vec3 invPos;
    float invRad;

    float prob;
} targetParams;

TargetSample sampleTarget(vec3 observer, uint idx, inout uint dim) {
    //every point is visible from within
    //unfortunately, importance sampling based on distance is mathematical difficult
    // -> for now sample uniformly
    vec3 normal = sampleUnitSphere(random2D(idx, dim));
    vec3 pos = targetParams.radius * normal + targetParams.position;

    //return sample
    float dist = distance(observer, pos);
    return TargetSample(
        pos, -normal, dist,
        normal, -normal,
        targetParams.prob, true,
        targetParams.invPos,
        mat3(targetParams.invRad)
    );
}

TargetSample intersectTarget(vec3 observer, vec3 direction) {
    //get nearest intersection
    Sphere sphere = Sphere(targetParams.position, targetParams.radius);
    float t = intersectSphere(sphere, observer, direction).y; //.x points in opposite direction
    //valid for 0.0 < t < +inf
    bool hit = (t > 0.0) && !isinf(t);

    //calculate position and normal
    vec3 pos = observer + direction * t;
    vec3 nrm = normalize(targetParams.position - pos);
    vec3 hitPos = -nrm;
    float prob = targetParams.prob * float(hit);
    //return sample
    return TargetSample(
        pos, nrm, t,
        hitPos, nrm,
        targetParams.prob, hit,
        targetParams.invPos,
        mat3(targetParams.invRad)
    );
}

bool isOccludedByTarget(vec3 position) {
    return distance(position, targetParams.position) >= targetParams.radius;
}

#endif
