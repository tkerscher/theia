#ifndef _INCLUDE_TARGET_SPHERE
#define _INCLUDE_TARGET_SPHERE

#include "math.glsl"
#include "math.sphere.glsl"
#include "util.sample.glsl"

layout(scalar) uniform TargetParams {
    vec3 position;
    float radius;

    vec3 invPos;
    float invRad;

    float hemisphereProb; // pre calc: 1.0 / (2pi * r^2)
} targetParams;

TargetSample sampleTarget(vec3 observer, uint idx, inout uint dim) {
    vec3 dir = normalize(observer - targetParams.position); //center -> observer
    float d = distance(observer, targetParams.position);

    //sample visible cone from the sphere's center
    float cosOpening = targetParams.radius / d;
    vec3 normal = createLocalCOSY(dir) * sampleDirectionCone(cosOpening, random2D(idx, dim));
    vec3 pos = normal * targetParams.radius + targetParams.position;

    //sample prob is 1 / area
    float prob = targetParams.hemisphereProb / (1.0 - cosOpening);
    //for very small cosOpening we might inf prob
    // -> set prob to zero to mark as invalid
    bool valid = !isinf(prob);
    if (!valid) prob = 0.0;

    //return sample
    float dist = distance(pos, observer);
    return TargetSample(
        pos, normal, dist,
        normal, normal,
        prob, valid,
        targetParams.invPos,
        mat3(targetParams.invRad)
    );
}

TargetSample intersectTarget(vec3 observer, vec3 direction) {
    //get nearest intersection
    Sphere sphere = Sphere(targetParams.position, targetParams.radius);
    float t = intersectSphere(sphere, observer, direction).x;
    //valid for 0.0 < t < +inf
    bool hit = (t > 0.0) && !isinf(t);

    if (hit) {
        //calculate position and normal
        vec3 pos = observer + direction * t;
        vec3 nrm = normalize(pos - targetParams.position);

        //calculate sample prob
        float d = distance(observer, targetParams.position);
        float cosOpening = targetParams.radius / d;
        float prob = targetParams.hemisphereProb / (1.0 - cosOpening);
        //again, for very small cosOpening prob might be inf -> mark as invalid
        bool valid = !isinf(prob);
        if (!valid) prob = 0.0;

        //return sample
        return TargetSample(
            pos, nrm, t,                                    //hit in world space
            nrm, nrm,                                       //hit in object space
            prob, valid,                                    //prob + valid
            targetParams.invPos,
            mat3(targetParams.invRad)
        );
    }
    else {
        //missed
        return createTargetSampleMiss();
    }
}

bool isOccludedByTarget(vec3 position) {
    //occluded if inside sphere
    return distance(position, targetParams.position) <= targetParams.radius;
}

#endif
