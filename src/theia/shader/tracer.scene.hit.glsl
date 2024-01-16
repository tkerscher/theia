#ifndef _INCLUDE_TRACER_SCENE_HIT
#define _INCLUDE_TRACER_SCENE_HIT

#include "ray.propagate.glsl"
#include "scatter.surface.glsl"
#include "scene.intersect.glsl"
#include "util.branch.glsl"

#include "response.common.glsl"
//user provided code
#include "response.glsl"

bool processHit(
    inout Ray ray,
    rayQueryEXT rayQuery,
    uint targetId,
    const PropagateParams params,
    bool update
) {
    //process ray query
    vec3 objPos, objNrm, objDir, worldPos, worldNrm, geomNormal;
    Material mat;
    bool inward;
    bool success = processRayQuery(
        ray, rayQuery,
        mat, inward,
        objPos, objNrm, objDir,
        worldPos, worldNrm,
        geomNormal
    );
    if (CHECK_BRANCH(!success))
        return true; //abort tracing

    //offset position to prevent self-intersection
    if (!inward)
        geomNormal = -geomNormal;
    worldPos = offsetRay(worldPos, geomNormal);
    //calculate distance
    float dist = length(worldPos - ray.position);

    //update samples
    success = updateSamples(ray, dist, params, false, true);
    if (CHECK_BRANCH(!success))
        return true;

    //calculate reflectance (needed for both branches)
    float r[N_LAMBDA];
    vec3 dir = normalize(ray.direction);
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        r[i] = reflectance(
            mat,
            ray.samples[i].wavelength,
            ray.samples[i].constants.n,
            dir, worldNrm
        );
    }

    //If hit target -> create response item
    int customId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    if (customId == targetId && (mat.flags & MATERIAL_TARGET_BIT) != 0) {
        //process hits
        [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
            //skip out of bound samples
            if (ray.samples[i].time > params.maxTime)
                continue;

            Sample s = ray.samples[i];
            //attenuate by transmission
            float contrib = exp(s.log_contrib) * s.lin_contrib * (1.0 - r[i]);
            response(HitItem(
                objPos, objDir, objNrm,
                s.wavelength,
                s.time,
                contrib
            ));
        }
    }

    //update if needed
    if (CHECK_BRANCH(update)) {
        //geometric effect (Lambert's cosine law)
        float lambert = -dot(dir, worldNrm);
        [[unroll]] for (int i = 0; i < N_LAMBDA; ++i) {
            ray.samples[i].lin_contrib *= r[i] * lambert;
        }
        //update ray
        ray.position = worldPos;
        ray.direction = normalize(reflect(dir, worldNrm));
    }

    //done
    return false; //dont abort tracing
}

#endif
