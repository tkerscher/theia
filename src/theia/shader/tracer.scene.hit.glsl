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
    float u, //random number
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
    //fetch flags
    uint flags = inward ? mat.flagsInwards : mat.flagsOutwards;

    //update samples
    float dist = length(worldPos - ray.position);
    success = updateSamples(ray, dist, params, false, true);
    if (CHECK_BRANCH(!success))
        return true;

    //calculate reflectance
    float r[N_LAMBDA];
    vec3 dir = normalize(ray.direction);
    float lambert = abs(dot(dir, worldNrm)); //geometric effect
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        r[i] = reflectance(
            mat,
            ray.samples[i].wavelength,
            ray.samples[i].constants.n,
            dir, worldNrm
        );
    }

    //check if target -> implicit no transmission
    bool isTarget = (flags & MATERIAL_DETECTOR_BIT) != 0;
    //check for black body flag
    bool black = (flags & MATERIAL_BLACK_BODY_BIT) != 0;

    //If hit target -> create response item
    int customId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    bool hitTarget = customId == targetId && isTarget;
    if (subgroupAny(hitTarget) && hitTarget) {
    // if (customId == targetId && (flags & MATERIAL_DETECTOR_BIT) != 0) {
        //process hits
        [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
            //skip out of bound samples
            if (ray.samples[i].time > params.maxTime)
                continue;

            Sample s = ray.samples[i];
            float contrib = exp(s.log_contrib) * s.lin_contrib;
            if (!black) {
                //attenuate by transmission if not completly absorbing
                contrib *= (1.0 - r[i]);
            }
            response(HitItem(
                objPos, objDir, objNrm,
                s.wavelength,
                s.time,
                contrib
            ));
        }
    }

    //check for black flag
    if (black) {
        //stop tracing
        return true;
    }

    //return early if we should not update the ray
    if (CHECK_BRANCH(!update)) {
        return false; //dont abort tracing
    }

#ifndef DISABLE_VOLUME_BORDER
    //Check for volume border (should never be a target)
    if ((flags & MATERIAL_VOLUME_BORDER_BIT) != 0) {
        //put ray on other side to prevent self intersection
        ray.position = offsetRay(worldPos, -geomNormal);
        //update medium & constants in sample and get going
        Medium medium = inward ? mat.inside : mat.outside;
        ray.medium = uvec2(medium);
        [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
            ray.samples[i].constants = lookUpMedium(medium, ray.samples[i].wavelength);
        }
        //done -> early return
        return false; //dont abort tracing
    }
#endif

#ifndef DISABLE_TRANSMISSION
    //check if we should reflect or transmit
    bool canReflect = (flags & MATERIAL_NO_REFLECT_BIT) == 0;
    bool canTransmit = (flags & MATERIAL_NO_TRANSMIT_BIT) == 0 && !isTarget;
    //either reflect, transmit, or stop if neither
    if (canReflect && (!canTransmit || u <= r[0])) {
        //update ray
        ray.direction = normalize(reflect(dir, worldNrm));
        //offset ray away from surface to prevent self intersection
        ray.position = offsetRay(worldPos, geomNormal);

        //update samples
        if (canTransmit) {
            //first sample importance sampling cancels reflectance -> do exact
            ray.samples[0].lin_contrib *= lambert;       
            [[unroll]] for (uint i = 1; i < N_LAMBDA; ++i) {
                //r[0] is the prob of reflecting the ray
                ray.samples[i].lin_contrib *= r[i] / r[0] * lambert;
            }
        }
        else {
            //no random choice happened
            [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
                ray.samples[i].lin_contrib *= r[i] * lambert;
            }
        }
    }
    else if (canTransmit) {
        //due to dispersion, we can only transmit the first sample
        // -> set others to zero contribution
        [[unroll]] for (uint i = 1; i < N_LAMBDA; ++i) {
            ray.samples[i].lin_contrib = 0.0;
        }

        //store inward refractive index
        float n_i = ray.samples[0].constants.n;
        //update constants
        Medium medium = inward ? mat.inside : mat.outside;
        ray.medium = uvec2(medium);
        ray.samples[0].constants = lookUpMedium(medium, ray.samples[0].wavelength);
        //store outward refractive index
        float n_o = ray.samples[0].constants.n;

        //calculate new ray direction
        float eta = n_i / n_o;
        ray.direction = refract(ray.direction, worldNrm, eta);
        //offset ray to prevent self intersection
        ray.position = offsetRay(worldPos, -geomNormal);

        //update sample contribution:
        //the factor (1-r) cancels with importance sampling the reflection
        //leaving only a factor of eta^-2 for the radiance due to rays getting
        //denser
        float inv_eta = n_o / n_i;
        ray.samples[0].lin_contrib *= inv_eta * inv_eta * lambert;
        if (!canReflect) {
            //no random choice happen -> need to also account for transmission factor
            ray.samples[0].lin_contrib *= (1.0 - r[0]);
        }
    }
    else {
        //no way to proceed -> abort tracing
        return true;
    }
#else
    //transmission disables -> only do reflection
    ray.direction = normalize(reflect(dir, worldNrm));
    //offset ray away from surface to prevent self intersection
    ray.position = offsetRay(worldPos, geomNormal);

    //update samples
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        ray.samples[i].lin_contrib *= r[i] * lambert;
    }
#endif

    //done
    return false; //dont abort tracing
}

#endif