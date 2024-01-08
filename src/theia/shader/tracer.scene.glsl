#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require

//check expected macros
#ifndef BATCH_SIZE
#error "BATCH_SIZE not defined"
#endif
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif
#ifndef N_LAMBDA
#error "N_LAMBDA not defined"
#endif
#ifndef N_SCATTER
#error "N_SCATTER not defined"
#endif
#ifndef DIM_OFFSET
#error "DIM_OFFSET not defined"
#endif
// #samples per iteration
#define DIM_STRIDE 5

layout(local_size_x = BLOCK_SIZE) in;

#include "material.glsl"
#include "scatter.surface.glsl"
#include "scatter.volume.glsl"
#include "scene.glsl"
#include "sphere.glsl"

#include "lightsource.common.glsl"
#include "response.common.glsl"
//user provided code
#include "rng.glsl"
#include "source.glsl"
#include "response.glsl"

struct Sample {
    float wavelength;
    float time;
    float lin_contrib;
    float log_contrib;
    MediumConstants constants;
};
struct Ray {
    vec3 position;
    vec3 direction;
    uvec2 medium;
    Sample samples[N_LAMBDA];
};

uniform accelerationStructureEXT tlas;
layout(scalar) readonly buffer Detectors {
    Sphere detectors[];
};

layout(scalar) uniform TraceParams {
    uint targetIdx;

    float scatterCoefficient;
    uvec2 sceneMedium;

    float maxTime;
    vec3 lowerBBoxCorner;
    vec3 upperBBoxCorner;
} params;

//Taken from Ray Tracing Gems: Chapter 6
// Normal points outward for rays exiting the surface, else is flipped.
vec3 offset_ray(vec3 p, vec3 n) {
    ivec3 of_i = ivec3(256.0 * n);
    
    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x)+((p.x < 0.0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y)+((p.y < 0.0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z)+((p.z < 0.0) ? -of_i.z : of_i.z))
    );

    return vec3(
        abs(p.x) < (1.0 / 32.0) ? p.x+ (1.0/65536.0)*n.x : p_i.x,
        abs(p.y) < (1.0 / 32.0) ? p.y+ (1.0/65536.0)*n.y : p_i.y,
        abs(p.z) < (1.0 / 32.0) ? p.z+ (1.0/65536.0)*n.z : p_i.z
    );
}

bool processHit(inout Ray ray, rayQueryEXT rayQuery, bool update) {
    //since shadow rays are sampled from a sphere, they might not hit the
    //detector -> check for hit so we dont produce garbage
    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionTriangleEXT)
        return true; //abort tracing; should only happen with shadow rays where it's ignored

    vec3 dir = normalize(ray.direction);    
    //fetch info about intersection
    int instanceId = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    int triangleId = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    vec2 barys = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    //reconstruct hit triangle
    Geometry geom = geometries[instanceId];
    ivec3 index = geom.indices[triangleId].idx;
    Vertex v0 = geom.vertices[index.x];
    Vertex v1 = geom.vertices[index.y];
    Vertex v2 = geom.vertices[index.z];
    precise vec3 e1 = v1.position - v0.position;
    precise vec3 e2 = v2.position - v0.position;
    precise vec3 objPos = v0.position + fma(vec3(barys.x), e1, barys.y * e2);
    vec3 geoNormal = cross(e1, e2); //used for offsetting
    //interpolate normal
    precise vec3 n1 = v1.normal - v0.normal;
    precise vec3 n2 = v2.normal - v0.normal;
    precise vec3 objNormal = v0.normal + fma(vec3(barys.x), n1, barys.y * n2);
    //translate from object to world space
    mat4x3 world2Obj = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    vec3 worldNrm = normalize(vec3(objNormal * world2Obj));
    geoNormal = normalize(vec3(geoNormal * world2Obj));

    //light models are generally unaware of the scene's geometry and might have
    //sampled a light ray inside a geometry
    //-> test against and discard
    bool inwards = dot(dir, worldNrm) <= 0.0; //normal and ray in opposite direction
    //address of expected ray medium
    uvec2 medium = inwards ? uvec2(geom.material.outside) : uvec2(geom.material.inside);
    if (ray.medium != medium)
        return true; //abort tracing
    
    //do matrix multiplication manually to improve error
    //See: https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
    mat4x3 m = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    precise vec3 worldPos;
    worldPos.x = m[3][0] + fma(m[0][0], objPos.x, fma(m[1][0], objPos.y, m[2][0] * objPos.z));
    worldPos.y = m[3][1] + fma(m[0][1], objPos.x, fma(m[1][1], objPos.y, m[2][1] * objPos.z));
    worldPos.z = m[3][2] + fma(m[0][2], objPos.x, fma(m[1][2], objPos.y, m[2][2] * objPos.z));
    //offset position to prevent self-intersection
    worldPos = offset_ray(worldPos, geoNormal);
    //calculate distance
    float dist = length(worldPos - ray.position);

    //update samples
    bool anyBelowMaxTime = false;
    float lambda = params.scatterCoefficient;
    float r[N_LAMBDA]; //reflectance;
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        //calculate reflectance
        r[i] = reflectance(
            geom.material,
            ray.samples[i].wavelength,
            ray.samples[i].constants.n,
            dir, worldNrm
        );

        //since we hit something the prob for the distance is to sample at
        //least dist -> p(d>=dist) = exp(-lambda*dist)
        //attenuation is simply Beer's law
        float mu_e = ray.samples[i].constants.mu_e;
        //write out multplication in hope to prevent catastrophic cancelation
        ray.samples[i].log_contrib += lambda * dist - mu_e * dist;

        //update travel time
        ray.samples[i].time += dist / ray.samples[i].constants.vg;
        //bound check
        if (ray.samples[i].time <= params.maxTime)
            anyBelowMaxTime = true;
    }
    //bounds check: max time
    if (!anyBelowMaxTime)
        return true; //abort tracing

    //If hit target -> create response item
    int customId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    if (customId == params.targetIdx &&
        (geom.material.flags & MATERIAL_TARGET_BIT) != 0
    ) {
        //transform direction to object space
        vec3 objDir = normalize(mat3(world2Obj) * ray.direction);
        objNormal = normalize(objNormal);

        //process hits
        [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
            //skip out of bound samples
            if (ray.samples[i].time > params.maxTime)
                continue;

            Sample s = ray.samples[i];
            //attenuate by transmission
            float contrib = exp(s.log_contrib) * s.lin_contrib * (1.0 - r[i]);
            response(HitItem(
                objPos, objDir, objNormal,
                s.wavelength,
                s.time,
                contrib
            ));
        }
    }

    //update if needed
    if (subgroupAll(update) || (!subgroupAll(!update) && update)) {
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

bool processScatter(
    inout Ray ray,
    float dist,
    rayQueryEXT rayQuery,
    uint idx, uint dim, // rng coords
    out Ray hitRay
) {
    //update position
    ray.position = ray.position + ray.direction * dist;

    //boundary check
    if (any(lessThan(ray.position, params.lowerBBoxCorner)) ||
        any(greaterThan(ray.position, params.upperBBoxCorner)))
    {
        return true; //abort tracing
    }

    //update samples
    float lambda = params.scatterCoefficient;
    bool anyBelowMaxTime = false;
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        //update throughput, i.e. transmission/probability, for all wavelengths

        //prob travel distance is lambda*exp(-lambda*dist)
        //  -> we split the linear from the exp part for the update
        //attenuation is exp(-mu_e*dist)
        //  -> log(delta) = (mu_e-lambda)*dist
        float mu_e = ray.samples[i].constants.mu_e;
        //write out multiplication in hope to prevent catastrophic cancelation
        ray.samples[i].log_contrib += lambda * dist - mu_e * dist;
        ray.samples[i].lin_contrib /= lambda;

        //scattering is mu_s * p(theta) -> mu_s shared by all processes
        float mu_s = ray.samples[i].constants.mu_s;
        ray.samples[i].lin_contrib *= mu_s;

        //update traveltime
        ray.samples[i].time += dist / ray.samples[i].constants.vg;
        //bounds check
        if (ray.samples[i].time <= params.maxTime)
            anyBelowMaxTime = true;
    }
    //bounds check: max time
    if (!anyBelowMaxTime)
        return true; //abort tracing
    
    /***************************************************************************
     * MIS: sample both scattering phase function & detector                   *
     *                                                                         *
     * We'll use the following naming scheme: pXY, where                       *
     * X: prob, distribution                                                   *
     * Y: sampled distribution                                                 *
     * S: scatter, D: detector                                                 *
     * e.g. pDS: p_det(dir ~ scatter)                                          *
     **************************************************************************/

    //sample detector
    Sphere detector = detectors[params.targetIdx];
    vec2 rng = random2D(idx, dim); dim += 2;
    float pDD, detDist;
    vec3 detDir = sampleSphere(detector, ray.position, rng, detDist, pDD);

    //sample scatter phase function
    Medium medium = Medium(ray.medium);
    rng = random2D(idx, dim); dim += 2;
    float pSS;
    vec3 scatterDir = scatter(medium, ray.direction, rng, pSS);

    //calculate cross probs pSD, pDS
    float pSD = scatterProb(medium, ray.direction, detDir);
    float pDS = sampleSphereProb(detector, ray.position, scatterDir);
    //calculate MIS weights
    float w_scatter = pSS*pSS / (pSS*pSS + pSD*pSD);
    float w_det = pDD*pSD / (pDD*pDD + pDS*pDS);
    //^^^ For the detector weight, two effects happen: attenuation due to phase
    //    phase function (= pSD) and canceling of sampled distribution:
    //      f(x)*phase(theta)/pDD * w_det = f(x)*pSD/pDD * w_det
    //    Note that mu_s was already applied

    //update ray; create copy for shadow ray
    hitRay = ray;
    ray.direction = scatterDir;
    hitRay.direction = detDir;
    //update samples
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        //udpate scattered ray
        ray.samples[i].lin_contrib *= w_scatter;
        //update detector ray
        hitRay.samples[i].lin_contrib *= w_det;
    }

    //trace shadow ray
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        hitRay.position,
        0.0,
        detDir,
        detDist);
    rayQueryProceedEXT(rayQuery);

    //done
    return false; //dont abort trace
}

bool trace(inout Ray ray, uint idx, uint dim) {
    //just to be safe
    vec3 dir = normalize(ray.direction);
    //sample distance
    float u = random(idx, dim); dim++;
    float dist = -log(1.0 - u) / params.scatterCoefficient;

    //trace ray
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(
        rayQuery, tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF, //mask -> hit anything
        ray.position,
        0.0, //t_min; self-intersections are handled via offsets
        dir,
        dist);
    rayQueryProceedEXT(rayQuery);
    //check if we hit anything
    bool hit = rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT;

    //volume scattering produces shadow rays aimed at the detector, which are
    //handled similar to direct hits. To reduce divergence, we handle them both
    //the same code.

    //create a local copy of ray, as it might either be the original one or
    //the shadow ray
    Ray hitRay;
    //check hit across subgroup, giving change to skip unnecessary code
    if (subgroupAll(!hit)) {
        //returns true, if we should abort tracing
        if (processScatter(ray, dist, rayQuery, idx, dim, hitRay))
            return true; //abort tracing
    }
    else if (subgroupAll(hit)) {
        hitRay = ray;
    }
    else {
        //mixed branching
        if (!hit) {
            //returns true, if we should abort tracing
            if (processScatter(ray, dist, rayQuery, idx, dim, hitRay))
                return true; //abort tracing
        }
        else {
            hitRay = ray;
        }
    }

    //handle hit: either directly from tracing or indirect via shadow ray
    //will also create a hit item in the queue (if one was created)
    bool hitResult = processHit(hitRay, rayQuery, hit);

    //copy hitRay back to ray if necessary
    if (subgroupAll(hit)) {
        //if direct hit -> update ray for tracing
        ray = hitRay;
        return hitResult;
    }
    else if (subgroupAll(!hit)) {
        //always continue after volume scattering
        return false; //dont abort tracing
    }
    else {
        //mixed branching
        if (hit) {
            ray = hitRay;
            return hitResult;
        }
        else {
            return false; //dont abort tracing
        }
    }
}

Ray sampleRay(uint idx) {
    SourceRay source = sampleLight(idx);  

    //transform source to tracing ray
    Medium medium = Medium(params.sceneMedium);
    Sample samples[N_LAMBDA];
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        samples[i] = Sample(
            source.samples[i].wavelength,
            source.samples[i].startTime,
            source.samples[i].contrib,
            0.0,
            lookUpMedium(medium, source.samples[i].wavelength));
    }
    return Ray(
        source.position,
        source.direction,
        params.sceneMedium,
        samples
    );    
}

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= BATCH_SIZE)
        return;
    
    //sample ray
    Ray ray = sampleRay(idx);

    //trace loop
    uint dim = DIM_OFFSET;
    [[unroll]] for (uint i = 0; i < N_SCATTER; ++i, dim += DIM_STRIDE) {
        //trace() returns true, if we should stop tracing
        if (trace(ray, idx, dim)) return;
    }
}
