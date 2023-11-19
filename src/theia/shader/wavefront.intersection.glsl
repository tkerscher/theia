#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_basic : require

#include "hits.glsl"
#include "scatter.surface.glsl"
#include "wavefront.common.glsl"

layout(local_size_x = LOCAL_SIZE) in;

#include "scene.glsl"

layout(scalar) readonly buffer IntersectionQueueBuffer {
    uint intersectionCount;
    IntersectionQueue intersectionQueue;
};
layout(scalar) writeonly buffer RayQueueBuffer {
    uint rayCount;
    RayQueue rayQueue;
};
layout(scalar) writeonly buffer HitQueueBuffer {
    uint hitCount;
    HitQueue hitQueue;
};

layout(scalar) uniform Params {
    TraceParams params;
};

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

void main() {
    //range check
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= intersectionCount)
        return;
    //load ray
    LOAD_RAY(ray, intersectionQueue.rays, idx)
    //just to be safe
    vec3 dir = normalize(ray.direction);

    //load rest of item
    int geometryIdx = intersectionQueue.geometryIdx[idx];
    int customIdx = intersectionQueue.customIdx[idx];
    int triangleIdx = intersectionQueue.triangleIdx[idx];
    vec2 barys = vec2(intersectionQueue.baryU[idx], intersectionQueue.baryV[idx]);
    mat4x3 obj2World = mat4x3(
        //note that glsl is column major
        intersectionQueue.obj2World00[idx],
        intersectionQueue.obj2World01[idx],
        intersectionQueue.obj2World02[idx],
        intersectionQueue.obj2World10[idx],
        intersectionQueue.obj2World11[idx],
        intersectionQueue.obj2World12[idx],
        intersectionQueue.obj2World20[idx],
        intersectionQueue.obj2World21[idx],
        intersectionQueue.obj2World22[idx],
        intersectionQueue.obj2World30[idx],
        intersectionQueue.obj2World31[idx],
        intersectionQueue.obj2World32[idx]
    );
    mat4x3 world2Obj = mat4x3(
        //note that glsl is column major
        intersectionQueue.world2Obj00[idx],
        intersectionQueue.world2Obj01[idx],
        intersectionQueue.world2Obj02[idx],
        intersectionQueue.world2Obj10[idx],
        intersectionQueue.world2Obj11[idx],
        intersectionQueue.world2Obj12[idx],
        intersectionQueue.world2Obj20[idx],
        intersectionQueue.world2Obj21[idx],
        intersectionQueue.world2Obj22[idx],
        intersectionQueue.world2Obj30[idx],
        intersectionQueue.world2Obj31[idx],
        intersectionQueue.world2Obj32[idx]
    );


    //reconstruct hit triangle
    Geometry geom = geometries[geometryIdx];
    ivec3 index = geom.indices[triangleIdx].idx;
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
    //vec3 worldPos = vec3(item.obj2World * vec4(objPos, 1.0));
    vec3 worldNrm = normalize(vec3(objNormal * world2Obj));
    geoNormal = normalize(vec3(geoNormal * world2Obj));
    
    //light models are generally unaware of the scene's geometry and might have
    //sampled a light ray inside a geometry
    //-> test against and discard
    bool inwards = dot(dir, worldNrm) <= 0.0; //normal and ray in opposite direction
    //address of expected ray medium
    uvec2 medium = inwards ? uvec2(geom.material.outside) : uvec2(geom.material.inside);
    if (ray.medium != medium)
        return; //discard

    //do matrix multiplication manually to improve error
    //See: https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/
    mat4x3 m = obj2World;
    precise vec3 worldPos;
    worldPos.x = m[3][0] + fma(m[0][0], objPos.x, fma(m[1][0], objPos.y, m[2][0] * objPos.z));
    worldPos.y = m[3][1] + fma(m[0][1], objPos.x, fma(m[1][1], objPos.y, m[2][1] * objPos.z));
    worldPos.z = m[3][2] + fma(m[0][2], objPos.x, fma(m[1][2], objPos.y, m[2][2] * objPos.z));
    //offset position to prevent self-intersection
    worldPos = offset_ray(worldPos, geoNormal);
    //calculate distance
    float dist = length(worldPos - ray.position);

    //update photons
    bool anyBelowMaxTime = false;
    float lambda = params.scatterCoefficient;
    float r[N_PHOTONS]; //reflectance
    for (int i = 0; i < N_PHOTONS; ++i) {
        //calculate reflectance
        r[i] = reflectance(geom.material, ray.photons[i], dir, worldNrm);

        //since we hit something the prob for the distance is to sample at
        //least dist -> p(d>=dist) = exp(-lambda*dist)
        //attenuation is simply Beer's law
        float mu_e = ray.photons[i].constants.mu_e;
        //write out multplication in hope to prevent catastrophic cancelation
        ray.photons[i].log_contrib += lambda * dist - mu_e * dist;

        //update travel time
        ray.photons[i].time += dist / ray.photons[i].constants.vg;
        //bound check
        if (ray.photons[i].time <= params.maxTime)
            anyBelowMaxTime = true;
    }
    //bounds check: max time
    if (!anyBelowMaxTime)
        return;
    
    //If hit target -> create response item
    if (customIdx == params.targetIdx &&
        (geom.material.flags & MATERIAL_TARGET_BIT) != 0
    ) {
        //create hits
        PhotonHit hits[N_PHOTONS];
        for (int i = 0; i < N_PHOTONS; ++i) {
            Photon ph = ray.photons[i];
            //attenuate by transmission
            float contrib = exp(ph.log_contrib) * ph.lin_contrib * (1.0 - r[i]);
            hits[i] = PhotonHit(ph.wavelength, ph.time, contrib);
        }
        //transform direction to object space
        vec3 objDir = normalize(mat3(world2Obj) * ray.direction);

        //count how many items we will be adding in this subgroup
        uint n = subgroupAdd(1);
        //elect one invocation to update counter in queue
        uint oldCount = 0;
        if (subgroupElect()) {
            oldCount = atomicAdd(hitCount, n);
        }
        //only the first(i.e. elected) one has correct oldCount value -> broadcast
        oldCount = subgroupBroadcastFirst(oldCount);
        //no we effectevily reserved us the range [oldCount..oldcount+n-1]

        //order the active invocations so each can write at their own spot
        uint id = subgroupExclusiveAdd(1);
        //create response item
        idx = oldCount + id;
        SAVE_HIT(objPos, objDir, objNormal, hits, hitQueue, idx)
    }

    //update photons
    //geometric effect (Lambert's cosine law)
    float lambert = -dot(dir, worldNrm);
    for (int i = 0; i < N_PHOTONS; ++i) {
        //and attenuate by reflectance
        ray.photons[i].lin_contrib *= r[i] * lambert;
    }

    //update ray
    ray.position = worldPos;
    ray.direction = normalize(reflect(dir, worldNrm));
    //skip four random numbers to match the amount
    //a volume scatter event would have drawn
    ray.rngCount += 4;

    //count how many items we will be adding in this subgroup
    uint n = subgroupAdd(1);
    //elect one invocation to update counter in queue
    uint oldCount = 0;
    if (subgroupElect()) {
        oldCount = atomicAdd(rayCount, n);
    }
    //only the first(i.e. elected) one has correct oldCount value -> broadcast
    oldCount = subgroupBroadcastFirst(oldCount);
    //no we effectevily reserved us the range [oldCount..oldcount+n-1]

    //order the active invocations so each can write at their own spot
    uint id = subgroupExclusiveAdd(1);
    //create item on the tracing queue
    idx = oldCount + id;
    SAVE_RAY(ray, rayQueue, idx)
}
