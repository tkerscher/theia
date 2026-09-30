#ifndef _INCLUDE_TRACER_SCENE_PORTAL
#define _INCLUDE_TRACER_SCENE_PORTAL

/**
 * Portal / multi-scene support for the forward scene target tracers.
 *
 * A "portal" is a surface carrying MATERIAL_PORTAL_BIT. When the surface model
 * TRANSMITS a ray across it (transmitRay / crossBorder), the photon is switched
 * to another scene / coordinate frame.
 *
 * The crossing is a finite-state machine: the photon carries
 *   frame = uvec2(sceneId, context)   (sceneId 0 = main).
 * Each scene corresponds to a TLAS as given by the SubTlasTable.
 *
 * The TransitionTable maps (sceneId, context, gl_InstanceID) -> (nextScene,
 * nextContext, T), where T is the object->world placement of the ARRIVAL box in
 * the target scene. `portalReframe` re-expresses the (already surface-processed)
 * ray in the new frame.
 */

#include "util/offset.glsl"
#include "util/buffers.glsl"

//Sub-scene TLAS device addresses, indexed by (sceneId - 1).
readonly buffer SubTlasTable { uvec2 subTlas[]; };
//Number of instances per scene, i.e. the row stride of TransitionTable and
//ObjectIdTable.
readonly buffer InstanceCountTable { uint instanceCount[]; };
//Per scene: device address of a uint[] transition table, 14 words per record:
//[0]=nextScene, [1]=nextContext, [2..13]=T (mat4x3, column-major, 12 floats).
//One record per (context, instance), non-portal instances hold an unused zero
//record.
readonly buffer TransitionTable { uvec2 transAdr[]; };
//Per scene: device address of an int[] holding the detector ids of that scene
//per (context, instance). Replaces the flat ObjectIdMap a plain Scene binds.
readonly buffer ObjectIdTable { uvec2 objectIdAdr[]; };

//Select the acceleration structure address for the photon's current frame.
//sceneId 0 -> main TLAS (passed in, i.e. params.tlas); else the sub-scene's TLAS.
uvec2 portalSelectTlas(uvec2 frame, uvec2 mainTlas) {
    return frame.x == 0u ? mainTlas : subTlas[frame.x - 1u];
}

#ifdef PORTAL_HIT_STAGE

//error estimation helpers shared with resolveIntersection()
#include "scene/intersect.glsl"

//objectId for the current hit in the photon's current scene AND context.
int portalObjectId(uvec2 frame) {
    uint idx = frame.y * instanceCount[frame.x] + uint(gl_InstanceID);
    return IntBuffer(objectIdAdr[frame.x]).values[idx];
}

//Stride of one transition record, in uint words.
const uint PORTAL_TRANSITION_STRIDE = 14u;

//A single decoded portal transition.
struct PortalTransition {
    uint nextScene;
    uint nextContext;
    mat4x3 transform;   //object->world placement of the arrival box
};

//Look up the transition for a portal hit in the photon's current frame.
PortalTransition lookupTransition(uint sceneId, uint context, uint instance) {
    uint stride = instanceCount[sceneId];
    uint base = (context * stride + instance) * PORTAL_TRANSITION_STRIDE;
    UIntBuffer tb = UIntBuffer(transAdr[sceneId]);

    PortalTransition t;
    t.nextScene = tb.values[base + 0];
    t.nextContext = tb.values[base + 1];
    t.transform = mat4x3(
        uintBitsToFloat(tb.values[base +  2]), uintBitsToFloat(tb.values[base +  3]),
        uintBitsToFloat(tb.values[base +  4]), uintBitsToFloat(tb.values[base +  5]),
        uintBitsToFloat(tb.values[base +  6]), uintBitsToFloat(tb.values[base +  7]),
        uintBitsToFloat(tb.values[base +  8]), uintBitsToFloat(tb.values[base +  9]),
        uintBitsToFloat(tb.values[base + 10]), uintBitsToFloat(tb.values[base + 11]),
        uintBitsToFloat(tb.values[base + 12]), uintBitsToFloat(tb.values[base + 13])
    );
    return t;
}

/**
 * Switch the ray to the portal's target scene/frame AFTER the surface model has
 * transmitted/crossed it. `ray.direction` must already hold the WORLD-space outgoing
 * direction the surface produced (refracted, or unchanged for a border); `ray.mediumIdx`
 * must already be set by the surface. This rewrites position, direction and frame:
 *   - direction is re-expressed in the target frame via T * worldToObj (rotation only)
 *   - the object-space position of the hit is mapped by the arrival transform T and offset 
 *     into the transmitted side; the previous world position is discarded
 */
void portalReframe(inout ForwardRay ray, const SurfaceHit hit) {
    PortalTransition t = lookupTransition(ray.frame.x, ray.frame.y, uint(gl_InstanceID));
    mat4x3 T = t.transform;
    //world -> object transformation of the arrival box
    mat3 Tinv = inverse(mat3(T));
    mat4x3 w2o = mat4x3(Tinv[0], Tinv[1], Tinv[2], -(Tinv * T[3]));

    //hit position in the target frame
    vec3 pos = transformPosition(T, hit.objPos);
    //direction of travel through the surface (object space)
    vec3 fwd = hit.inward ? -hit.objNrm : hit.objNrm;
    vec3 nrm = fwd * Tinv;
    float scale = inversesqrt(dot(nrm, nrm));
    nrm *= scale;

    //offset from error estimate, see resolveIntersection()
    float objOffset = hit.objOffset
        + dot(world2ObjError(w2o, pos, vec3(0.0)), abs(hit.objNrm));
    float worldOffset = dot(obj2WorldError(T, hit.objPos), abs(nrm));
    float offset = fmaKHR(scale, objOffset, worldOffset);

    ray.direction = normalize(mat3(T) * (hit.worldToObj * ray.direction));
    ray.position  = offsetRay(pos, nrm, offset);
    ray.frame     = uvec2(t.nextScene, t.nextContext);
}

#endif //PORTAL_HIT_STAGE

#endif
