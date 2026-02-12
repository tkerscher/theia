#ifndef _SCENE_INCLUDE
#define _SCENE_INCLUDE

/**
 * Structure describing the intersection of a ray query with geometry.
 * Includes information about hit both in world space (where the ray lives)
 * and object space (where geometry is defined), i.e. before transformation
 * applied.
*/
struct SurfaceHit {
    //Geometries mark a boundary between an inside and an outside volume.
    //Their normals distinguish them by pointing outwards.
    //Materials finally assign both volumes a medium.

    uint materialIdx;           ///< Material of the intersected geometry
    uint flags;                 ///< Material flags for specific direction (inward)
    uint otherMediumIdx;        ///< Index of the medium on the other side
    bool inward;                ///< Direction of ray respective to geometry

    //Variables defined in world space, i.e. the same as the ray

    precise vec3 worldPos;      ///< Hit position in world space
    // vec3 worldNrm;              ///< Geometry normal at hit position in world space
    vec3 rayNrm;                ///< surface normal opposing ray direction

    //The following variables are define object space, i.e. geometry's
    //coordinates before any transformation were applied

    precise vec3 objPos;        ///< Hit position in object space
    precise vec3 objNrm;        ///< Geometry normal at hit position in object space
    vec3 objDir;                ///< Ray direction in object space

    //Lastly, we may need to transform from world to object space.
    //We skip the translating part for now.

    mat3 worldToObj;            ///< Transformation from world to object space
};

#endif
