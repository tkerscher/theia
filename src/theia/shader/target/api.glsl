#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rest of the code interacts with
//targets.

/**
 * Samples a hit point on the target visible from the given observer.
*/
TargetSample sampleTarget(
    vec3 observer,              ///< Observer that wants to hit the target
    uint idx, inout uint dim    ///< RNG state
);

/**
 * Checks whether an infinite ray given by position and direction intersects the
 * target.
*/
TargetSample intersectTarget(
    vec3 observer,              ///< origin of ray
    vec3 direction              ///< direction of ray
);

/**
 * Returns true, if the given position is inside the target and thus occluded.
*/
bool isOccludedByTarget(
    vec3 observer               ///< position to check
);
