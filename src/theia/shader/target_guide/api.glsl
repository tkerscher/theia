#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rest of the code interacts with
//target guides.

/**
 * Samples the target guide for the given observer position and returns a sample
 * specifying both direction and distance a target hit is likely.
*/
TargetGuideSample sampleTargetGuide(
    vec3 observer,              ///< position from which to sample the target guide
    uint idx, inout uint dim    ///< RNG state
);

/**
 * Evaluates the target guide for the specified direction from the given 
 * observer's point. Returns whether a hit is to be expected and how likely
 * such a sample would have been drawn by sampleTargetGuide.
*/
TargetGuideSample evalTargetGuide(
    vec3 observer,              ///< position from which to evaluate
    vec3 direction              ///< direction to evaluate
);
