#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rest of the code interacts with
//volume models. Some functions are mandatory while some are only required for
//an extended feature set.
//
//If the volume model wants to support particle rays, it may not rely on the
//contrib attribute of the ray, even if it code belongs to uncalled functions
//such as the ones used by NEE.
//
//Note, that RAY stands for either ForwardRay or BackwardRay. A volume model may
//choose to implement one or the other, or both. In the latter case, two
//separate functions for each ray type is required.

/**
 * Samples the length the ray has to propagate for the next volume interaction
 * to occur.
*/
float sampleInteractionLength(
    const RAY ray,                  ///< Ray for which to sample
    uint idx, inout uint dim        ///< RNG state
);

/**
 * Samples an interaction of the given ray with the surrounding media after
 * propagation.
*/
ResultCode sampleVolumeInteraction(
    inout RAY ray,                  ///< Ray which interacts
    uint idx, inout uint dim        ///< RNG state
);

/**
 * Called by the tracer after an call to `sampleInteractionLength` and the ray
 * has been propagated to apply any effects of the volume on the ray.
 * It is safe to assume any importance sampling happened in
 * `sampleInteractionLength` have been applied.
*/
ResultCode applyVolumeSampled(
    inout RAY ray,                  ///< Ray to apply volume effects to
    float dist,                     ///< Distance [m] the ray has propagated
    bool hit,                       ///< Whether the ray has hit a surface
    uint idx, inout uint dim        ///< RNG state
);

/**
 * Called by the tracer after propagating the ray *without* a previous call to
 * `sampleInteractionLength` to any effects of the volume on the ray. Therefore
 * no importance sampling has happened.
 *
 * OPTIONAL. Required for backward tracing and NEE.
*/
ResultCode applyVolume(
    inout RAY ray,
    float dist,
    bool hit,
    uint idx, inout uint dim
);

/**
 * Samples a volume interaction that scatters the ray into the given direction.
 * Is expected to update the ray accordingly, e.g. by calling `scatterRay`.
 *
 * OPTIONAL. Required for backward tracing and NEE
*/
ResultCode volumeScatterRay(
    inout RAY ray,                  ///< Ray to scatter
    vec3 newDir,                    ///< New direction after scattering
    uint idx, inout uint dim        ///< RNG state
);

/**
 * Importance samples a scattering direction at the ray's current position.
 *
 * OPTIONAL. Required for NEE
*/
vec3 sampleVolumeScattering(
    inout RAY ray,                  ///< Ray to sample
    out float prob,                 ///< Probability of the returned sample
    uint idx, inout uint dim        ///< RNG state
);

/**
 * Returns the sampling probability of the given scattering direction. That is,
 * the probability the same direction would have been sampled by
 * `sampleVolumeScattering`.
 *
 * OPTIONAL. Required for NEE.
*/
float volumeScatterProb(
    RAY ray,                        ///< Ray to sample
    vec3 scatteredDir               ///< Scattered direction
);
