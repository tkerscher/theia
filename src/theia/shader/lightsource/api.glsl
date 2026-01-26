#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rest of the code interacts with
//light sources.
//
//Note that light sources return forward rays and thus depend on the used ray
//model. This dependency is not further specified by this package allowing for
//some leeway. However, it is a good idea to use some sorts of factory functions
//instead of the constructor of ForwardRay to allow the ray model to be changed.

/**
 * Samples a new light ray from the source.
 *
 * Required for forward tracing.
*/
ForwardRay sampleLight(
    uint idx, inout uint dim        ///< RNG state
);

/**
 * Samples the light source from a specific observation point. This may be
 * either from a surface or from within a volume. In the latter case the normal
 * will be the zero vector.
 * The light source must ensure, that the returned ray points towards the
 * observer, but it may do so from the "wrong" side of the surface as indicated
 * by the surface normal.
 *
 * Required for backward tracing and direct light sampling.
*/
ForwardRay sampleLight(
    vec3 observer,              ///< Position from which the source is sampled
    vec3 normal,                ///< Surface normal at observer
    float wavelength,           ///< Wavelength to sample at
    uint mediumIdx,             ///< Index of the medium surrounding the observer
    uint idx, inout uint dim    ///< RNG state
);
