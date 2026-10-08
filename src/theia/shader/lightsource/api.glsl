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

//Instead of only performing next event estimates (NEE) at the scattering points
//sampled by volume models, we let the light source choose an alternate
//scattering point and combine these two samples using multiple importance
//sampling (MIS) to improve the performance. This can even be used to get rid of
//singularities as encountered by e.g. point light sources caused by including a
//factor 1/d^2.
//To announce support for this optional feature, set the following macro
#define LIGHT_SOURCE_SUPPORTS_MIS

/**
 * OPTIONAL. Structure caching calculations used by other functions.
*/
struct LightNeeContext{ };
//Need to set this macro to tell other code LightNeeContext exists.
#define LightNeeContext

/**
 * Samples an alternate stepping size to create an alternate earlier scattering
 * point used for NEE.
 *
 * OPTIONAL. Required for backward MIS
*/
float sampleLightNeeStepSize(
    vec3 origin,                ///< Ray position
    vec3 direction,             ///< Ray direction
    float maxStep,              ///< Upper limit on the sampled step size
    float wavelength,           ///< Wavelength at which to sample
    uint mediumIdx,             ///< Medium from which to sample the light source
    #ifdef LightNeeContext
    out LightNeeContext context,///< Optionally cache for later evaluation
    #endif
    uint idx, inout uint dim    ///< RNG state
);

/**
 * Samples the light source for a light ray pointing at the current position as
 * part of a MIS mixed NEE. Use the balance heuristic to calculate the ray's
 * contribution as follows:
 *
 *            L(x)
 *  ---------------------------
 *   p_light(t) + residualProb
 *
 * Note that residualProb may contain the sum of probabilites, which is why the
 * balance heuristic is used here.
*/
ForwardRay sampleLightMIS(
    vec3 rayOrigin,             ///< Ray position before propagation
    vec3 rayDirection,          ///< Ray direction
    float tStep,                ///< Sampled step size
    float wavelength,           ///< Wavelength to sample at
    uint mediumIdx,             ///< Medium from which was sampled
    float residualProb,         ///< Sum of probabilites from other strategies
    #ifdef LightNeeContext
    out LightNeeContext context,///< Optionally cache for later evaluation
    #endif
    uint idx, inout uint dim    ///< RNG state
);
