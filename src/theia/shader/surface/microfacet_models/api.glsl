#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rough surface models interact
//with a micro-facet distribution. A model provides the functions below and is
//prepended to `surface/dielectric_rough/template.glsl` during compilation (see
//theia.surface). The Geant4 UNIFIED model (`surface/unified/`) uses the gaussian
//model the same way.
//
//The models are intentionally free in how they parametrize themselves: nothing
//outside assumes a particular material property, let alone a single scalar
//roughness. Everything a model needs it loads itself in prepare_microfacet().
//
//All directions are in world space. `rayDir` is the incident ray direction, i.e.
//it points *towards* the surface, opposite to `hit.rayNrm`.

/**
 * MANDATORY. Structure caching whatever is constant across a single surface
 * interaction, i.e. does not depend on the sampled facet. Built once per hit and
 * passed into every other function, so loop invariant work - loading material
 * properties, building a tangent basis - happens once rather than per attempt.
*/
struct MicrofacetParams { };

/**
 * MANDATORY. Fills MicrofacetParams. This is where a model loads its own
 * parameters from the material table; the amount and meaning of those is
 * entirely up to the model.
*/
MicrofacetParams prepare_microfacet(
    vec3 rayDir,                    ///< Incident ray direction
    const SurfaceHit hit            ///< Surface intersection being processed
);

/**
 * MANDATORY. Samples a micro-facet normal in the hemisphere around
 * `hit.rayNrm`. May itself use rejection internally.
 *
 * Note
 * ----
 * The number of random numbers drawn here has to be reported through
 * `SurfaceRNGDraws.prepareSurface` of the surface model using this file.
*/
vec3 sample_microfacet_normal(
    const MicrofacetParams params,
    vec3 rayDir,                    ///< Incident ray direction
    const SurfaceHit hit,           ///< Surface intersection being processed
    uint idx, inout uint dim        ///< RNG state
);

/**
 * MANDATORY. Probability that a facet is accepted for the given outgoing
 * direction, in [0,1]. The caller turns it into a decision by comparing against
 * a single uniform, so a model needing no randomness simply returns 0.0 or 1.0.
 *
 * This is where a model states its own validity criterion: that the facet is hit
 * from the front, that the outgoing direction is not masked by the micro
 * structure, or anything else. The macroscopic hemisphere is *not* checked here -
 * the surface template does that.
 *
 * Note
 * ----
 * The facet loop evaluates this for every attempt, so a model whose acceptance
 * is expensive makes the whole loop expensive, since it dominates the cost of a
 * surface hit.
*/
float microfacet_accept_prob(
    const MicrofacetParams params,
    vec3 dirOut,                    ///< Outgoing direction to test
    vec3 microfacetNormal,          ///< Facet the direction came from
    const SurfaceHit hit,           ///< Surface intersection being processed
    vec3 rayDir                     ///< Incident ray direction
);
