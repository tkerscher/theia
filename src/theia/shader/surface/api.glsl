#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rest of the code interacts with
//surface models. Some functions are manndatory, while some are optional.
//
//If the surface model wants to support particle rays, it may not rely on the
//contrib attribute of the ray, even if it belongs to uncalled functions as the
//otherwise the code will not compile.
//
//Note, that RAY stands for either ForwardRay or BackwardRay. A surface model
//may choose to implement one or the other, or both. In the latter case, two
//separate functions for each ray type is required.

/**
 * OPTIONAL. Structure caching calculations used by other functions.
*/
struct SurfaceProperties{ };
//Need to set this macro to tell other code SurfaceProperties exists.
#define SurfaceProperties SurfaceProperties

/**
 * OPTIONAL. Called before any other surface model function. Can be used to do
 * common calculations like reflectance only once.
*/
SurfaceProperties prepareSurface(
    const RAY ray,                  ///< Ray to process
    const SurfaceHit hit,           ///< Surface intersection to process
    uint idx, inout uint dim        ///< RNG state
);

/**
 * Samples and applies an interaction of the ray with the surface preparing it
 * for the next tracing step or aborting it as indicated by its return value.
*/
ResultCode sampleSurfaceInteraction(
    inout RAY ray,                  ///< Ray interacting with the surface
    const SurfaceHit hit,           ///< Description of surface hit
    #ifdef SurfaceProperties
    const SurfaceProperties props,  ///< Optional cached calculation
    #endif
    uint idx, inout uint dim        ///< RNG state
);

/**
 * Process the surface hit to create a hit item. Returns true if successfull.
 * If called, it will be called before `sampleSurfaceInteraction`. Needs to be
 * implemented even if surface does not generate hits. In that case always
 * return false.
*/
bool processSurfaceTargetHit(
    ForwardRay ray,                 ///< Ray interacting with the surface
    const SurfaceHit hit,           ///< Description of surface hit
    #ifdef SurfaceProperties
    const SurfaceProperties props,  ///< Optional cached calculations
    #endif
    int objectId,                   ///< Id of the object the surface belongs to
    out HitItem item,               ///< Produced hit item
    uint idx, inout uint dim        ///< RNG state
);
