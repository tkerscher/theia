#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rest of the code interacts with
//cameras
//
//Note that cameras return BackwardRay and CameraHit and thus depends on the
//used ray model. This dependency is not further specified by this package
//allowing for some leeway. However, it is a good idea to use some sorts of
//factory functions instead of the corresponding constructors to allow the ray
//model to be changed.

/**
 * Samples a new backward ray from the camera.
 *
 * Required for backward tracing.
*/
BackwardRay sampleCameraRay(
    float wavelength,           ///< Wavelength of the ray
    out CameraHit,              ///< Corresponding hit
    uint idx, inout uint dim    ///< RNG state
);

/**
 * Structure describing a camera sample used as a target for forward rays.
 * The camera must define this structure. Besides the required attributes listed
 * below, the camera is free to add additional ones that can serve as a cache
 * for createCameraRay().
*/
struct CameraSample {
    vec3 position;              ///< Position on camera in world space
    vec3 normal;                ///< Surface normal at position
    float contrib;              ///< Sample contribution, i.e. reciprocal sample probability
};

/**
 * Samples the camera for a potential target. A later call to createCameraRay()
 * transforms the returned sample into a full BackwardRay.
 *
 * Required for direct sampling.
*/
CameraSample sampleCamera(
    float wavelength,           ///< Wavelength to sample at
    uint idx, inout uint dim    ///< RNG state
);

/**
 * Takes a previously sampled camera sample and the direction of an incident
 * light ray to create a BackwardRay. The returned backward ray's direction must
 * be the opposite of the incident light direction.
 *
 * Required for direct sampling.
*/
BackwardRay createCameraRay(
    CameraSample camSample,     ///< Previously sampled camera sample
    vec3 lightDirection,        ///< Incident light direction
    float wavelength,           ///< Wavelength of incident light
    out CameraHit hit           ///< Corresponding hit
);
