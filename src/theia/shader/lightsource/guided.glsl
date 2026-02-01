#ifndef _INCLUDE_LIGHTSOURCE_GUIDED
#define _INCLUDE_LIGHTSOURCE_GUIDED

#ifdef LIGHTSOURCE_GUIDED_CHECK_VIS
#include "scene/intersect.glsl"
#endif

//on the Python side we will prepend the source code of the principal light
//source and rename its public API by adding the prefix "principal_"

#ifdef LIGHTSOURCE_GUIDED_USE_CAM
LightTargetSample sampleLightTarget(
    float wavelength,
    uint idx, inout uint dim
) {
    CameraSample cam = sampleCamera(wavelength, idx, dim);
    return LightTargetSample(
        cam.position,
        cam.normal,
        cam.mediumIdx,
        cam.contrib
    );
}
#endif

ForwardRay sampleLight(uint idx, inout uint dim) {
    //sample wavelength using corresponding source
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    //sample guide
    LightTargetSample targetSample = sampleLightTarget(wavelength, idx, dim);
    contrib *= targetSample.contrib;    
    //sample actual light source
    ForwardRay ray = principal_sampleLight(
        targetSample.position,
        targetSample.normal,
        wavelength,
        targetSample.mediumIdx,
        idx, dim
    );
    //apply contrib
    ray.lin_contrib *= contrib;

    return ray;
}

#endif
