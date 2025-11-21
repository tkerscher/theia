#ifndef _INCLUDE_LIGHTSOURCE_GUIDED
#define _INCLUDE_LIGHTSOURCE_GUIDED

#ifdef LIGHTSOURCE_GUIDED_CHECK_VIS
#include "scene.intersect.glsl"
#endif

//on the Python side we will prepend the source code of the principal light
//source and rename its public API by adding the prefix "principal_"

#ifdef LIGHTSOURCE_GUIDED_USE_CAM
float sampleLightTarget(
    float wavelength,
    out vec3 samplePos, out vec3 sampleNrm,
    uint idx, inout uint dim
) {
    CameraSample cam = sampleCamera(wavelength, idx, dim);
    samplePos = cam.position;
    sampleNrm = cam.normal;
    return cam.contrib;
}
#endif

SourceRay sampleLight(
    float wavelength,
    const MediumConstants medium,
    uint idx, inout uint dim
) {
    //sample guide
    vec3 samplePos, sampleNrm;
    float contrib = sampleLightTarget(wavelength, samplePos, sampleNrm, idx, dim);

    //sample actual light source
    SourceRay sourceRay = principal_sampleLight(
        samplePos, sampleNrm, wavelength, medium, idx, dim);
    //apply contrib
    sourceRay.contrib *= contrib;

    //optionally, check for occlusion between sampled position and light source
    #ifdef LIGHTSOURCE_GUIDED_CHECK_VIS
    if (!isVisible(sourceRay.position, samplePos))
        return badSourceRay;
    #endif

    return sourceRay;
}

#endif
