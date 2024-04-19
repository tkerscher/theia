#ifndef _INCLUDE_LIGHTSOURCE_PENCIL
#define _INCLUDE_LIGHTSOURCE_PENCIL

layout(scalar) uniform LightParams {
    vec3 position;
    vec3 direction;
    //always keep polarization params
    //makes it easier on the python side
    vec4 stokes;
    vec3 polRef;
} lightParams;

SourceRay sampleLight(uint idx) {
    //sample photon
    SourceSample photon = sampleSource(idx, 0);
    //assemble source and return
    return createSourceRay(
        lightParams.position,
        lightParams.direction,
        lightParams.stokes,
        lightParams.polRef,
        photon
    );
}

#endif
