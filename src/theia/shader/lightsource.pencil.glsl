#ifndef _INCLUDE_LIGHTSOURCE_PENCIL
#define _INCLUDE_LIGHTSOURCE_PENCIL

layout(scalar) uniform LightParams {
    vec3 position;
    vec3 direction;
    float budget;

    float t_min;
    float t_max;

    //always keep polarization params
    //makes it easier on the python side
    vec4 stokes;
    vec3 polRef;
} lightParams;

SourceRay sampleLight(uint idx, uint dim) {
    //sample startTime
    float u = random(idx, dim); dim++;
    float startTime = mix(lightParams.t_min, lightParams.t_max, u);
    //sample photon
    WavelengthSample photon = sampleWavelength(idx, dim);
    //apply budget
    photon.contrib *= lightParams.budget;
    //assemble source and return
    return createSourceRay(
        lightParams.position,
        lightParams.direction,
        startTime,
        lightParams.stokes,
        lightParams.polRef,
        photon
    );
}

#endif
