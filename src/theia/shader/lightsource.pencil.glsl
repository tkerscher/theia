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

SourceRay sampleLight(float wavelength, uint idx, inout uint dim) {
    //sample startTime
    float u = random(idx, dim);
    float startTime = mix(lightParams.t_min, lightParams.t_max, u);
    //assemble source and return
    return createSourceRay(
        lightParams.position,
        lightParams.direction,
        lightParams.stokes,
        lightParams.polRef,
        startTime,
        lightParams.budget
    );
}

#endif
