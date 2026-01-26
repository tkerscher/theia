#ifndef _INCLUDE_LIGHTSOURCE_PENCIL
#define _INCLUDE_LIGHTSOURCE_PENCIL

uniform LightParams {
    vec3 position;
    vec3 direction;
    float budget;

    uint mediumIdx;

    float t_min;
    float t_max;
} lightParams;

ForwardRay sampleLight(uint idx, inout uint dim) {
    //sample wavelength using wavelength source
    #ifdef LIGHT_SOURCE_EMIT_PARTICLE
    float wavelength = sampleWavelength(idx, dim);
    #else
    float contrib;
    float wavelength = sampleWavelength(contrib, idx, dim);
    #endif

    //sample startTime
    float u = random(idx, dim);
    float startTime = mix(lightParams.t_min, lightParams.t_max, u);

    //assemble forward ray
    return createForwardRay(
        lightParams.position,
        lightParams.direction,
        wavelength,
        lightParams.mediumIdx,
        startTime
        #ifndef LIGHT_SOURCE_EMIT_PARTICLE
        , contrib * lightParams.budget
        #endif
    );
}

#endif
