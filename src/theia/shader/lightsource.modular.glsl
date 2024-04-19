#ifndef _INCLUDE_LIGHTSOURCE_MODULAR
#define _INCLUDE_LIGHTSOURCE_MODULAR

SourceRay sampleLight(uint idx) {
    //sample ray (offset rng behind samples to make api easier)
    RaySample ray = sampleRay(idx, RNG_RAY_SAMPLE_OFFSET);
    //sample photons (at rng offset 0, sampler advances itself)
    SourceSample photon = sampleSource(idx, 0);

    //assemble source ray and return
    return createSourceRay(
        ray.position,
        ray.direction,
        photon.wavelength,
        photon.startTime,
        photon.contrib * ray.contrib
    );
}

#endif
