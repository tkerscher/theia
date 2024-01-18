#ifndef _INCLUDE_LIGHTSOURCE_MODULAR
#define _INCLUDE_LIGHTSOURCE_MODULAR

SourceRay sampleLight(uint idx) {
    //sample ray (offset rng behind samples to make api easier)
    RaySample ray = sampleRay(idx, RNG_RAY_SAMPLE_OFFSET);

    //sample photons (at rng offset 0, sampler advances itself)
    SourceSample samples[N_LAMBDA];
    [[unroll]] for (uint i = 0; i < N_LAMBDA; ++i) {
        samples[i] = sampleSource(ray, idx, i);
        samples[i].contrib *= ray.contrib;
    }

    //assemble source ray and return
    return SourceRay(
        ray.position,
        ray.direction,
        samples
    );
}

#endif
