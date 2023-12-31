#ifndef _INCLUDE_LIGHTSOURCE_SAMPLING
#define _INCLUDE_LIGHTSOURCE_SAMPLING

struct SourceSample {
    float wavelength;
    float startTime;
    float contrib;
};
struct SourceRay {
    vec3 position;
    vec3 direction;
    SourceSample samples[N_LAMBDA];
};

#endif