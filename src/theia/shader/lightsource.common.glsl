#ifndef _INCLUDE_LIGHTSOURCE_SAMPLING
#define _INCLUDE_LIGHTSOURCE_SAMPLING

struct RaySample {
    vec3 position;
    vec3 direction;
    float contrib;
};

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
