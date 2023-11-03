#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL
#define _INCLUDE_LIGHTSOURCE_SPHERICAL

#include "math.glsl"

//light params
layout(scalar) uniform LightParams {
    vec3 position;
    float lambda_min;
    float lambdaRange;
    float t0;
    float timeRange;
    //1/r^2 needed for intensity->radiance cancels with sampling => const
    float contribution; // intensity / prob
} lightParams;

SourceRay sampleLight() {
    //float prob = INV_4PI / dLam / time_duration;
    SourcePhoton photons[N_PHOTONS];
    for (int i = 0; i < N_PHOTONS; ++i) {
        float lambda = lightParams.lambda_min + lightParams.lambdaRange * rand();
        float t = lightParams.t0 + lightParams.timeRange * rand();

        photons[i] = SourcePhoton(
            lambda,
            t,            
            lightParams.contribution,   // lin_contrib
            0.0                         // log_contrib
        );
    }

    //sample direction
    float cos_theta = 2.0 * rand() - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * rand();
    vec3 rayDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );

    //create ray
    return SourceRay(
        lightParams.position,
        rayDir,
        photons
    );
}

#endif
