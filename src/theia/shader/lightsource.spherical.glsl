#ifndef _INCLUDE_LIGHTSOURCE_SPHERICAL
#define _INCLUDE_LIGHTSOURCE_SPHERICAL

#include "rng.glsl"
#include "math.glsl"

//light params
layout(scalar) uniform LightParams {
    vec3 position;
    float lambda_min;
    float lambda_max;
    float t_min;
    float t_max;
    //1/r^2 needed for intensity->radiance cancels with sampling => const
    float contribution; // intensity / prob
} lightParams;

SourceRay sampleLight(uint idx) {
    float delta_lambda = lightParams.lambda_max - lightParams.lambda_min;
    float delta_time = lightParams.t_max - lightParams.t_min;
    //float prob = INV_4PI / dLam / time_duration;
    SourceSample samples[N_LAMBDA];
    for (int i = 0; i < N_LAMBDA; ++i) {
        float lambda = lightParams.lambda_min + delta_lambda * random(idx, 2*i + 0);
        float t = lightParams.t_min + delta_time * random(idx, 2*i + 1);

        samples[i] = SourceSample(
            lambda,
            t,            
            lightParams.contribution
        );
    }

    //sample direction
    vec2 u = random2D(idx, 2 * N_LAMBDA);
    float cos_theta = 2.0 * u.x - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * u.y;
    vec3 rayDir = vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );

    //create ray
    return SourceRay(
        lightParams.position,
        rayDir,
        samples
    );
}

#endif
