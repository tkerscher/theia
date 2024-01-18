#ifndef _INCLUDE_PHOTONSAMPLER_UNIFORM
#define _INCLUDE_PHOTONSAMPLER_UNIFORM

layout(scalar) uniform SourceParams {
    float lambda_min;
    float lambda_max;
    float t_min;
    float t_max;
    float contrib;
} sourceParams;

SourceSample sampleSource(const RaySample ray, uint idx, uint i) {
    float delta_lambda = sourceParams.lambda_max - sourceParams.lambda_min;
    float delta_time = sourceParams.t_max - sourceParams.t_min;
    
    float lambda = sourceParams.lambda_min + delta_lambda * random(idx, 2*i + 0);
    float t = sourceParams.t_min + delta_time * random(idx, 2*i + 1);

    return SourceSample(lambda, t, sourceParams.contrib);
}

#endif
