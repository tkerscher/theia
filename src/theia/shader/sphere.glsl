#ifndef _SPHERE_INCLUDE
#define _SPHERE_INCLUDE

#include "math.glsl"

struct Sphere{
    vec3 position;
    float radius;
};

//For very small sampling cones we run out of numerical precision. To circumvent
//this we have two options: Collapse to a Dirac delta and always return the
//direction to the center (feels bad) or enforce a lower limit on the opening
//angle of the viewing cone (sampled direction might miss). We choose the latter
#define SPHERE_SAMPLE_MAX_COS_MAX 1.0-5e-6

//we only sample the visible half of the sphere visible from the observer
//furthermore, we want to be uniform in direction (not area!)

vec3 sampleSphere(
    const Sphere sphere,
    const vec3 observer,
    const vec2 rng,
    out float p
) {
    //calculate visible cone
    vec3 delta = sphere.position - observer;
    float d = distance(sphere.position, observer);
    float sin_max = sphere.radius / d;
    float cos_max = sqrt(max(1.0 - sin_max*sin_max, 0.0));
    //Limiting the viewing angle prevents from creatint inf prob (div by zero)
    cos_max = min(cos_max, SPHERE_SAMPLE_MAX_COS_MAX);

    //sample cone
    float phi = TWO_PI * rng.x;
    float cos_theta = (1.0 - rng.y) + rng.y * cos_max;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    //calc sample point in local space (need only to guarentee it hit the disk)
    vec3 pos = vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );

    //Calculate probability
    p = 1.0 / (TWO_PI * (1.0 - cos_max));

    //create trafo for local -> observer
    mat3 trafo = createLocalCOSY(delta / d);
    //transform p from object space to world space
    return normalize(trafo * pos);
}

float sampleSphereProb(
    const Sphere sphere,
    const vec3 observer,
    vec3 direction
) {
    //check if direction will hit the sphere
    direction = normalize(direction);
    vec3 f = observer - sphere.position;
    float b = dot(f, direction);
    float c = dot(f, f) - (sphere.radius*sphere.radius);
    //check discrimant
    if (b*b - c < 0.0) {
        return 0.0; //miss
    }

    //calculate visible cone
    float d = distance(sphere.position, observer);
    float sin_max = sphere.radius / d;
    float cos_max = sqrt(max(1.0 - sin_max*sin_max, 0.0));
    //Limiting the viewing angle prevents from creatint inf prob (div by zero)
    cos_max = min(cos_max, SPHERE_SAMPLE_MAX_COS_MAX);

    //Calculate probability
    return 1.0 / (TWO_PI * (1.0 - cos_max));
}

#endif
