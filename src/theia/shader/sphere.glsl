#ifndef _SPHERE_INCLUDE
#define _SPHERE_INCLUDE

#include "cosy.glsl"
#include "math.glsl"

struct Sphere{
    vec3 position;
    float radius;
};

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
    float d2 = dot(delta, delta);
    float d = sqrt(d2);                                 //dist to sphere center
    float t = sqrt(d2 + sphere.radius*sphere.radius);   //dist to sphere edge
    float cos_max = d / t;

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

    //calculate sample probability
    p = 1.0 / (TWO_PI * (1.0 - cos_max));

    //create trafo for local -> observer
    mat3 trafo = createLocalCOSY(delta / d);
    //transform p to observer space, than world space
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
    vec3 delta = sphere.position - observer;
    float d2 = dot(delta, delta);
    float d = sqrt(d2);                                 //dist to sphere center
    float t = sqrt(d2 + sphere.radius*sphere.radius);   //dist to sphere edge
    float cos_max = d / t;

    //return constant probability
    return 1.0 / (TWO_PI * (1.0 - cos_max));
}

#endif
