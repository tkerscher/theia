#include "math.glsl"
#include "polarization.glsl"
#include "scatter.volume.glsl"

layout(local_size_x = 32) in;

struct Query{
    vec4 stokes;
    vec3 dir;
    vec3 ref;
    float phi;
    float theta;    
};

struct Result {
    vec4 stokes;
    vec3 ref;
};

layout(scalar) readonly buffer QueryBuffer{ Query queries[]; };
layout(scalar) writeonly buffer ResultBuffer{ Result results[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    //create new direction
    Query q = queries[i];
    vec3 ref;
    vec3 dir = scatterDir(q.dir, cos(q.theta), q.phi);
    vec4 stokes = rotatePolRef(q.dir, q.ref, dir, ref) * q.stokes;
    results[i] = Result(stokes, ref);
}
