#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "math.glsl"
#include "polarization.glsl"

#include "random.glsl"

layout(local_size_x = 32) in;

layout(scalar) writeonly buffer ResultBuffer { mat4 result[]; };

vec3 sampleDir(uint idx, uint dim) {
    vec2 u = random2D(idx, dim);
    float cos_theta = 2.0 * u.x - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta*cos_theta, 0.0));
    float phi = TWO_PI * u.y;

    return vec3(
        sin_theta * sin(phi),
        sin_theta * cos(phi),
        cos_theta
    );
}

void main() {
    uint i = gl_GlobalInvocationID.x;

    //We exploit that the we have a separate test for rotatePolRef() and just
    //test whether we can use alignPolRef to undo the rotation

    //sample two random unit vectors
    vec3 dirIn = sampleDir(i, 0);
    vec3 dirOut = sampleDir(i, 2);
    //get a unit vector normal to dirIn
    vec3 polRef = createLocalCOSY(dirIn)[0];

    //rotate polRef using rotatePolRef()
    vec3 newPolRef;
    mat4 rot = rotatePolRef(dirIn, polRef, dirOut, newPolRef);

    //unrotate using alignPolRef
    mat4 unrot = alignPolRef(dirIn, newPolRef, polRef);
    result[i] = unrot * rot;
}
