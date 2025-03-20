#include "math.glsl"
#include "polarization.glsl"
#include "util.sample.glsl"

#include "random.glsl"

layout(local_size_x = 32) in;

layout(scalar) writeonly buffer ResultBuffer { mat4 result[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint dim = 0;

    //We exploit that the we have a separate test for rotatePolRef() and just
    //test whether we can use alignPolRef to undo the rotation

    //sample two random unit vectors
    vec3 dirIn = sampleUnitSphere(random2D(i, dim));
    vec3 dirOut = sampleUnitSphere(random2D(i, dim));
    //get a unit vector normal to dirIn
    vec3 polRef = createLocalCOSY(dirIn)[0];

    //rotate polRef using rotatePolRef()
    vec3 newPolRef;
    mat4 rot = rotatePolRef(dirIn, polRef, dirOut, newPolRef);

    //unrotate using alignPolRef
    mat4 unrot = alignPolRef(dirIn, newPolRef, polRef);
    result[i] = unrot * rot;
}
