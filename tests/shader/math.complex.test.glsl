layout(local_size_x = 32) in;

#include "complex.glsl"

struct Query {
    complex a;
    complex b;
};

struct Result {
    float cabs;
    float carg;
    complex cmul;
    complex cdiv;
    complex csqrt;
    complex cpyth;
    complex cexp;
    complex clog;
    complex cpowf;
    complex cpow;
    complex csin;
    complex ccos;
    complex ctan;
};

readonly buffer QueryBuffer { Query q[]; };
writeonly buffer ResultBuffer { Result r[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    complex a = q[i].a;
    complex b = q[i].b;
    r[i] = Result(
        cabs(a),
        carg(a),
        cmul(a, b),
        cdiv(a, b),
        csqrt(a),
        cpyth(a),
        cexp(a),
        clog(a),
        cpow(a, creal(b)),
        cpow(a, b),
        csin(a),
        ccos(a),
        ctan(a)
    );
}
