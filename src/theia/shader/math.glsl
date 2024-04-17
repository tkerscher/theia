#ifndef _MATH_INCLUDE
#define _MATH_INCLUDE

#define HALF_PI 1.570796326794896619
#define PI 3.141592653589793238
#define TWO_PI 6.283185307179586477
#define FOUR_PI 12.56637061435917295

#define INV_PI 0.318309886183790671
#define INV_2PI 1.57079632679489661
#define INV_4PI 0.0795774715459476679

#define INV_SQRT2 0.70710678118654752440

//returns the sign bit as float, i.e. like sign() but maps 0.0 to 1.0 or -1.0
//depending on the sign bit
float signBit(float f) {
    return uintBitsToFloat((floatBitsToInt(f) & 0x80000000) | 0x3F800000);
}

//more accurate version to calculate ab-cd trying to mitigate the danger of
//catastrophic cancelation. See PBRTv4: B.2.9
float prodDiff(float a, float b, float c, float d) {
    precise float cd = c * d;
    precise float diff = fma(a, b, -cd);
    precise float err = fma(-c, d, cd);
    return diff + err;
}

//more accurate version of cross product using prodDiff().
//See PBRTv4: 3.3.2
vec3 crosser(vec3 v, vec3 w) {
    return vec3(
        prodDiff(v.y, w.z, v.z, w.y),
        prodDiff(v.z, w.x, v.x, w.z),
        prodDiff(v.x, w.y, v.y, w.x)
    );
}

//small helper function to create a local coordinate system from a vector
//as the new z axis, and two (not specified) perpendicular ones
//returns the transformation matrix to transfrom local in global system
//See PBRTv4: chapter 3.3.3
mat3 createLocalCOSY(vec3 vz) {
    float s = signBit(vz.z);
    float a = -1.0 / (s + vz.z);
    float b = vz.x * vz.y * a;
    vec3 vx = vec3(1.0 + s * vz.x * vz.x * a, s * b, -s * vz.x);
    vec3 vy = vec3(b, s + vz.y * vz.y * a, -vz.y);
    //compile matrix
    vx = normalize(vx); //should not needed but I wont hunt that bug down...
    vy = normalize(vy);
    return mat3(vx,vy,vz);
}

#endif
