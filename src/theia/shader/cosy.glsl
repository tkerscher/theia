#ifndef _COSY_INCLUDE
#define _COSY_INCLUDE

//small helper function to create a local coordinate system from a vector
//as the new z axis, and two (not specified) perpendicular ones
//returns the transformation matrix to transfrom local in global system
//See PBRTv4: chapter 3.3.3
mat3 createLocalCOSY(vec3 vz) {
    //sign returns 0.0 for exactly 0.0, but we want 1.0 in that case
    //  -> use twice with offset
    float s = sign(sign(vz.z) + 0.5); //bit ugly, but nicer than extracting sign bit
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
