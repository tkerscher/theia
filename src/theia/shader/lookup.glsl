#ifndef _LOOKUP_INCLUDE
#define _LOOKUP_INCLUDE

layout(buffer_reference, scalar) readonly buffer Table1D {
    float nx; //Number of samples - 1; float to save casting
    float samples[];
};

layout(buffer_reference, scalar) readonly buffer Table2D {
    float nu, nv; //Number of samples per dim - 1; float to save casting
    float samples[];
};

float lookUp(const Table1D table, float u, float nullValue) {
    if (uint64_t(table) == 0)
        return nullValue;

    u = clamp(u, 0.0, 1.0);

    u *= table.nx;

    int lo = int(floor(u));
    int hi = int(ceil(u));
    float l = fract(u);

    return mix(table.samples[lo], table.samples[hi], l);
}
float lookUp(const Table1D table, float u) {
    return lookUp(table, u, 0.0);
}

// looks up value and derivative
vec2 lookUpDx(const Table1D table, float u, vec2 nullValue) {
    if (uint64_t(table) == 0)
        return nullValue;

    u = clamp(u, 0.0, 1.0);
    u *= table.nx;

    //we have to ensure lo and hi are distinct
    int iMax = int(table.nx);
    int lo = max(int(floor(u)), 0);
    int hi = min(lo + 1, iMax);
    float l = fract(u);

    int lolo = max(lo - 1, 0);
    int hihi = min(hi + 1, iMax);

    // look one further in each direction for numerical derivative
    float vLoLo = table.samples[lolo];
    float vLo = table.samples[lo];
    float vHi = table.samples[hi];
    float vHiHi = table.samples[hihi];
    // edge case: lo/hi was clamped at array border
    // -> divide by 1 not 2 (equals to forward/backward difference)
    // edge case: only one sample point -> difference always 0
    //  -> to prevent div by zero, clamp to 1
    float dxLo = (vHi - vLoLo) / float(max(hi - lolo, 1));
    float dxHi = (vHiHi - vLo) / float(max(hihi - lo, 1));

    float value = mix(vLo, vHi, l);
    float dx = mix(dxLo, dxHi, l) * table.nx;

    return vec2(value, dx);
}
vec2 lookUpDx(const Table1D table, float u) {
    return lookUpDx(table, u, vec2(0.0));
}

float lookUp2D(const Table2D table, float u, float v, float nullValue) {
    if (uint64_t(table) == 0)
        return nullValue;

    u = clamp(u, 0.0, 1.0);
    v = clamp(v, 0.0, 1.0);

    int stride = int(table.nu) + 1;
    u *= table.nu;
    v *= table.nv;

    int u_lo = int(floor(u));
    int u_hi = int(ceil(u));
    float ul = fract(u);

    int v_lo = int(floor(v));
    int v_hi = int(ceil(v));
    float vl = fract(v);

    int Q11 = v_lo * stride + u_lo;
    int Q12 = v_lo * stride + u_hi;
    int Q21 = v_hi * stride + u_lo;
    int Q22 = v_hi * stride + u_hi;

    float lo = mix(table.samples[Q11], table.samples[Q12], ul);
    float hi = mix(table.samples[Q21], table.samples[Q22], ul);

    return mix(lo, hi, vl);
}
float lookUp2D(const Table2D table, float u, float v) {
    return lookUp2D(table, u, v, 0.0);
}

#endif
