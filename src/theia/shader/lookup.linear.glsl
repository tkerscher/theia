#ifndef _INCLUDE_LOOKUP_LINEAR
#define _INCLUDE_LOOKUP_LINEAR

//Note that LUT are padded with boundary conditions for cubic interpolation
//index 0 is therefore padding and not actual data

float lookUp_linear(const Table1D table, float u) {
    //sample indices (starts after padding!)
    int lo = int(floor(u)) + LUT_PADDING;
    int hi = int(ceil(u)) + LUT_PADDING;
    float l = fract(u);

    //mix() does not play nice with inf, so we do it manually
    return table.samples[lo] * (1.0 - l) + table.samples[hi] * l;
}

float lookUp_linear(const Table2D table, float u, float v) {
    //sample coordinates (starts after padding per dimension!)
    int u_lo = int(floor(u)) + LUT_PADDING;
    int u_hi = int(ceil(u)) + LUT_PADDING;
    float lu = fract(u);
    int v_lo = int(floor(v)) + LUT_PADDING;
    int v_hi = int(ceil(v)) + LUT_PADDING;
    float lv = fract(v);
    //We adopt the row-major memory layout from numpy:
    //-> stride between rows (mind the padding at each side)
    int stride = table.nv + 2 * LUT_PADDING;
    //sample indices
    int i11 = stride * u_lo + v_lo;
    int i12 = stride * u_hi + v_lo;
    int i21 = stride * u_lo + v_hi;
    int i22 = stride * u_hi + v_hi;

    //again, we cannot use mix as it does not like inf...
    float lo = table.samples[i11] * (1.0 - lu) + table.samples[i12] * lu;
    float hi = table.samples[i21] * (1.0 - lu) + table.samples[i22] * lu;
    return lo * (1.0 - lv) + hi * lv;
}

#endif
