#ifndef _INCLUDE_LOOKUP_CUBIC
#define _INCLUDE_LOOKUP_CUBIC

/*
 * Cubic interpolation based on the unique solution presented in:
 * R.G. Keys: "Cubic Convolution Interpolation for Digital Image Processing" (1981)
*/

vec4 _evalCubicKernel(float s) {
    vec4 t = 0.5 * vec4(1.0, s, s*s, s*s*s);
    const mat4 kernel = mat4(
        0.0, -1.0,  2.0, -1.0,
        2.0,  0.0, -5.0,  3.0,
        0.0,  1.0,  4.0, -3.0,
        0.0,  0.0, -1.0,  1.0
    );
    return t * kernel;
}

vec4 _loadLUTSamples(const Table2D table, ivec4 idx) {
    return vec4(
        table.samples[idx[0]],
        table.samples[idx[1]],
        table.samples[idx[2]],
        table.samples[idx[3]]
    );
}

float lookUp_cubic(const Table1D table, float u) {
    //sample index of c_-1 (here we will use -1 -> 0)
    //we use both floor and ceil to catch the case of u being at the right edge
    //in that case i_lo = i_hi and i_hi + 2 will not point out of bounds
    //whereas i_lo + 3 would
    int i_lo = int(floor(u)) + LUT_PADDING;
    int i_hi = int(ceil(u)) + LUT_PADDING;
    float l = fract(u);
    //load samples
    vec4 c = vec4(
        table.samples[i_lo - 1],
        table.samples[i_lo],
        table.samples[i_hi],
        table.samples[i_hi + 1]
    );
    //interpolate
    return dot(c, _evalCubicKernel(l));
}

float lookUp_cubic(const Table2D table, float u, float v) {
    //sample coordinates
    int u_lo = int(floor(u)) + LUT_PADDING;
    int u_hi = int(ceil(u)) + LUT_PADDING;
    vec4 ku = _evalCubicKernel(fract(u));
    int v_lo = int(floor(v)) + LUT_PADDING;
    int v_hi = int(ceil(v)) + LUT_PADDING;
    vec4 kv = _evalCubicKernel(fract(v));
    //We adopt the row-major memory layout from numpy:
    //axis 0 (u) maps to rows, axis 1 (v) to columns
    //-> stride between rows (mind the padding at each side)
    int stride = table.nv + 2 * LUT_PADDING;
    //sample indices
    ivec4 iu = ivec4(u_lo - 1, u_lo, u_hi, u_hi + 1);
    ivec4 iv = ivec4(v_lo - 1, v_lo, v_hi, v_hi + 1);
    iu *= stride; //apply strides to rows

    //in order to not skip too much in memory, first interpolate single rows...
    vec4 cu = vec4(
        dot(_loadLUTSamples(table, iv + iu[0]), kv),
        dot(_loadLUTSamples(table, iv + iu[1]), kv),
        dot(_loadLUTSamples(table, iv + iu[2]), kv),
        dot(_loadLUTSamples(table, iv + iu[3]), kv)
    );
    //...and then the columns
    return dot(cu, ku);
}

#endif
