#ifndef _INCLUDE_LOOKUP_STEFFEN
#define _INCLUDE_LOOKUP_STEFFEN

/**
 * Local monotone cubic hermite interpolation based on the paper:
 * M. Steffen: A simple method for monotonic interpolation in one dimension (1990)
 *
 * Note that this algorithm does not generalize to higher dimensions.
*/

float lookUp_steffen(const Table1D table, float u) {
    //calculate sample indices
    //we use both floor and ceil to catch the case of u being exactly at the right edge
    //in that case we would otherwise load the last float beyond the table
    int i_lo = int(floor(u)) + LUT_PADDING;
    int i_hi = int(ceil(u)) + LUT_PADDING;
    float l = fract(u);
    //load samples
    float y0 = table.samples[i_lo - 1];
    float y1 = table.samples[i_lo];
    float y2 = table.samples[i_hi];
    float y3 = table.samples[i_hi + 1];
    //calculate slopes
    float s0 = y1 - y0;
    float s1 = y2 - y1;
    float s2 = y3 - y2;
    // float p1 = 0.5 * (y2 - y0);
    // float p2 = 0.5 * (y3 - y1);
    //estimate derivatives
    float dy1 = (sign(s0) + sign(s1)) * min(min(abs(s0), abs(s1)), 0.25 * abs(y2 - y0));
    float dy2 = (sign(s1) + sign(s2)) * min(min(abs(s1), abs(s2)), 0.25 * abs(y3 - y1));

    //eval polynom
    float a = dy1 + dy2 - 2.0 * s1;
    float b = 3.0 * s1 - 2.0 * dy1 - dy2;
    // float c = dy1
    // float d = y1
    // y = a*u^3 + b*u^2 + c*u + d
    //   = ((a*u + b)*u + c)*u + d
    return ((a*l + b)*l + dy1)*l + y1;
}

#endif
