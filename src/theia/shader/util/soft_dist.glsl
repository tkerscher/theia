#ifndef _INCLUDE_SOFT_DIST
#define _INCLUDE_SOFT_DIST

/*
* The smoothing kernels shown here follow the same idea outlined in
* W. Dehnen: "Towards optimal softening in 3D N-body codes: I. minimizing the
* force error", Monthly Notices of the Royal Astronomical Society 324, p. 273
* (2001)
*
* To avoid infinities caused by 1/r^2 as we approach r -> 0, we replace it with
* a compact kernel inside an epsilon-ball, i.e. for r >= eps we evaluate 1/r^2
* otherwise the kernel. We use a polynomial in u = (r/eps)^2 as ansatz:
*
*  r^-2 ~= k(u)/eps^2 if r < eps
*
* Imposing C1 smoothness already fixes a linear (in u) polynomial:
*
*  C1(u) = 2 - u
*
* This kernel always underestimates the true value. Used to estimate radiance
* from a light source will dim it causing bias.
* We can compensate for that and get a quadratic compensating kernel:
*
*  K1(u) = 9 - 15u + 7u^2
*
* The compensating kernel will overshoot 1/r^2 as it approaches r -> eps to
* compensate the underestimation at r -> 0. When creating time distributions
* this causes energy to be redistributed to later times. However, as long as
* binning is course enough this should not be observable.
* (typically > 1ns but depends on scene)
*/

float softKernel(float x2, float eps2) {
    float u = x2 / eps2;
    return 2.0 - u;
}

float softKernel_comp(float x2, float eps2) {
    float u = x2 / eps2;
    return 9.0 + u * (-15.0 + 7.0 * u);
}

float softInvSquare(float x2, float eps2) {
    if (x2 >= eps2)
        return 1.0 / x2;
    else
        return softKernel(x2, eps2) / eps2;
}

float softInvSquare_comp(float x2, float eps2) {
    if (x2 >= eps2)
        return 1.0 / x2;
    else
        return softKernel_comp(x2, eps2) / eps2;
}

//softDist gives the corresponding r, that would produce the same 1/r^2 as
//softInvSquare

float softDist(float x, float eps) {
    if (x >= eps)
        return x;
    else
        return eps * inversesqrt(softKernel(x * x, eps * eps));
}

float softDist_comp(float x, float eps) {
    if (x >= eps)
        return x;
    else
        return eps * inversesqrt(softKernel_comp(x * x, eps * eps));
}

#endif
