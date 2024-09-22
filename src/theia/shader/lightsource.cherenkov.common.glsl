#ifndef _INCLUDE_LIGHTSOURCE_CHERENKOV_COMMON
#define _INCLUDE_LIGHTSOURCE_CHERENKOV_COMMON

//We either want to track photon count or radiance
//TODO: make charge a variable?
#ifndef FRANK_TAMM_USE_PHOTON_COUNT
/***** Frank-Tamm formula
 *     d^3 E                       1   /            1        \
 * ------------ = 0.5*c^2*e*mu_0 ----- | 1 - --------------- |
 * dx dlam dphi                  lam^3 \     beta^2*n(lam)^2 /
*/
//Difference from pi*c^2*^2*mu_0 in literature stems from energy in eV and dphi (radial)
//Constant also has a factor 10^9 from lambda in µm and dlam in nm^-1

#define FRANK_TAMM_CONST 9.04756408986352 // eV / m*nm

float frank_tamm(float n, float lambda) {
    //for better numerical stability, convert lambda from nm -> µm (closer to one)
    lambda *= 1e-3;
    float result = FRANK_TAMM_CONST / (lambda*lambda*lambda) * (1.0 - (1.0 / (n*n)));
    //result will be negative for n < 1.0 -> no radiation (zero contrib)
    return max(result, 0.0);
}

#else //sample photon count
/***** Frank-Tamm formula
 *  d^2 N                1   /            1        \     N
 * ------- = alpha ----- | 1 - --------------- |  [ --- ]
 * dx dlam             lam^2 \     beta^2*n(lam)^2 /    m*nm
 *
 * alpha: fine structure constant
*/
//Difference from 2pi*alpha in literature stems from dphi (radial)
//Constant also has a factor 10^3 from lambda in µm and dlam in nm^-1

#define FRANK_TAMM_CONST 7.2973525693

float frank_tamm(float n, float lambda) {
    //for better numerical stability, convert lambda from nm -> µm (closer to one)
    lambda *= 1e-3;
    float result = FRANK_TAMM_CONST / (lambda*lambda) * (1.0 - (1.0 / (n*n)));
    //result will be negative for n < 1.0 -> no radition (zero contrib)
    return max(result, 0.0);
}

#endif
#undef FRANK_TAMM_CONST

#endif
