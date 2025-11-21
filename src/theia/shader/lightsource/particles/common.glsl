#ifndef _INCLUDE_LIGHTSOURCE_PARTICLES_COMMON
#define _INCLUDE_LIGHTSOURCE_PARTICLES_COMMON

#include "math.glsl"

/*
Methods shown here are based on L. Raedel's master thesis and subsequent papers
[1-3]

The equation he uses for the angular emission profile is unfortunately not
suitable for sampling via inverting the CDF. As a substitute we instead use the
same equation as IceCube [4]:

 1     dN
--- -------- ~ exp(-b*x^a) * x^(a-1) = f(x) ; x = 1 - cos(theta)
 N   dOmega

Note that IceCube treats this as the angular distribution of secondary particle
directions. Its shape is however still very similar (difference are only really
noticeable in log plot) to the function Raedel uses and is thus used as a
substitute, i.e. we use x = |cos(theta) - cos(cherenkov)|

For completeness sake, here are also the integral and its inverse:
 
        / x           /  -1              \ x=x'     1
 u(x) = | f(x') dx' = | ---- exp(-bx'^a) |      = ---- (1 - exp(-bx^a))
        / 0           \  ab              / x=0     ab

 x = (-1/b log(1 - ab*u))^(1/a)

[1] L. Raedel "Simulation Studies of the Cherenkov Light Yield from Relativistic
    Particles in High-Energy Neutrino Telescopes with Geant 4" (2012)
[2] L. Raedel, C. Wiebusch: "Calculation of the Cherenkov light yield from low
    energetic secondary particles accompanying high-energy muons in ice and
    water with Geant4 simulations" (2012) arXiv:1206.5530v2
[3] L. Raedel, C. Wiebusch: "Calculation of the Cherenkov light yield from
    electromagnetic cascades in ice with Geant4" (2013) arXiv:1210.5140v2
[4] M. G. Aartsen et al.: "Measurement of South Pole ice transparency with the
    IceCube LED calibration system" (2013) arXiv:1301.5361v1
*/

/**
 * Frank-Tamm formula
 *   d^2 N          alpha  /       1   \
 * --------- = 2pi ------- | 1 - ----- |
 *  dx dlam         lam^2  \      n^2  /
 *
 * where alpha is the fine structure constant
 *
 * Note that we assume particle travels at light speed (beta = 1).
*/
float frank_tamm(float lambda, float n) {
    //for better numerical stability, convert lambda from nm -> µm (close to one)
    lambda *= 1e-3;
    float sin2_theta = 1.0 - 1.0 / (n*n);
    //Note that alpha has an additional factor 10^3 because of the mismatch in
    //lambda (µm) and dlam (nm) / dx (m)
    float result = TWO_PI * 7.2973525693 / (lambda*lambda) * sin2_theta;
    //result will be negative for n < 1.0 -> no radiation (zero contrib)
    return max(result, 0.0);
}

/**
 * Samples the angular emission profile for the cosine emission angle as
 * deviation from the original particle direction and NOT the Cherenkov angle.
 * The latter is taken account for.
*/
float particle_sampleEmissionAngle(
    float n,    ///< Refractive index of surrounding media
    float a,    ///< Parameterization of emission profile
    float b,    ///< ^^^^
    float u     ///< Random number used for sampling
) {
    //calculate cherenkov angle
    float cos_chev = 1.0 / n;

    //we shift the distribution so that cos(theta=0) => cos(theta_chev)
    //we do this by sampling over two separate ranges:
    // lower: [cos_chev, 1]
    // upper: [cos_chev, -1]
    //finally, by scaling the random number, we can use the sign to decide wether
    //we sampled the lower or upper range
    
    //integrate upper and lower bound
    //Note we are missing here a factor 1/(ab)
    float int_lower = 1.0 - exp(-b*pow(1.0 - cos_chev, a));
    float int_upper = 1.0 - exp(-b*pow(1.0 + cos_chev, a));

    //rescale u. negative values mean angles shallower than cos_chev
    //note the missing factor 1/(ab) cancels with a factor ab in the inverse cdf
    u = u * (int_upper + int_lower) - int_lower;
    //evaluate inverse integral
    float x = pow(-log(1-abs(u)) / b, 1 / a);
    //add cherenkov angle to get x param, subtract from one to get cos(theta)
    //(see comment at the top)
    return cos_chev - sign(u) * x;
}
/**
 * Specialization to sample the angular emission profile on a narrowed down
 * angular range. Provides an additional output parameter returning the weight
 * or contribution of the importance sample, i.e. f(x)/p(x)
*/
float particle_sampleEmissionAngle(
    float n,                        ///< Refractive index of surrounding media
    float a, float b,               ///< Parameterization of emission profile
    float u,                        ///< Random number used for sampling
    float cos_min, float cos_max,   ///< Ranges within to sample.
    out float contrib               ///< Contribution/weight from importance sampling
) {
    //calculate cherenkov angle
    float cos_chev = 1.0 / n;

    //integrate upper and lower bound
    //int(cos_min->cos_max)=int(cos_min->cos_chev) + int(cos_chev->cos_max)
    //                     =int(cos_chev->cos_max) - int(cos_chev->cos_min)
    //                     =int_upper              - int_lower
    //Note we are missing here a factor 1/(ab)
    float int_upper = 1.0 - exp(-b*pow(abs(cos_max - cos_chev), a));
    float int_lower = 1.0 - exp(-b*pow(abs(cos_min - cos_chev), a));
    //take sign of cos_i - cos_chev into account
    int_upper *= sign(cos_max - cos_chev);
    int_lower *= sign(cos_min - cos_chev);

    //calculate the importance sample contribution f(x)/p(x)
    //both are equal up to a normalization factor
    //they share a factor a*b that cancel out
    contrib = abs(int_upper - int_lower);
    contrib /= 2.0 - exp(-b*pow(1.0 - cos_chev, a)) - exp(-b*pow(1.0 + cos_chev, a));

    //rescale u. negative values mean angles shallower than cos_chev
    //note the missing factor 1/(ab) cancels with a factor ab in the inverse cdf
    u = u * (int_upper - int_lower) + int_lower;
    //evaluate inverse integral
    float x = -pow(-log(1-abs(u)) / b, 1 / a);
    //add cherenkov angle to get emission angle w.r.t. particle direction
    return cos_chev - sign(u) * x;
}

float particle_evalEmissionAngle(
    float n,
    float a, float b,
    float cos_theta
) {
    //calculate cherenkov angle
    float cos_chev = 1.0 / n;

    //calculate normalization constant by integration
    float norm = a * b;
    norm /= 2.0 - exp(-b*pow(1.0 - cos_chev, a)) - exp(-b*pow(1.0 + cos_chev, a));

    //evaluate pdf
    float x = abs(cos_theta - cos_chev);
    return exp(-b * pow(x, a)) * pow(x, a - 1.0) * norm;
}

#endif
