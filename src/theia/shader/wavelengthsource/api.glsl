#error "This file is meant as documentation and does not contain valid code!"

//This file documents the API through which the rest of the code interacts with
//wavelength sources
//
//Wavelength sources only task as the name implies is to sample wavelengths.
//This abstraction allows to share common code between forward and backward
//tracing.
//
//The API consists of two functions and one may choose to implement one or the
//the other or both. They differ in that one can return an additional
//contribution factor.

/**
 * Samples a wavelength. All samples must have equal (implicit) contribution.
 * Mainly used for particle tracing.
*/
float sampleWavelength(
    uint idx, inout uint dim    ///< RNG state
);

/**
 * Samples a wavelength. Returns also the corresponding contribution, that is
 * the reciprocal probability of that sample.
*/
float sampleWavelength(
    out float contrib,          ///< Sample contribution
    uint idx, inout uint dim    ///< RNG state
);
