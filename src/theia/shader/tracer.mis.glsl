#ifndef _INCLUDE_TRACER_MIS
#define _INCLUDE_TRACER_MIS

#include "sphere.intersect.glsl"
#include "scatter.volume.glsl"

//MIS is a sampling method that while it increases the estimators variance, it
//will usually generate much more samples resulting in a faster reduction
//(in terms of time). The following procedure aims to minimize the increase in
//variance while using this method.

/*******************************************************************************
 * MIS: sample both scattering phase function & detector                       *
 *                                                                             *
 * We'll use the following naming scheme: pXY, where                           *
 * X: prob, distribution                                                       *
 * Y: sampled distribution                                                     *
 * S: scatter, D: detector                                                     *
 * e.g. pDS: p_det(dir ~ scatter)                                              *
 ******************************************************************************/

void scatterMIS_sample(
    const Sphere target, const Medium medium,
    const vec3 position, const vec3 direction,
    const vec2 rngDet, const vec2 rngScatter,
    out vec3 detDir, out float detDist,
    out vec3 scatterDir,
    out float pDD, out float pDS, out float pSS, out float pSD
) {
    //sample detector
    detDir = sampleSphere(target, position, rngDet, detDist, pDD);
    //sample scatter phase function
    scatterDir = scatter(medium, direction, rngScatter, pSS);
    //calculate cross probs pSD, pDS
    pSD = scatterProb(medium, direction, detDir);
    pDS = sampleSphereProb(target, position, scatterDir);
}

void scatterMIS_balance(
    const Sphere target, const Medium medium,
    const vec3 position, const vec3 direction,
    const vec2 rngDet, const vec2 rngScatter,
    //!NOTE! w_det includes attenuation from phase function, but not mu_s
    out vec3 detDir, out float w_det, out float detDist,
    out vec3 scatterDir, out float w_scatter
) {
    //sample all strategies
    float pDD, pDS, pSS, pSD;
    scatterMIS_sample(
        target, medium,
        position, direction,
        rngDet, rngScatter,
        detDir, detDist, scatterDir,
        pDD, pDS, pSS, pSD
    );

    //calculate MIS weights
    w_scatter = pSS / (pSS + pSD);
    //For the detector weight, two effects happen: attenuation due to
    //phase function (= pSD) and canceling of sampled distribution:
    //  f(x)*phase(theta)/pDD * w_det = f(x)*pSD/pDD * w_det
    //Note that mu_s is missing, but since both paths need it, we expect to
    //happen elsewhere
    w_det = pSD / (pDD + pDS);
}

void scatterMIS_power(
    const Sphere target, const Medium medium,
    const vec3 position, const vec3 direction,
    const vec2 rngDet, const vec2 rngScatter,
    //!NOTE! w_det includes attenuation from phase function, but not mu_s
    out vec3 detDir, out float w_det, out float detDist,
    out vec3 scatterDir, out float w_scatter    
) {
    //sample all strategies
    float pDD, pDS, pSS, pSD;
    scatterMIS_sample(
        target, medium,
        position, direction,
        rngDet, rngScatter,
        detDir, detDist, scatterDir,
        pDD, pDS, pSS, pSD
    );
    
    //calculate MIS weights
    w_scatter = pSS*pSS / (pSS*pSS + pSD*pSD);
    //For the detector weight, two effects happen: attenuation due to
    //phase function (= pSD) and canceling of sampled distribution:
    //  f(x)*phase(theta)/pDD * w_det = f(x)*pSD/pDD * w_det
    //Note that mu_s is missing, but since both paths need it, we expect to
    //happen elsewhere
    w_det = pDD*pSD / (pDD*pDD + pDS*pDS);
}

#endif
