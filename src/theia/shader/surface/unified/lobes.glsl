#ifndef _INCLUDE_SURFACE_UNIFIED_LOBES
#define _INCLUDE_SURFACE_UNIFIED_LOBES

#include "math.glsl"

/*
Reflection-lobe logic of the Geant4 UNIFIED model (see template.glsl). The
reflected component is decomposed into specular spike, specular lobe, diffuse
lobe and backscattering. The split is controlled by the optional material
properties `prob_backscatter`, `prob_specularspike`, `prob_specularlobe` and
`prob_diffuselobe`.
*/

struct ReflectionLobes {
    bool doBackScatter;
    bool doSpecularSpike;
    bool doSpecularLobe;
    bool doDiffuseLobe;
};

//Samples which reflection lobe to use. Falls back to a pure specular lobe (the
//default case) when the probabilities are not provided or backscatter/spike/
//diffuse are all zero (to catch the case where the material slot exists but no value
//is specified for this material, such that zeros are written by default)
ReflectionLobes sampleReflectionLobes(const SurfaceHit hit, uint idx, inout uint dim) {
    float probBackScatter, probSpecularSpike, probSpecularLobe, probDiffuseLobe;
    #if defined(MATERIAL_SLOT_PROB_BACKSCATTER) && defined(MATERIAL_SLOT_PROB_SPECULARSPIKE) \
        && defined(MATERIAL_SLOT_PROB_SPECULARLOBE) && defined(MATERIAL_SLOT_PROB_DIFFUSELOBE)
        probBackScatter   = loadMaterialConstant(PROB_BACKSCATTER, hit.materialIdx);
        probSpecularSpike = loadMaterialConstant(PROB_SPECULARSPIKE, hit.materialIdx);
        probSpecularLobe  = loadMaterialConstant(PROB_SPECULARLOBE, hit.materialIdx);
        probDiffuseLobe   = loadMaterialConstant(PROB_DIFFUSELOBE, hit.materialIdx);
        bool isDefault = (probBackScatter == 0.0 && probSpecularSpike == 0.0 && probDiffuseLobe == 0.0);
    #else
        bool isDefault = true;
    #endif

    ReflectionLobes lobes;
    lobes.doBackScatter = false;
    lobes.doSpecularSpike = false;
    lobes.doSpecularLobe = true;
    lobes.doDiffuseLobe = false;
    if (!isDefault) {
        //normalize the probabilities so they sum to 1 (the cumulative thresholds
        //below assume a normalized distribution). 
        float probTotal = probBackScatter + probSpecularSpike + probSpecularLobe + probDiffuseLobe;
        if (abs(1 - probTotal) > 1.0e-5) {
            probBackScatter /= probTotal;
            probSpecularSpike /= probTotal;
            probSpecularLobe /= probTotal;
            probDiffuseLobe /= probTotal;
        }
        float u = random(idx, dim);
        lobes.doBackScatter   = (u < probBackScatter);
        lobes.doSpecularSpike = (!lobes.doBackScatter && u < probBackScatter + probSpecularSpike);
        lobes.doSpecularLobe  = (!lobes.doBackScatter && !lobes.doSpecularSpike &&
            u < probBackScatter + probSpecularSpike + probSpecularLobe);
        lobes.doDiffuseLobe   = (!lobes.doBackScatter && !lobes.doSpecularSpike && !lobes.doSpecularLobe);
    }
    return lobes;
}

//Returns the actual reflected direction for the sampled lobe. `microfacetNormal`
//is used by the specular lobe, `diffuseDir` is a pre-sampled cosine-weighted
//hemisphere direction used by the diffuse lobe.
vec3 lobeReflectedDir(
    const ReflectionLobes lobes,
    vec3 rayDir, vec3 rayNrm, vec3 microfacetNormal, vec3 diffuseDir
) {
    if (lobes.doSpecularLobe) {
        return reflect(rayDir, microfacetNormal);
    }
    else if (lobes.doSpecularSpike) {
        return reflect(rayDir, rayNrm);
    }
    else if (lobes.doBackScatter) {
        return -rayDir;
    }
    else { //doDiffuseLobe
        return diffuseDir;
    }
}

#endif
