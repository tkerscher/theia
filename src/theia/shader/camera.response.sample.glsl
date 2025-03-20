#ifndef BLOCK_SIZE
#error "BLOCK_SIZE not defined"
#endif

layout(local_size_x = BLOCK_SIZE) in;

#include "camera.common.glsl"
#include "wavelengthsource.common.glsl"
#include "response.common.glsl"

//user provided code
#include "rng.glsl"
#include "camera.glsl"
#include "photon.glsl"
#include "response.glsl"

//output queue
layout(scalar) writeonly buffer QueueOut {
    float posX[QUEUE_SIZE];
    float posY[QUEUE_SIZE];
    float posZ[QUEUE_SIZE];
    float dirX[QUEUE_SIZE];
    float dirY[QUEUE_SIZE];
    float dirZ[QUEUE_SIZE];
    float nrmX[QUEUE_SIZE];
    float nrmY[QUEUE_SIZE];
    float nrmZ[QUEUE_SIZE];

    float wavelength[QUEUE_SIZE];
    float timeDelta[QUEUE_SIZE];
    float contrib[QUEUE_SIZE];
    
    #ifdef POLARIZATION
    float polRefX[QUEUE_SIZE];
    float polRefY[QUEUE_SIZE];
    float polRefZ[QUEUE_SIZE];
    //stokes vector
    float Q[QUEUE_SIZE];
    float U[QUEUE_SIZE];
    float V[QUEUE_SIZE];
    #endif
} queueOut;

void sampleMain() {
    uint dim = 0;
    uint idx = gl_GlobalInvocationID.x;
    
    //sample camera
    WavelengthSample photon = sampleWavelength(idx, dim);
    CameraRay ray = sampleCameraRay(photon.wavelength, idx, dim);

    //save samples
    queueOut.posX[idx] = ray.hit.position.x;
    queueOut.posY[idx] = ray.hit.position.y;
    queueOut.posZ[idx] = ray.hit.position.z;
    queueOut.dirX[idx] = ray.hit.direction.x;
    queueOut.dirY[idx] = ray.hit.direction.y;
    queueOut.dirZ[idx] = ray.hit.direction.z;
    queueOut.nrmX[idx] = ray.hit.normal.x;
    queueOut.nrmY[idx] = ray.hit.normal.y;
    queueOut.nrmZ[idx] = ray.hit.normal.z;

    queueOut.wavelength[idx] = photon.wavelength;
    queueOut.timeDelta[idx] = ray.timeDelta;
    queueOut.contrib[idx] = ray.contrib;

    #ifdef POLARIZATION
    queueOut.polRefX[idx] = ray.hit.polRef.x;
    queueOut.polRefY[idx] = ray.hit.polRef.y;
    queueOut.polRefZ[idx] = ray.hit.polRef.z;

    //sample and save random stokes vector
    float Q = random(idx, dim);
    float U = random(idx, dim);
    float V = random(idx, dim);
    vec4 stokes = vec4(1.0, Q, U, V);

    queueOut.Q[idx] = Q;
    queueOut.U[idx] = U;
    queueOut.V[idx] = V;
    #endif

    //pass sampled hit to response
    HitItem hit = HitItem(
        ray.hit.position,
        ray.hit.direction,
        ray.hit.normal,
        #ifdef POLARIZATION
        stokes,
        ray.hit.polRef,
        #endif
        photon.wavelength,
        ray.timeDelta,
        ray.contrib
    );
    response(hit);
}

void main() {
    initResponse();
    sampleMain();
    finalizeResponse();
}
