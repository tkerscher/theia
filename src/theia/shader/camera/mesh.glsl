#ifndef _INCLUDE_CAMERA_MESH
#define _INCLUDE_CAMERA_MESH

#include "math.glsl"
#include "scene/geometry.glsl"
#include "util/offset.glsl"
#include "util/sample.glsl"

uniform CameraParams {
    uvec2 verticesAddress;
    uvec2 indicesAddress;
    uint triangleCount;

    float outward; // outward ? 1.0 : -1.0
    float timeDelta;

    uint mediumIdx;
    int objectId;

    mat4x3 objToWorld;
    mat4x3 worldToObj;
} cameraParams;

struct CameraSample {
    vec3 position;
    vec3 normal;
    float contrib;
    uint mediumIdx;

    vec3 hitPosition;
    vec3 hitNormal;
};

CameraSample sampleCamera(float wavelength, uint idx, inout uint dim) {
    //sample triangle
    float u = random(idx, dim);
    u *= float(cameraParams.triangleCount);
    uint triIdx = min(uint(floor(u)), cameraParams.triangleCount - 1);
    //fetch triangle
    ivec3 index = Index(cameraParams.indicesAddress)[triIdx].idx;
    Vertex vertices = Vertex(cameraParams.verticesAddress);
    Vertex v0 = vertices[index.x];
    Vertex v1 = vertices[index.y];
    Vertex v2 = vertices[index.z];

    //sample point on triangle
    vec2 barys = random2D(idx, dim);
    barys = vec2(1.0 - sqrt(barys.x), barys.y * sqrt(barys.x)); //ensure uniform
    vec3 e1 = v1.position - v0.position;
    vec3 e2 = v2.position - v0.position;
    vec3 localPos = v0.position + fma(vec3(barys.x), e1, barys.y * e2);
    #ifndef OUTWARD_FACE_CLOCK_WISE
    vec3 localNrm = normalize(cross(e1, e2));
    #else
    vec3 localNrm = normalize(cross(e2, e2));
    #endif
    localNrm *= cameraParams.outward; //flip sign dependent on direction

    //transform from local to world
    vec3 rayPos = mat3(cameraParams.objToWorld) * localPos + cameraParams.objToWorld[3];
    vec3 rayNrm = normalize(vec3(localNrm * cameraParams.worldToObj));
    //offset ray to prevent self intersection
    rayPos = offsetRay(rayPos, rayNrm);

    //calculate contribution
    e1 = mat3(cameraParams.objToWorld) * e1 + cameraParams.objToWorld[3];
    e2 = mat3(cameraParams.objToWorld) * e2 + cameraParams.objToWorld[3];
    float area = 0.5 * length(cross(e1, e2));
    float contrib = area * float(cameraParams.triangleCount);

    //assemble sample
    return CameraSample(
        rayPos,
        rayNrm,
        contrib,
        cameraParams.mediumIdx,
        localPos,
        localNrm
    );
}

BackwardRay createCameraRay(
    const CameraSample cam,
    vec3 lightDir,
    float wavelength,
    out CameraHit hit
) {
    //convert lightDir to object space
    vec3 hitDir = mat3(cameraParams.worldToObj) * lightDir;
    
    //calculate contribution
    float cos_theta = dot(lightDir, -cam.normal);
    float contrib = cam.contrib * cos_theta;
    //check light comes from the right side
    contrib *= float(dot(cam.normal, lightDir) < 0.0);

    //assemble camera hit
    hit = createCameraHit(
        cam.hitPosition,
        hitDir,
        cam.hitNormal,
        cameraParams.objectId
    );
    //assemble backward ray
    return createBackwardRay(
        cam.position,
        -lightDir,
        wavelength,
        cameraParams.mediumIdx,
        cameraParams.timeDelta,
        contrib
    );
}

BackwardRay sampleCameraRay(
    float wavelength,
    out CameraHit hit,
    uint idx, inout uint dim
) {
    //sample position
    CameraSample camSample = sampleCamera(wavelength, idx, dim);    

    //sample ray direction (upper hemisphere)
    vec3 localDir = sampleHemisphere(random2D(idx, dim));
    float cos_theta = localDir.z;
    //align direction with normal
    localDir = createLocalCOSY(camSample.hitNormal) * localDir;
    //transform from local to world
    vec3 rayDir = normalize(mat3(cameraParams.objToWorld) * localDir);    
    //calculate contribution
    float contrib = cos_theta * TWO_PI * camSample.contrib;

    //assemble camera hit
    hit = createCameraHit(
        camSample.hitPosition,
        -localDir,
        camSample.hitNormal,
        cameraParams.objectId
    );
    //assemble backward ray
    return createBackwardRay(
        camSample.position,
        rayDir,
        wavelength,
        cameraParams.mediumIdx,
        cameraParams.timeDelta,
        contrib
    );
}

#endif
