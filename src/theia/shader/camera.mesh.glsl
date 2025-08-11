#ifndef _INCLUDE_CAMERARAYSOURCE_MESH
#define _INCLUDE_CAMERARAYSOURCE_MESH

#include "math.glsl"
#include "ray.surface.glsl"
#include "scene.types.glsl"
#include "util.sample.glsl"

uniform CameraParams {
    uvec2 verticesAddress;
    uvec2 indicesAddress;
    uint triangleCount;

    float outward; // outward ? 1.0 : -1.0
    float timeDelta;

    mat4x3 objToWorld;
    mat4x3 worldToObj;
} cameraParams;

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
    vec3 n1 = v1.normal - v0.normal;
    vec3 n2 = v2.normal - v0.normal;
    vec3 localNrm = normalize(cross(e1, e2));
    vec3 intNrm = v0.normal + fma(vec3(barys.x), n1, barys.y * n2);
    localNrm *= signBit(dot(localNrm, intNrm)); //align geometric and interpolated normal
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
    return createCameraSample(
        rayPos,
        rayNrm,
        contrib,
        localPos,
        localNrm
    );
}

CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim) {
    //sample position
    CameraSample camSample = sampleCamera(wavelength, idx, dim);    

    //sample ray direction (upper hemisphere)
    vec3 localDir = sampleHemisphere(random2D(idx, dim));
    float cos_theta = localDir.z;
    //align direction with normal
    localDir = createLocalCOSY(camSample.hitNormal) * localDir;
    //transform from local to world
    vec3 rayDir = normalize(mat3(cameraParams.objToWorld) * localDir);
    //create polarization reference frame in plane of incidence
    vec3 hitPolRef, polRef;
    mat4 mueller;
    #ifdef POLARIZATION
    hitPolRef = perpendicularTo(localDir, camSample.hitNormal);
    vec3 worldNormal = normalize(vec3(camSample.hitNormal * cameraParams.worldToObj));
    polRef = perpendicularTo(rayDir, worldNormal);
    //for non-orthogonal transformation the transformed polRef may not lie in
    //the plane of incidence, but require a rotation of the stokes parameter
    vec3 expPolRef = normalize(mat3(cameraParams.objToWorld) * hitPolRef);
    mueller = alignPolRef(-rayDir, polRef, expPolRef);
    #endif
    
    //calculate contribution
    float contrib = cos_theta * TWO_PI * camSample.contrib;

    //assemble camera ray
    return createCameraRay(
        camSample.position,         //ray position
        rayDir,                     //ray direction
        polRef,                     //ray polRef
        mueller,                    //ray mueller matrix
        contrib,                    //contribution
        cameraParams.timeDelta,  //time delta
        camSample.hitPosition,      //hit pos in object space
        -localDir,                  //local (light) dir in object space
        camSample.hitNormal,        //normal on unit sphere
        hitPolRef                   //hit polRef
    );
}

CameraRay createCameraRay(CameraSample cam, vec3 lightDir, float wavelength) {
    //convert lightDir to object space
    vec3 hitDir = mat3(cameraParams.worldToObj) * lightDir;
    
    //calculate contribution
    float cos_theta = dot(lightDir, -cam.normal);
    float contrib = cam.contrib * cos_theta;
    //check light comes from the right side
    contrib *= float(dot(cam.normal, lightDir) < 0.0);

    //create polarization reference frame in plane of incidence
    vec3 hitPolRef, polRef;
    mat4 mueller;
    #ifdef POLARIZATION
    hitPolRef = perpendicularTo(hitDir, cam.hitNormal);
    vec3 worldNormal = normalize(vec3(cam.hitNormal * cameraParams.worldToObj));
    polRef = perpendicularTo(lightDir, worldNormal);
    //for non-orthogonal transformation the transformed polRef may not lie in
    //the plane of incidence, but require a rotation of the stokes parameter
    vec3 expPolRef = normalize(mat3(cameraParams.objToWorld) * hitPolRef);
    mueller = alignPolRef(lightDir, polRef, expPolRef);
    #endif

    //assemble ray
    return createCameraRay(
        cam.position,
        -lightDir,
        polRef,
        mueller,
        contrib,
        cameraParams.timeDelta,
        cam.hitPosition,
        hitDir,
        cam.hitNormal,
        hitPolRef
    );
}

#endif
