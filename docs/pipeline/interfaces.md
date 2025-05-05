# Component Interfaces

Here we will list the interfaces of the [simulation components](components.md)
on the GPU side.

## Camera

??? quote "GLSL Code"

    ```GLSL
    //Camera Interface
    CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim);
    //Optional. Required for direct tracing
    CameraSample sampleCamera(float wavelength, uint idx, inout uint dim);
    CameraRay createCameraRay(CameraSample cam, vec3 lightDir, float wavelength);
    ```

    ```GLSL
    //defined in "camera.common.glsl" (already included by tracer)
    struct CameraSample {
        vec3 position;
        vec3 normal;

        float contrib;

        int objectId;

        vec3 hitPosition;
        vec3 hitNormal;
    };

    struct CameraHit {
        #ifdef POLARIZATION
        vec3 polRef;
        #endif

        vec3 position;
        vec3 direction;
        vec3 normal;
    };

    struct CameraRay {
        vec3 position;
        vec3 direction;

        #ifdef POLARIZATION
        vec3 polRef;
        mat4 mueller;
        #endif

        float contrib;
        float timeDelta;

        CameraHit hit;
    };
    ```

### `CameraRay sampleCameraRay(float wavelength, uint idx, inout uint dim)` { data-toc-label="sampleCameraRay" }

Samples a ray originating at the detector used to initialize a tracer for the
given wavelength.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `wavelength` | `float` | Wavelength in nm to be simulated. |
| `idx` | `uint` | RNG stream number. |
| `dim` | `uint` | RNG count into stream. |

**Returns**

`CameraRay`: Sampled camera ray.

### `CameraSample sampleCamera(float wavelength, uint idx, inout uint dim)` { data-toc-label="sampleCamera" }

Samples a position on the detector used in direct tracing.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `wavelength` | `float` | Wavelength in nm to be simulated. |
| `idx` | `uint` | RNG stream number. |
| `dim` | `uint` | RNG count into stream. |

**Returns**

`CameraSample`: Sampled position on the detector.

### `CameraRay createCameraRay(CameraSample cam, vec3 lightDir, float wavelength)` { data-toc-label="createCameraRay" }

Creates a camera ray from the given `CameraSample` previously returned from
`sampleCamera` and an incident light direction.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `cam` | `CameraSample` | Sample previously returned from `sampleCamera`. |
| `lightDir` | `vec3` | Incident light direction. |
| `wavelength` | `float` | Wavelength in nm to be simulated. |

**Returns**

`CameraRay`: Sampled camera ray.

## Hit Response

??? quote "GLSL Code"

    ```GLSL
    void initResponse();
    void response(HitItem item);
    void finalizeResponse();
    ```

    ```GLSL
    //defined in "response.common.glsl" (already included by tracer)
    struct HitItem {
        vec3 position;
        vec3 direction;
        vec3 normal;

        #ifdef POLARIZATION
        vec4 stokes;
        vec3 polRef;
        #endif
        
        float wavelength;
        float time;
        float contrib;

        int objectId;
    };
    ```

### `void initResponse()` { data-toc-label="initResponse" }

Initializes the response. Called by the tracer at the start of the simulation.

### `void response(HitItem item)` { data-toc-label="response" }

Consumes the given `HitItem` produced by the tracer.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `item` | `HitItem` | Description of the hit produced by the tracer. |

### `finalizeResponse()` { data-toc-label="finalizeResponse" }

Finalizes the response. Called by the tracer after all tracing in a single
workgroup has finished.

## Light Source

??? quote "GLSL Code"

    ```GLSL
    SourceRay sampleLight(float wavelength, uint idx, inout uint dim);
    //optional for use in backward tracing
    SourceRay sampleLight(vec3 observer, vec3 normal, float wavelength, uint idx, inout uint dim);
    ```

    ```GLSL
    //defined in lightsource.common.glsl (already included by tracer)
    struct SourceRay {
        vec3 position;
        vec3 direction;

        #ifdef POLARIZATION
        vec4 stokes;
        vec3 polRef;
        #endif
        
        float startTime;
        float contrib;
    };
    ```

### `SourceRay sampleLight(float wavelength, uint idx, inout uint dim)` { data-toc-label="sampleLight" }

Samples the light source for a source ray used to initialize forward tracing.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `wavelength` | `float` | Wavelength in nm to be simulated. |
| `idx` | `uint` | RNG stream number. |
| `dim` | `uint` | RNG count into stream. |

**Returns**

`SourceRay`: Ray used to initialize forward tracing.

### `SourceRay sampleLight(vec3 observer, vec3 normal, float wavelength, uint idx, inout uint dim)` { data-toc-label="sampleLight" }

Samples the light source as observed at the given position. Used in backward and
direct tracing.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `observer` | `vec3` | Position from which the light source is observed. |
| `wavelength` | `float` | Wavelength in nm to be simulated. |
| `idx` | `uint` | RNG stream number. |
| `dim` | `uint` | RNG count into stream. |

**Returns**

`SourceRay`: Ray estimating emitted light into the direction of `observer`.

## Target

??? quote

    ```GLSL
    TargetSample sampleTarget(vec3 observer, uint idx, inout uint dim);
    TargetSample intersectTarget(vec3 observer, vec3 direction);
    bool isOccludedByTarget(vec3 position);
    ```

    ```GLSL
    struct TargetSample {
        vec3 position;      //Hit position in world space
        vec3 normal;        //Normal at position in world space
        float dist;         //Distance from observer to sample

        vec3 objPosition;   //Hit position in object space
        vec3 objNormal;     //Hit normal in object space
        
        float prob;         //Sample probability over Area (dA)
        bool valid;         //True for valid hit, false if missed

        vec3 offset;        //Translation from world to object space
        mat3 worldToObj;    //Orthogonal trafo from world to object space
    };
    ```

### `TargetSample sampleTarget(vec3 observer, uint idx, inout uint dim)` { data-toc-label="sampleTarget" }

Samples a point on the target ideally visible from the observer given by its
position.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `observer` | `vec3` | Position of the observer querying the target. |
| `idx` | `uint` | RNG stream number. |
| `dim` | `uint` | RNG count into stream. |

**Returns**

`TargetSample`: Sampled position on the target.

### `TargetSample intersectTarget(vec3 observer, vec3 direction)` { data-toc-label="intersectTarget" }

Determines whether the given ray intersects the target and returns an
appropriate sample indicating a hit by settings `valid` to `true`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `observer` | `vec3` | Origin of the ray. |
| `direction` | `vec3` | Direction of the ray. |

**Returns**

`TargetSample`: Intersection or miss of the ray with the target.

### `bool isOccludedByTarget(vec3 position)` { data-toc-label="isOccludedByTarget" }

Determines whether the given position is occluded by the target. Used by the
tracer to test starting points of light paths.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `position` | `vec3` | Position to be tested.

**Returns**

`bool`: True, if the position is occluded and no tracing should start there.

## Target Guide

??? quote

    ```GLSL
    TargetGuideSample sampleTargetGuide(vec3 observer, uint idx, inout uint dim);
    TargetGuideSample evalTargetGuide(vec3 observer, vec3 direction);
    ```

    ```GLSL
    //defined in "target_guide.common.glsl" (already included by tracer)
    struct TargetGuideSample {
        vec3 dir;       ///< Sampled direction
        float dist;     ///< Max dist to trace
        float prob;     ///< Prob of sampled direction
    };
    ```

### `TargetGuideSample sampleTargetGuide(vec3 observer, uint idx, inout uint dim)` { data-toc-label="sampleTargetGuide" }

Samples the target guide for a promising direction at the given observation
point.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `observer` | `vec3` | Point from which the target is sampled.
| `idx` | `uint` | RNG stream number. |
| `dim` | `uint` | RNG count into stream. |

**Returns**

`TargetGuideSample`: Sampled target guide.

### `TargetGuideSample evalTargetGuide(vec3 observer, vec3 direction)` { data-toc-label="evalTargetGuide" }

Evaluates the given ray using the underlying probability distribution of the
target guide.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `observer` | `vec3` | Origin of the ray. |
| `direction` | `vec3` | Direction of the ray. |

**Returns**

`TargetGuideSample`: Target guide sample evaluated for the given ray.

## Wavelength Source

??? quote

    ```GLSL
    WavelengthSample sampleWavelength(uint idx, uint dim);
    ```

    ````GLSL
    struct WavelengthSample {
        float wavelength;
        float contrib;
    };
    ```

### `WavelengthSample sampleWavelength(uint idx, uint dim)` { data-toc-label="sampleWavelength" }

Samples the source for a wavelength.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `idx` | `uint` | RNG stream number. |
| `dim` | `uint` | RNG count into stream. |

**Returns**

`WavelengthSample`: Sampled wavelength.
