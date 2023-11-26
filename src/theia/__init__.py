import hephaistos as hp

# enable atomic add with floats
hp.enableAtomics({"sharedFloat32AtomicAdd"})
# enable ray tracing if available
if hp.isRaytracingSupported():
    hp.enableRaytracing()
else:
    import warnings

    warnings.warn(
        "Ray tracing is not supported on this machine! Some functions are not available."
    )
    # TODO: add checks in functions that require ray tracing

# check if there is suitable device available
if not hp.suitableDeviceAvailable():
    raise RuntimeError("No suitable device available!")
