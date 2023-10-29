import hephaistos as hp

# enable atomic add with floats
hp.enableAtomics({"sharedFloat32AtomicAdd"})
# enable ray tracing
hp.enableRaytracing()

# check if there is suitable device available
if not hp.suitableDeviceAvailable():
    raise RuntimeError("No suitable device available!")
