import hephaistos as hp

# globally enable ray tracing
if not hp.isRaytracingSupported():
    raise RuntimeError("No Raytracing capable hardware available!")
hp.enableRaytracing()
