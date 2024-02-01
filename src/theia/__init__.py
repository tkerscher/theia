import hephaistos as hp
import importlib

# enable atomic add with floats
hp.enableAtomics({"sharedFloat32AtomicAdd"})
# enable ray tracing if available
if hp.isRaytracingSupported():
    hp.enableRaytracing()
else:
    from warnings import warn as _warn

    _warn(
        "Ray tracing is not supported on this machine! Some functions are not available."
    )
    # TODO: add checks in functions that require ray tracing

# check if there is suitable device available
if not hp.suitableDeviceAvailable():
    raise RuntimeError("No suitable device available!")

# import modules
submodules = [
    "estimator",
    "light",
    "lookup",
    "material",
    "random",
    "scene",
    "trace",
]
for module in submodules:
    globals()[module] = importlib.import_module(f"theia.{module}")

__all__ = submodules


def __dir__():
    return __all__
