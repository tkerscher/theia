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

# define submodules
__all__ = [
    "camera",
    "cascades",
    "light",
    "lookup",
    "material",
    "random",
    "response",
    "scene",
    "target",
    "task",
    "testing",
    "trace",
    "units",
]


def __dir__():
    return __all__


# lazy import of submodules
def __getattr__(attr):
    if attr in __all__:
        return importlib.import_module(f"theia.{attr}")

    raise AttributeError(f"module 'theia' has no attribute {attr!r}")
