import importlib
from . import device

# init device
device.initializeDevice()

# define submodules
__all__ = [
    "camera",
    "cascades",
    "compiler",
    "device",
    "light",
    "lookup",
    "material",
    "property",
    "random",
    "ray",
    "response",
    "scene",
    "surface",
    "target",
    "task",
    "testing",
    "trace",
    "units",
    "util",
    "volume",
]


def __dir__():
    return __all__


# lazy import of submodules
def __getattr__(attr):
    if attr == "device":
        return device
    if attr in __all__:
        return importlib.import_module(f"theia.{attr}")

    raise AttributeError(f"module 'theia' has no attribute {attr!r}")
