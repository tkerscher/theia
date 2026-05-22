# Get Started

Theia is a framework for creating photon propagation simulations running on any
modern GPU utilizing hardware accelerated ray tracing. It is designed to be
modular and extendable allowing to incorporate different physical models.

## Installation

Theia is a Python package and can be installed via `pip`:

```bash
pip install git+https://github.com/tkerscher/theia
```

!!! note

    Theia relies on [hephaistos](https://github.com/tkerscher/hephaistos) to
    interact with the GPU, which contains compiled code. There are pre-built
    binaries uploaded to [`PyPI`](https://pypi.org/project/hephaistos/). If
    `pip` tries to compile it from source, you might want to check whether you
    use a Python version for which pre-builts exist.

## System Requirements

Theia currently runs on Linux and Windows, and requires a Vulkan compatible GPU.
Virtually all common vendors such as AMD, Intel and NVidia met this requirement
including the integrated graphics on CPUs.

For the full feature set a ray tracing capable GPU is required, which are:

- NVidia since GTX 1000 series (2016)
- AMD since RX6000 series (2020)
- Intel since Arc Alchemist (2022)

The integrated graphics on modern CPUs begin to support ray tracing, too:

- AMD since Ryzen 7000 series (2022)
- Intel since Core Ultra 100 Series (2023)

In case no suitable GPU is available, Mesa provides a software implementation
emulating a GPU on the CPU called _llvmpipe_. This is however strongly
discouraged as performance is expected to be much worse than even the slowest
GPUs.

### GPU Selection

If no GPU is explicitly select, theia choses the first suitable GPU while
preferring dedicated over integrated ones. To specify on which GPU to run the
simulation you have to first import `hephaistos` and select the desired device:

```python
import hephaistos as hp

# print all available devices
for device in hp.enumerateDevices():
    print(device)

# select the 3rd device
hp.selectDevice(2)

# now we can import theia
import theia
```

!!! warning

    If you want to use `selectDevice()`, you have to call it before importing
    `theia`.

### Mac Support

Apple implements their own version of the Vulkan API called _Metal_, that is not
out of the box compatible. There exist a translation layer that allow software
targeting Vulkan to run on MacOS, but this currently does not support ray
tracing. Thus for now, Mac is currently not supported but may become so in a
future version.

## What's next?

You can either browse the [examples](https://github.com/tkerscher/theia/tree/master/examples)
or read the [documentation](index.md).
