# Theia

Theia is a package for creating Monte Carlo simulation of optionally polarized
light propagation through volumes of arbitrary shapes containing
(non-)scattering media while keeping track of the elapsed time. This includes
physically correct reflections and transmission at their boundaries. The
simulation runs on the GPU utilizing dedicated ray tracing hardware found on
modern ones for increased performance.

This package allows through its modular design maximal flexibility in
defining simulations. While this also includes the produced results, usually
one wants to produce _light curves_ describing the expected signal at the
detector as function of time:

![light curve example](docs/images/light_curve.png)

## Installation

Theia is a python package and can installed using pip via a single line:

```bash
pip install git+https://github.com/tkerscher/theia
```

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

## Documentation and Examples

The documentation is contained in the `docs` directory. For example notebooks
on how to use this package see the `notebooks` directory.
