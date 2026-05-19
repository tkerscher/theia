from __future__ import annotations

import hephaistos as hp
import warnings

from functools import cache

__all__ = [
    "initializeDevice",
    "isDeviceSuitable",
    "isRayTracingEnabled",
    "getEnabledRayTracingFeatures",
    "selectDevice",
]


def __dir__():
    return __all__


@cache
def getEnabledAtomics() -> set[hp.Atomics]:
    """Returns a cached version of currently enabled atomics"""
    return hp.getEnabledAtomics()


@cache
def getEnabledRayTracingFeatures() -> hp.RayTracingFeatures:
    """Returns a cached version of currently enabled ray tracing features"""
    return hp.getEnabledRayTracingFeatures()


def isRayTracingEnabled() -> bool:
    """Returns True, if ray tracing is currently enabled"""
    # at minimum, we need ray tracing pipelines
    return getEnabledRayTracingFeatures().pipeline


def isDeviceSuitable(id: int, *, requireRayTracing: bool = False) -> bool:
    """Checks whether the device given by its id can run theia at all"""
    # check for ray tracing
    if requireRayTracing:
        rtSupport = hp.getRayTracingFeatures(id)
        if not rtSupport.pipeline:
            return False
    # it must support 64 bit integers
    supportedTypes = hp.getSupportedTypes(id)
    if not supportedTypes.int64:
        return False
    # must support atomic add for 32 bit floats
    atomics = hp.getSupportedAtomics(id)
    if hp.Atomics.BufferFloat32Add not in atomics:
        return False

    # everything needed is there
    return True


def selectDevice() -> int | None:
    """
    Selects the most suitable GPU present. Returns its id or `None` if no
    suitable device was found
    """
    # we want to select a device on the following criteria in descending priority
    # - full ray tracing support
    # - partial ray tracing support
    # - discrete GPU
    # This means that we will select a software implementation if no ray tracing
    # GPU is found. We issue a warning in that case

    # query devices
    devices = hp.enumerateDevices()
    maxScore = -1
    selectedId = None
    discreteFound = False
    for i in range(len(devices)):
        # suitable
        if not isDeviceSuitable(i):
            continue
        # query ray tracing support
        rt_features = hp.getRayTracingFeatures(i)
        rt_support = rt_features.pipeline and rt_features.query
        rt_full = rt_support and rt_features.indirectDispatch
        # calculate score
        score = 0
        if rt_full:
            score += 2 << 3
        if rt_support:
            score += 2 << 2
        if devices[i].isDiscrete:
            score += 2 << 1
            discreteFound = True
        # choose this device if better
        if score > maxScore:
            selectedId = i
            maxScore = score

    # issue warnings
    if selectedId is not None and discreteFound and not devices[selectedId].isDiscrete:
        warnings.warn(
            "A discrete GPU is present but was not selected as it lacks (full) "
            "ray tracing support. You can override this decision and choose a "
            "specific device using hephaistos.selectDevice()."
        )

    return selectedId


def initializeDevice(*, useSelectedDevice: bool = True, force: bool = False) -> bool:
    """
    Configures and initializes the selected device for use with `theia`.
    If no device has been selected, a suitable one will be chosen automatically.
    See `hephaistos.selectDevice` for how to pre-select a device.

    Parameters
    ----------
    useSelectedDevice: bool, default=True
        If `True`, use currently selected device if any, otherwise ignores the
        selection.
    force: bool, default=False
        Whether to destroy an already existing GPU context. If `False` and such
        a context already exist, an exception will be raised.
    """
    # cyclic dependency -> load lazily inside function
    from theia.compiler import getPreamble

    # reset any cached device specific properties
    getEnabledAtomics.cache_clear()
    getEnabledRayTracingFeatures.cache_clear()
    getPreamble.cache_clear()

    # select most suitable GPU if none is specified
    if deviceId := hp.getSelectedDeviceId() is None or not useSelectedDevice:
        deviceId = selectDevice()
        if deviceId is None:
            warnings.warn("No suitable device found to run theia")
            return False  # init failed
        hp.selectDevice(deviceId)

    # check extended type support
    supportedTypes = hp.getSupportedTypes(deviceId)
    if not supportedTypes.int64:
        warnings.warn("Selected device does not support 64 bit integers.")
        return False
    if not supportedTypes.float64:
        warnings.warn(
            "Selected device does not support double precision floats. "
            "Some features might not be available."
        )

    # configure gpu

    # query atomics support
    requestedAtomics: set[hp.Atomics] = {hp.Atomics.BufferFloat32Add}
    supportedAtomics = hp.getSupportedAtomics(deviceId)
    if hp.Atomics.BufferFloat64Add in supportedAtomics:
        requestedAtomics.add(hp.Atomics.BufferFloat64Add)
    if hp.Atomics.BufferInt64 in supportedAtomics:
        requestedAtomics.add(hp.Atomics.BufferInt64)
    else:
        warnings.warn(
            "Selected device does not support 64 bit atomics. "
            "Some features might not be available."
        )
    hp.enableAtomics(requestedAtomics)

    # query ray tracing support
    rt_features = hp.getRayTracingFeatures(deviceId)
    rt_supported = rt_features.pipeline and rt_features.query
    # select ray tracing features we may need
    rt_request = hp.RayTracingFeatures(
        query=rt_supported,
        pipeline=rt_supported,
        indirectDispatch=rt_supported and rt_features.indirectDispatch,
        positionFetch=rt_supported and rt_features.positionFetch,
        hitObjects=False,  # currently not widely supported
    )
    hp.enableRayTracing(rt_request, force=force)
    # force init of context to see if everything went well
    hp.getCurrentDevice()

    # tell the user if no ray tracing is supported
    if not rt_supported:
        warnings.warn(
            "Ray tracing is not supported on this system or selected device. "
            "Some functions of theia are thus not available."
        )
    # on some hardware ray tracing is only emulated in software as they lack the
    # dedicated hardware. Let the user know. A common indicator is only
    # supporting ray tracing pipelines (at least on nvidia)
    if rt_supported and not rt_features.query:
        warnings.warn(
            "The selected device reports support for only a reduced set of "
            "ray tracing features. Usually this indicates implementation of "
            "ray tracing through software. Performance on this device might "
            "be less than expected."
        )

    # success
    return True
