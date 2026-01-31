import pytest

import hephaistos as hp
import numpy as np
import scipy.stats as stats
import theia.response
import theia.units as u

from theia.random import PhiloxRNG
from theia.ray import UnpolarizedRay

from hephaistos.pipeline import runPipeline, runPipelineStage


def test_record(rng):
    N = 8192

    ray = UnpolarizedRay()
    record = theia.response.HitRecorder()
    replay = theia.response.HitReplay(N, ray, record)

    buffer = replay.queue.buffer(0)
    assert buffer is not None
    samples = buffer.view
    samples.count = N
    samples["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    samples["direction"] = rng.random((N, 3))
    samples["normal"] = rng.random((N, 3))
    samples["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    samples["time"] = (rng.random((N,)) * 100.0) * u.ns
    samples["objectId"] = rng.integers(0, 10, N)
    # use contrib to encode ordering (gpu will scramble items)
    samples["contrib"] = np.arange(N).astype(np.float32)

    runPipeline(replay.collectStages())

    assert record.queue is not None
    results = record.queue.view(0)
    assert results.count == N
    # restore ordering via contrib
    results = results[np.argsort(results["contrib"])]
    assert np.all(samples["position"] == results["position"])
    assert np.all(samples["direction"] == results["direction"])
    assert np.all(samples["normal"] == results["normal"])
    assert np.all(samples["wavelength"] == results["wavelength"])
    assert np.all(samples["time"] == results["time"])
    assert np.all(samples["objectId"] == results["objectId"])
    assert np.all(samples["contrib"] == results["contrib"])


@pytest.mark.parametrize("useDouble", [False, True])
@pytest.mark.parametrize("useCPU", [False, True])
def test_BinReducer(rng, useDouble: bool, useCPU):
    N_BINS = 100
    N_BUF = 16 if useCPU else 128
    NORM = 1.0

    reducer = theia.response.BinReducer(
        binCount=N_BINS,
        bufferCount=N_BUF,
        normalization=NORM,
        useDoublePrecision=useDouble,
    )
    # fill input buffer
    if useDouble:
        buffer = hp.DoubleBuffer(N_BINS * N_BUF)
    else:
        buffer = hp.FloatBuffer(N_BINS * N_BUF)
    buffer.numpy()[:] = rng.random(N_BINS * N_BUF)
    hp.execute(hp.updateTensor(buffer, reducer.bufferIn))
    # run reducer
    runPipelineStage(reducer)

    # calculate expected result
    exp = buffer.numpy().reshape((N_BUF, N_BINS)).sum(0)
    # check result
    result = reducer.result(0)
    assert np.allclose(exp, result)


@pytest.mark.parametrize("useDouble", [False, True])
def test_HistogramHitResponse(rng, useDouble: bool):
    # params
    N = 256 * 1024
    N_BINS = 50
    BIN_SIZE = 4.0 * u.ns
    T0 = 20.0 * u.ns
    T1 = T0 + N_BINS * BIN_SIZE
    NORM = 5e-4

    # create pipeline
    ray = UnpolarizedRay()
    value = theia.response.UniformValueResponse()
    response = theia.response.HistogramHitResponse(
        value,
        binCount=N_BINS,
        t0=T0,
        binSize=BIN_SIZE,
        normalization=NORM,
        bufferCount=64,
        useDoublePrecision=useDouble,
    )
    replay = theia.response.HitReplay(N, ray, response)
    # fill samples to process
    samples = replay.queue.view(0)
    samples["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    samples["direction"] = rng.random((N, 3))
    samples["normal"] = rng.random((N, 3))
    samples["objectId"] = rng.integers(0, 10, N)
    samples["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    samples["time"] = rng.random((N,)) * T1 * 1.2
    samples["contrib"] = rng.random((N,)) * 10.0
    # run pipeline
    runPipeline(replay.collectStages())

    # calculate expected results
    bin_edge = np.arange(N_BINS + 1) * BIN_SIZE + T0
    exp_hist, _ = np.histogram(samples["time"], bin_edge, weights=samples["contrib"])
    exp_hist *= NORM
    # check results
    result = response.result(0)
    # TODO: Somewhat large error independent of float vs double (about the same)
    #       -> Is there a difference in how we create the histogram and how numpy
    #          does it?
    assert np.allclose(result, exp_hist, rtol=1e-4)


@pytest.mark.parametrize("useDouble", [False, True])
def test_KernelHistogramHitResponse(rng, useDouble: bool):
    N = 64 * 1024
    N_BINS = 200
    BIN_SIZE = 1.2 * u.ns
    T0 = 20.0 * u.ns
    T1 = T0 + N_BINS * BIN_SIZE
    T_PEAK = T0 + 0.3 * (T1 - T0)
    BANDWIDTH = 10.0 * u.ns
    SUPPORT = 50.0 * u.ns
    NORM = 5e-4

    # create pipeline
    ray = UnpolarizedRay()
    value = theia.response.UniformValueResponse()
    response = theia.response.KernelHistogramHitResponse(
        value,
        binCount=N_BINS,
        t0=T0,
        binSize=BIN_SIZE,
        kernelBandwidth=BANDWIDTH,
        kernelSupport=SUPPORT,
        normalization=NORM,
        bufferCount=64,
        useDoublePrecision=useDouble,
    )
    replay = theia.response.HitReplay(N, ray, response)
    # fill samples to process
    samples = replay.queue.view(0)
    samples["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    samples["direction"] = rng.random((N, 3))
    samples["normal"] = rng.random((N, 3))
    samples["objectId"] = rng.integers(0, 10, N)
    samples["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    # samples["time"] = rng.random((N,)) * T1 * 1.2
    samples["time"] = rng.normal(T_PEAK, 10.0 * BIN_SIZE, (N,))
    samples["contrib"] = rng.random((N,)) * 10.0
    # run pipeline
    runPipeline(replay.collectStages())

    # calculate expected result
    x = np.arange(N_BINS + 1) * BIN_SIZE + T0
    exp_result = np.zeros_like(x)
    # TODO: At some point fix this slow loop...
    for i in range(N):
        t = samples["time"][i]
        v = samples["contrib"][i]
        exp_result += v * stats.norm.cdf(x, loc=t, scale=BANDWIDTH)
    exp_result = np.diff(exp_result) * NORM
    # check result
    result = np.asarray(response.result(0))
    # TODO: Somewhat large error independent of double vs. float
    #       -> Check our implementation is correct (CPU and GPU...)
    assert np.abs(result - exp_result).max() < 6e-4


def test_StoreValueHitResponse(rng):
    N = 32 * 1024

    # create pipeline
    ray = UnpolarizedRay()
    value = theia.response.UniformValueResponse()
    response = theia.response.StoreValueHitResponse(N, value)
    replay = theia.response.HitReplay(N, ray, response)
    # fill samples to process
    samples = replay.queue.view(0)
    samples["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    samples["direction"] = rng.random((N, 3))
    samples["normal"] = rng.random((N, 3))
    samples["objectId"] = np.arange(N)  # for matching values later
    samples["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    samples["time"] = rng.random((N,)) * 100.0 * u.ns
    samples["contrib"] = rng.random((N,)) * 10.0
    # run pipeline
    runPipeline(replay.collectStages())

    # check result
    result = response.queue.view(0)
    sort = np.argsort(result["objectId"])
    assert np.all(result["objectId"][sort] == samples["objectId"])
    assert np.allclose(result["time"][sort], samples["time"])
    assert np.allclose(result["value"][sort], samples["contrib"])


def test_StoreTimeHitResponse(rng):
    N = 32 * 1024

    # create pipeline
    ray = UnpolarizedRay()
    philox = PhiloxRNG(key=0xABBA)
    value = theia.response.UniformValueResponse()
    response = theia.response.StoreTimeHitResponse(N, value)
    replay = theia.response.HitReplay(N, ray, response, rng=philox)
    # fill samples to process
    samples = replay.queue.view(0)
    samples["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    samples["direction"] = rng.random((N, 3))
    samples["normal"] = rng.random((N, 3))
    samples["objectId"] = np.arange(N)  # for matching values later
    samples["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    samples["time"] = rng.random((N,)) * 100.0 * u.ns
    samples["contrib"] = rng.random((N,))
    # run pipeline
    runPipeline(replay.collectStages())

    # check result
    result = response.queue.view(0)
    # we expect about half the photons to be accepted
    assert abs(result.count / N - 0.5) < 0.005
    # we can't reproduce the sampling as the random numbers are lost
    # -> simply check whether we can find the sampled time stamps in the result
    assert len(np.unique(result["objectId"])) == result.count
    assert np.allclose(result["time"], samples["time"][result["objectId"]])


@pytest.mark.parametrize("integrateAll", [True, False])
@pytest.mark.parametrize("useDouble", [True, False])
def test_IntegratingHitResponse(rng, integrateAll: bool, useDouble: bool):
    N = 64 * 1024
    detCount = None if integrateAll else 3

    ray = UnpolarizedRay()
    value = theia.response.UniformValueResponse()
    response = theia.response.IntegratingHitResponse(
        value,
        detectorCount=detCount,
        bufferCount=256,
        useDoublePrecision=useDouble,
    )
    replay = theia.response.HitReplay(N, ray, response)

    samples = replay.queue.view(0)
    samples["contrib"] = 2.0 * rng.random((N,)) + 0.5
    samples["objectId"] = rng.integers(0, 3, N)

    runPipeline(replay.collectStages())

    if integrateAll:
        exp = samples["contrib"].sum()
    else:
        exp = np.array(
            [samples["contrib"][samples["objectId"] == i].sum() for i in range(3)]
        )
    results = response.result(0)
    assert np.allclose(results, exp)
