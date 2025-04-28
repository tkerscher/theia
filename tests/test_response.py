import pytest

import numpy as np
import scipy.stats as stats
import theia.response
import theia.units as u

from theia.camera import PointCamera, CameraRayItem, PolarizedCameraRayItem
from theia.light import UniformWavelengthSource, WavelengthSampleItem
from theia.random import PhiloxRNG

from hephaistos.pipeline import RetrieveTensorStage, UpdateTensorStage, runPipeline
from hephaistos.queue import QueueTensor, as_queue


@pytest.mark.parametrize("polarized", [True, False])
def test_record(rng, polarized: bool):
    N = 8192

    record = theia.response.HitRecorder(polarized=polarized)
    replay = theia.response.HitReplay(N, record, polarized=polarized)

    samples = replay.queue.view(0)
    samples["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    samples["direction"] = rng.random((N, 3))
    samples["normal"] = rng.random((N, 3))
    samples["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    samples["time"] = (rng.random((N,)) * 100.0) * u.ns
    # use contrib to encode ordering (gpu will scramble items)
    samples["contrib"] = np.arange(N).astype(np.float32)
    if polarized:
        samples["stokes"] = rng.random((N, 4))
        samples["polarizationRef"] = rng.random((N, 3))

    runPipeline(replay.collectStages())

    results = record.queue.view(0)
    assert results.count == N
    # restore ordering via contrib
    results = results[np.argsort(results["contrib"])]
    assert np.all(samples["position"] == results["position"])
    assert np.all(samples["direction"] == results["direction"])
    assert np.all(samples["normal"] == results["normal"])
    assert np.all(samples["wavelength"] == results["wavelength"])
    assert np.all(samples["time"] == results["time"])
    assert np.all(samples["contrib"] == results["contrib"])
    if polarized:
        assert np.all(samples["stokes"] == results["stokes"])
        assert np.all(samples["polarizationRef"] == results["polarizationRef"])


def test_hitPolarizationMismatch():
    with pytest.raises(RuntimeError):
        record = theia.response.HitRecorder(polarized=True)
        replay = theia.response.HitReplay(128, record, polarized=False)
    with pytest.raises(RuntimeError):
        record = theia.response.HitRecorder(polarized=False)
        replay = theia.response.HitReplay(128, record, polarized=True)


def test_histogramResponse(rng):
    N = 256 * 1024
    N_BINS = 50
    BIN_SIZE = 4.0 * u.ns
    T0 = 20.0 * u.ns
    T1 = T0 + N_BINS * BIN_SIZE
    NORM = 1e-5

    value = theia.response.UniformValueResponse()
    response = theia.response.HistogramHitResponse(
        value, nBins=N_BINS, t0=T0, binSize=BIN_SIZE, normalization=NORM
    )
    replay = theia.response.HitReplay(N, response, blockSize=128)

    samples = replay.queue.view(0)
    samples["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    samples["direction"] = rng.random((N, 3))
    samples["normal"] = rng.random((N, 3))
    samples["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    samples["time"] = rng.random((N,)) * T1
    samples["contrib"] = rng.random((N,)) * 10.0

    runPipeline(replay.collectStages())

    result = response.result(0)
    # calculate expected results
    bin_edge = np.arange(N_BINS + 1) * BIN_SIZE + T0
    exp_hist, _ = np.histogram(samples["time"], bin_edge, weights=samples["contrib"])
    exp_hist *= NORM
    # check result
    # TODO: Somewhat large error. Check if this is really only due to double vs float
    assert np.allclose(result, exp_hist, rtol=1e-4)


def test_histogramEstimator(rng):
    N = 32 * 1024
    N_BINS = 50
    BIN_SIZE = 4.0 * u.ns
    T0 = 20.0 * u.ns
    T1 = T0 + N_BINS * BIN_SIZE
    NORM = 0.01

    queue = theia.response.createValueQueue(N)
    estimator = theia.response.HistogramEstimator(
        queue,
        nBins=N_BINS,
        t0=T0,
        binSize=BIN_SIZE,
        normalization=NORM,
        clearQueue=False,
    )
    updater = UpdateTensorStage(queue)
    data = as_queue(updater.buffer(0), theia.response.ValueItem)

    data.count = N
    data["value"] = rng.random(N)
    data["time"] = rng.random(N) * 300.0 * u.ns

    runPipeline([updater, estimator])

    result = estimator.result(0)
    expected = np.histogram(data["time"], N_BINS, (T0, T1), weights=data["value"])[0]
    assert np.allclose(result, expected * NORM)


def test_kernelHistogramEstimator(rng):
    N = 64 * 1024
    N_BINS = 200
    BIN_SIZE = 1.2 * u.ns
    T0 = 20.0 * u.ns
    T1 = T0 + N_BINS * BIN_SIZE
    T_PEAK = T0 + 0.3 * (T1 - T0)
    BANDWIDTH = 10.0 * u.ns
    SUPPORT = 50.0 * u.ns
    NORM = 5e-4

    value = theia.response.UniformValueResponse()
    response = theia.response.KernelHistogramHitResponse(
        value,
        nBins=N_BINS,
        t0=T0,
        binSize=BIN_SIZE,
        kernelBandwidth=BANDWIDTH,
        kernelSupport=SUPPORT,
        normalization=NORM,
    )
    replay = theia.response.HitReplay(N, response)

    samples = replay.queue.view(0)
    samples["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    samples["direction"] = rng.random((N, 3))
    samples["normal"] = rng.random((N, 3))
    samples["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    # samples["time"] = rng.random((N,)) * T1
    samples["time"] = rng.normal(T_PEAK, 10.0 * BIN_SIZE, (N,))
    # samples["contrib"] = rng.random((N,)) * 5.0 + 5.0
    samples["contrib"] = rng.random((N,)) + np.sqrt(samples["time"])

    runPipeline(replay.collectStages())

    # calculate expected result
    x = np.arange(N_BINS + 1) * BIN_SIZE + T0
    exp_result = np.zeros_like(x)
    for i in range(N):
        t = samples["time"][i]
        v = samples["contrib"][i]
        exp_result += v * stats.norm.cdf(x, loc=t, scale=BANDWIDTH)
    exp_result = np.diff(exp_result) * NORM
    # check result
    result = response.result(0)
    assert np.abs(result - exp_result).max() < 6e-4


def test_uniformResponse(rng):
    N = 32 * 1024

    queue = QueueTensor(theia.response.ValueItem, N)
    value = theia.response.UniformValueResponse()
    response = theia.response.StoreValueHitResponse(value, queue)
    replay = theia.response.HitReplay(N, response)
    fetch = RetrieveTensorStage(queue)

    hits = replay.queue.view(0)
    hits["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    dir = rng.random((N, 3))
    dir = dir / np.sqrt(np.square(dir).sum(-1))[:, None]
    hits["direction"] = dir
    nrm = -rng.random((N, 3))  # put normal in opposite octant
    nrm = nrm / np.sqrt(np.square(nrm).sum(-1))[:, None]
    hits["normal"] = nrm
    hits["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    hits["contrib"] = rng.random((N,))
    # use time to match samples
    hits["time"] = np.arange(N).astype(np.float32) * u.ns

    runPipeline([replay, fetch])

    result = as_queue(fetch.buffer(0), theia.response.ValueItem)
    time, value = result["time"], result["value"]
    # sort by time
    value = value[time.argsort()]

    # compare with expected results
    assert np.allclose(value, hits["contrib"])


@pytest.mark.parametrize("polarized", [True, False])
def test_CameraHitResponseSamples(polarized: bool):
    N = 64 * 1024

    # create sampling pipline
    rng = PhiloxRNG(key=0xC0FFEE)
    camera = PointCamera()
    photons = UniformWavelengthSource(lambdaRange=(400.0, 500.0) * u.nm)
    value = theia.response.UniformValueResponse()
    response = theia.response.SampleValueResponse(value)
    sampler = theia.response.CameraHitResponseSampler(
        N,
        photons,
        camera,
        response,
        rng=rng,
        polarized=polarized,
    )
    runPipeline(sampler.collectStages())

    # check wether we can access the queue
    queue = sampler.queue.view(0)
    result = response.result(0)
    assert queue.count == N
    assert len(result) == N
    if polarized:
        item = theia.response.PolarizedCameraHitResponseItem
    else:
        item = theia.response.CameraHitResponseItem
    exp_fields = {name for name, type in item._fields_}
    for field in exp_fields:
        assert queue[field] is not None

    # we do not check the fields for now as this would make the test a test of
    # the camera
