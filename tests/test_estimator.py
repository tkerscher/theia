import numpy as np
import theia.estimator
import theia.units as u

from hephaistos.pipeline import RetrieveTensorStage, UpdateTensorStage, runPipeline
from hephaistos.queue import QueueTensor, as_queue


def test_record(rng):
    N = 8192

    record = theia.estimator.HitRecorder()
    replay = theia.estimator.HitReplay(N, record)

    samples = replay.view(0)
    samples["position"] = (10.0 * rng.random((N, 3)) - 5.0) * u.m
    samples["direction"] = rng.random((N, 3))
    samples["normal"] = rng.random((N, 3))
    samples["wavelength"] = (rng.random((N,)) * 100.0 + 400.0) * u.nm
    samples["time"] = (rng.random((N,)) * 100.0) * u.ns
    # use contrib to encode ordering (gpu will scramble items)
    samples["contrib"] = np.arange(N).astype(np.float32)

    runPipeline([replay, record])

    results = record.view(0)
    assert results.count == N
    # restore ordering via contrib
    results = results[np.argsort(results["contrib"])]
    assert np.all(samples["position"] == results["position"])
    assert np.all(samples["direction"] == results["direction"])
    assert np.all(samples["normal"] == results["normal"])
    assert np.all(samples["wavelength"] == results["wavelength"])
    assert np.all(samples["time"] == results["time"])
    assert np.all(samples["contrib"] == results["contrib"])


def test_histogram(rng):
    N = 32 * 1024
    N_BINS = 50
    BIN_SIZE = 4.0 * u.ns
    T0 = 20.0 * u.ns
    T1 = T0 + N_BINS * BIN_SIZE
    NORM = 0.01

    queue = theia.estimator.createValueQueue(N)
    estimator = theia.estimator.HistogramEstimator(
        queue,
        nBins=N_BINS,
        t0=T0,
        binSize=BIN_SIZE,
        normalization=NORM,
        clearQueue=False,
    )
    updater = UpdateTensorStage(queue)
    data = as_queue(updater.buffer(0), theia.estimator.ValueItem)

    data.count = N
    data["value"] = rng.random(N)
    data["time"] = rng.random(N) * 300.0 * u.ns

    runPipeline([updater, estimator])

    result = estimator.result(0)
    expected = np.histogram(data["time"], N_BINS, (T0, T1), weights=data["value"])[0]
    assert np.allclose(result, expected * NORM)


def test_lambertResponse(rng):
    N = 32 * 1024

    queue = QueueTensor(theia.estimator.ValueItem, N)
    lambert = theia.estimator.LambertHitResponse(queue)
    replay = theia.estimator.HitReplay(N, lambert)
    fetch = RetrieveTensorStage(queue)

    hits = replay.view(0)
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

    result = as_queue(fetch.buffer(0), theia.estimator.ValueItem)
    time, value = result["time"], result["value"]
    # sort by time
    value = value[time.argsort()]

    # compare with expected results
    expected = -np.multiply(hits["direction"], hits["normal"]).sum(-1) * hits["contrib"]
    assert np.allclose(value, expected)
