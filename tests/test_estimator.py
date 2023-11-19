import numpy as np
import theia.estimator

from common.queue import *
from hephaistos.pipeline import RetrieveTensorStage, runPipeline


def test_histogram(rng):
    N = 32 * 256
    N_BINS = 64
    BIN_SIZE = 2.0
    NORM = 0.3
    T0 = 30.0
    T1 = T0 + N_BINS * BIN_SIZE

    # upload data via cached estimator
    cache = theia.estimator.CachedSampleSource(N, N_PHOTONS)
    # create estimator
    estimator = theia.estimator.HistogramEstimator(
        N,
        cache.queue,
        None,
        nBins=N_BINS,
        nPhotons=N_PHOTONS,
        t0=T0,
        binSize=BIN_SIZE,
        normalization=NORM,
    )

    # fill data
    samples = cache.view(0)
    samples["position"] = np.stack(
        [rng.normal(0.0, 10.0, N), rng.normal(0.0, 10.0, N), rng.normal(0.0, 10.0, N)],
        axis=-1,
    )
    cos_theta_dir = 2.0 * rng.random(N) - 1.0
    sin_theta_dir = np.sqrt(1.0 - cos_theta_dir**2)
    phi_dir = 2.0 * np.pi * rng.random(N)
    samples["direction"] = np.stack(
        [
            sin_theta_dir * np.cos(phi_dir),
            sin_theta_dir * np.sin(phi_dir),
            cos_theta_dir,
        ],
        axis=-1,
    )
    cos_theta_nrm = 2.0 * rng.random(N) - 1.0
    sin_theta_nrm = np.sqrt(1.0 - cos_theta_nrm**2)
    phi_nrm = 2.0 * np.pi * rng.random(N)
    samples["normal"] = np.stack(
        [
            sin_theta_nrm * np.cos(phi_nrm),
            sin_theta_nrm * np.sin(phi_nrm),
            cos_theta_nrm,
        ],
        axis=-1,
    )
    samples["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    time = rng.random((N, N_PHOTONS)) * 200.0
    contrib = rng.random((N, N_PHOTONS)) * 5.0 - 2.0
    samples["time"] = time
    samples["contribution"] = contrib

    # run estimator
    retriever = RetrieveTensorStage(estimator.histogram)
    runPipeline([cache, estimator, retriever])

    # calculate expected result
    cosine = -np.multiply(samples["normal"], samples["direction"]).sum(-1)
    weights = (contrib * cosine[:, None] * NORM).flatten()
    hist_exp, _ = np.histogram(time.flatten(), N_BINS, (T0, T1), weights=weights)
    # check result
    hist = retriever.view(0)
    assert np.allclose(hist, hist_exp)


def test_hostEstimator(rng):
    N = 32 * 256

    # create pipeline
    cache = theia.estimator.CachedSampleSource(N, N_PHOTONS)
    estimator = theia.estimator.HostEstimator(N, N_PHOTONS, cache.queue)

    # fill data
    samples = cache.view(0)
    samples["position"] = np.stack(
        [rng.normal(0.0, 10.0, N), rng.normal(0.0, 10.0, N), rng.normal(0.0, 10.0, N)],
        axis=-1,
    )
    cos_theta_dir = 2.0 * rng.random(N) - 1.0
    sin_theta_dir = np.sqrt(1.0 - cos_theta_dir**2)
    phi_dir = 2.0 * np.pi * rng.random(N)
    samples["direction"] = np.stack(
        [
            sin_theta_dir * np.cos(phi_dir),
            sin_theta_dir * np.sin(phi_dir),
            cos_theta_dir,
        ],
        axis=-1,
    )
    cos_theta_nrm = 2.0 * rng.random(N) - 1.0
    sin_theta_nrm = np.sqrt(1.0 - cos_theta_nrm**2)
    phi_nrm = 2.0 * np.pi * rng.random(N)
    samples["normal"] = np.stack(
        [
            sin_theta_nrm * np.cos(phi_nrm),
            sin_theta_nrm * np.sin(phi_nrm),
            cos_theta_nrm,
        ],
        axis=-1,
    )
    samples["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    samples["time"] = rng.random((N, N_PHOTONS)) * 100.0
    samples["contribution"] = rng.random((N, N_PHOTONS)) * 50.0

    # run estimator
    runPipeline([cache, estimator])

    # check result
    result = estimator.view(0)
    for field in samples.fields:
        assert np.all(samples[field] == result[field])
