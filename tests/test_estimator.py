import numpy as np
import theia.estimator

from common.queue import *
from hephaistos.glsl import stackVector
from numpy.lib.recfunctions import structured_to_unstructured as s2u


def test_histogram(rng):
    N = 32 * 256
    N_BINS = 64
    BIN_SIZE = 2.0
    NORM = 0.3
    T0 = 30.0
    T1 = T0 + N_BINS * BIN_SIZE

    # create estimator
    estimator = theia.estimator.HistogramEstimator(
        None,
        N,
        nDetectors=4,
        nBins=N_BINS,
        nPhotons=N_PHOTONS,
        t0=T0,
        binSize=BIN_SIZE,
        normalization=NORM,
    )
    # upload data via cached estimator
    cache = theia.estimator.CachedEstimator(estimator, N, N_PHOTONS)

    # fill data
    samples = cache.numpy()
    samples["position"] = stackVector(
        [rng.normal(0.0, 10.0, N), rng.normal(0.0, 10.0, N), rng.normal(0.0, 10.0, N)],
        vec3,
    )
    cos_theta_dir = 2.0 * rng.random(N) - 1.0
    sin_theta_dir = np.sqrt(1.0 - cos_theta_dir**2)
    phi_dir = 2.0 * np.pi * rng.random(N)
    samples["direction"] = stackVector(
        [
            sin_theta_dir * np.cos(phi_dir),
            sin_theta_dir * np.sin(phi_dir),
            cos_theta_dir,
        ],
        vec3,
    )
    cos_theta_nrm = 2.0 * rng.random(N) - 1.0
    sin_theta_nrm = np.sqrt(1.0 - cos_theta_nrm**2)
    phi_nrm = 2.0 * np.pi * rng.random(N)
    samples["normal"] = stackVector(
        [
            sin_theta_nrm * np.cos(phi_nrm),
            sin_theta_nrm * np.sin(phi_nrm),
            cos_theta_nrm,
        ],
        vec3,
    )
    samples["hits"]["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    time = rng.random((N, N_PHOTONS)) * 200.0
    contrib = rng.random((N, N_PHOTONS)) * 5.0 - 2.0
    samples["hits"]["time"] = time
    samples["hits"]["contribution"] = contrib

    # run estimator
    cache.estimate(2)

    # calculate expected result
    cosine = -np.multiply(s2u(samples["normal"]), s2u(samples["direction"])).sum(-1)
    weights = (contrib * cosine[:, None] * NORM).flatten()
    hist_exp, _ = np.histogram(time.flatten(), N_BINS, (T0, T1), weights=weights)
    # check result
    hist = estimator.histogram(2)
    assert np.allclose(hist, hist_exp)


def test_hostEstimator(rng):
    N = 32 * 256

    # create estimator
    estimator = theia.estimator.HostEstimator(N, N_PHOTONS)
    # upload data via cached estimator
    cache = theia.estimator.CachedEstimator(estimator, N, N_PHOTONS)

    # fill data
    samples = cache.numpy()
    samples["position"] = stackVector(
        [rng.normal(0.0, 10.0, N), rng.normal(0.0, 10.0, N), rng.normal(0.0, 10.0, N)],
        vec3,
    )
    cos_theta_dir = 2.0 * rng.random(N) - 1.0
    sin_theta_dir = np.sqrt(1.0 - cos_theta_dir**2)
    phi_dir = 2.0 * np.pi * rng.random(N)
    samples["direction"] = stackVector(
        [
            sin_theta_dir * np.cos(phi_dir),
            sin_theta_dir * np.sin(phi_dir),
            cos_theta_dir,
        ],
        vec3,
    )
    cos_theta_nrm = 2.0 * rng.random(N) - 1.0
    sin_theta_nrm = np.sqrt(1.0 - cos_theta_nrm**2)
    phi_nrm = 2.0 * np.pi * rng.random(N)
    samples["normal"] = stackVector(
        [
            sin_theta_nrm * np.cos(phi_nrm),
            sin_theta_nrm * np.sin(phi_nrm),
            cos_theta_nrm,
        ],
        vec3,
    )
    samples["hits"]["wavelength"] = rng.random((N, N_PHOTONS)) * 600.0 + 200.0
    samples["hits"]["time"] = rng.random((N, N_PHOTONS)) * 100.0
    samples["hits"]["contribution"] = rng.random((N, N_PHOTONS)) * 50.0

    # run estimator
    cache.estimate(4)  # id does not really matter...

    # check result
    result = estimator.numpy()
    assert np.all(samples == result)
