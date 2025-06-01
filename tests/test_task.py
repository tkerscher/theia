import pytest

import numpy as np
import hephaistos as hp
import hephaistos.pipeline as pl
import theia.units as u
import theia.task

from theia.light import SphericalLightSource, UniformWavelengthSource
from theia.material import MaterialStore
from theia.random import PhiloxRNG
from theia.response import HistogramHitResponse, UniformValueResponse
from theia.target import SphereTarget
from theia.trace import VolumeForwardTracer
from theia.testing import WaterTestModel


def test_ConvergeHistogramTask():
    # scene settings
    target_pos = (0.0, 0.0, 0.0) * u.m
    radius = 5.0 * u.m
    # light settings
    light_pos = (-6.0, 0.0, 0.0) * u.m
    budget = 1e5
    lambda_range = (400.0, 700.0) * u.nm
    light_t0 = 20.0 * u.ns
    # tracer settings
    batch_size = 256 * 1024
    max_length = 10
    scatter_coef = 0.01
    maxTime = 600.0 * u.ns
    # binning settings
    bin_t0 = 0.0
    bin_size = 5.0 * u.ns
    n_bins = 100
    # task settings
    initBatches = 10
    newBatches = 5
    max_batches = 400
    atol = 0.1
    rtol = 5e-4

    # create medium
    medium = WaterTestModel().createMedium()
    store = MaterialStore([], media=[medium])
    # create pipeline
    rng = PhiloxRNG(key=0xC0FFEE)
    photons = UniformWavelengthSource(lambdaRange=lambda_range)
    light = SphericalLightSource()  # we will pass light params through the task
    target = SphereTarget(position=target_pos, radius=radius)
    response = HistogramHitResponse(
        UniformValueResponse(), nBins=n_bins, binSize=bin_size, t0=bin_t0
    )
    tracer = VolumeForwardTracer(
        batch_size,
        light,
        target,
        photons,
        response,
        rng,
        medium=store.media["water"],
        maxTime=maxTime,
        nScattering=max_length,
        scatterCoefficient=scatter_coef,
    )
    rng.autoAdvance = tracer.nRNGSamples
    # create pipeline + scheduler
    pipeline = pl.Pipeline(tracer.collectStages())
    scheduler = pl.DynamicTaskScheduler(pipeline)

    # create params dict
    params = {
        "lightSource__position": light_pos,
        "lightSource__budget": budget,
        "lightSource__timeRange": (light_t0, light_t0),
    }
    # create task
    task = theia.task.ConvergeHistogramTask(
        response,
        params,
        initialBatchCount=initBatches,
        extraBatchCount=newBatches,
        maxBatchCount=max_batches,
        atol=atol,
        rtol=rtol,
    )
    # schedule task
    scheduler.scheduleTask(task)
    scheduler.waitAll()

    # check result
    assert task.converged
    assert task.totalBatches <= max_batches
    assert task.totalBatches > initBatches  # to ensure we test the reissuing of batches
    error_thres = budget * rtol + atol  # upper limit
    assert task.error > 0.0 and task.error <= error_thres
    # to ensure we didn't "converge" to zero -> compare with the last batch
    lastBatch = response.result(0)
    assert lastBatch is not None
    # just a quick n dirty check that the result has the same shape as the batches
    assert np.abs(task.result - lastBatch).max() < 1e-3 * budget
