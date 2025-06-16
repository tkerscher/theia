from __future__ import annotations

import warnings

import numpy as np
from hephaistos.pipeline import DynamicTask, PipelineParams

from theia.response import HistogramHitResponse, KernelHistogramHitResponse

from numpy.typing import NDArray
from typing import Callable

__all__ = [
    "ConvergeHistogramTask",
]


def __dir__():
    return __all__


class ConvergeHistogramTask(DynamicTask):
    """
    Dynamic task issuing and combining new batches of a histogram until they
    converge, using the standard error of the mean on the sum of bins as an
    estimate of the convergence process. It considers the histogram to be
    converged, if this error drops below a pre-defined threshold.

    Parameters
    ----------
    response: HistogramHitResponse | KernelHistogramResponse
        Response pipeline stage producing the histograms
    params: PipelineParams
        Pipeline parameters to use for issuing new batches of histograms.
    pipeline: str | None, default=None
        Name of the pipeline to use, or `None` if the corresponding scheduler
        only has a single pipeline.
    initialBatchCount: int, default=4
        Number of batches to issue at the beginning. Must be at least 2,
        otherwise an error cannot be estimated.
    extraBatchCount: int, default=2
        Number of batches to issue when the error threshold has not yet been
        reached and there are no more batches in flight.
    maxBatchCount: int, default=50
        Maximum number of batches to issue. If afterwards the error threshold
        is still not reached, the task is considered to have failed. You can
        check this via the `converged` property.
    atol: float, default=0.1
        Absolute error tolerance. See Notes
    rtol: float, default=5e-5
        Relative error tolerance. See Notes.
    finishedCallback: (ConvergeHistogram) -> None | None, default=None
        Optional callback called once the task finishes. Note that it will be
        called from a different thread.

    Notes
    -----
    The error threshold to determine convergence is calculated like this:

    thres = atol + rtol * sum bins
    """

    def __init__(
        self,
        response: HistogramHitResponse | KernelHistogramHitResponse,
        params: PipelineParams,
        pipeline: str | None = None,
        *,
        initialBatchCount: int = 4,
        extraBatchCount: int = 2,
        maxBatchCount: int = 50,
        # targetError: float = 0.1,
        atol: float = 0.1,
        rtol: float = 5e-5,
        finishedCallback: Callable[[ConvergeHistogramTask], None] | None = None,
    ) -> None:
        super().__init__(params, pipeline, initialBatchCount=initialBatchCount)
        # check arguments
        if not response.retrieve:
            raise ValueError("Task requires response to retrieve results!")
        if initialBatchCount < 2:
            raise ValueError("initialBatchCount must be at least 2!")
        if extraBatchCount < 1:
            raise ValueError("extraBatchCount must be at least1!")
        # save params
        self._response = response
        self._extraCount = extraBatchCount
        self._maxBatchCount = maxBatchCount
        self._rtol = rtol
        self._atol = atol
        self._totalBatches = 0
        self._converged = False
        self._callback = finishedCallback

        # allocate variables for result
        self._result = np.zeros(response.nBins)
        self._totalPhotonsMean = 0.0
        self._sumSquareErr = 0.0

    @property
    def converged(self) -> bool:
        """True if the result has converged"""
        return self._converged

    @property
    def error(self) -> float:
        """
        Returns the standard error of the mean summed over all bins.
        Note that this includes a correction factor for small sample size,
        making the estimate a bit pessimistic.
        """
        # from Welford: var ~ sumSquare / (n - 1)
        # normally, std = sqrt(var)
        # but here we use a (sloppy) correction for small sample size:
        #  std = sqrt(sumSquare / (n - 1.5))
        # This is an approximation of the c4 corrections.
        # note, that this is more pessimistic than the correct correction,
        # which is fine if we are only interested in an "upper bound"
        # finally, we divide by sqrt(n) to get the standard error
        n = self.totalBatches
        return np.sqrt(self._sumSquareErr / (n - 1.5)) / np.sqrt(n)

    @property
    def atol(self) -> float:
        """Absolute error threshold"""
        return self._atol

    @property
    def batchCountLimit(self) -> int:
        """Maximum allowed amount of batches"""
        return self._maxBatchCount

    @property
    def response(self) -> HistogramHitResponse | KernelHistogramHitResponse:
        """Underlying response producing histograms"""
        return self._response

    @property
    def result(self) -> NDArray[np.float64]:
        """Final histogram. Check `converged` to see whether this result is accurate."""
        return self._result

    @property
    def rtol(self) -> float:
        """Relative error threshold"""
        return self._rtol

    @property
    def totalBatches(self) -> int:
        """Total amount of batches used so far"""
        return self._totalBatches

    def onTaskFinished(self) -> None:
        if self._callback is not None:
            self._callback(self)

    def processBatch(self, config: int) -> int:
        # update counter
        self._totalBatches += 1
        # fetch result
        result = self._response.result(config)
        assert result is not None

        # use Welford's algorithm to update result and error estimate
        self._result += (result - self._result) / self._totalBatches
        mean_i = result.sum()
        oldMean = self._totalPhotonsMean
        self._totalPhotonsMean += (mean_i - oldMean) / self._totalBatches
        self._sumSquareErr += (mean_i - oldMean) * (mean_i - self._totalPhotonsMean)

        # wait for any batches still in flight before issuing new ones
        if self.batchesRemaining > 0:
            return 0

        # proccessed all batches issued so far
        # -> check error to see if we need more
        thres = self.atol + self.rtol * self._totalPhotonsMean
        if self.error <= thres:
            # converged!
            self._converged = True
            # no more batches
            return 0
        else:
            # issue more batches but no more than maxBatchCount
            # if we reached the maximum return 0 to prevent infinite loop
            remaining = max(self.batchCountLimit - self.totalBatches, 0)
            # issue no more than extraCount batches at once
            n = min(remaining, self._extraCount)
            # issue warning to notify about failed convergence
            if n == 0:
                warnings.warn(
                    f"Failed to converge histogram (error: {self.error:.3e})!\n"
                    f"Pipeline: {self.pipeline}\n"
                    f"Params: {self.parameters}"
                )
            return n
