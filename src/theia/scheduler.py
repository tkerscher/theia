from __future__ import annotations
import hephaistos as hp
import numpy as np

from abc import ABC, abstractmethod
from ctypes import Structure, addressof, memmove, sizeof, c_uint8
from itertools import chain
from queue import Queue
from threading import Thread

from numpy.typing import DTypeLike, NDArray
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union


class PipelineStage(ABC):
    """Base class for pipeline stages used in schedulers"""

    def __init__(self, nConfigs: int = 2) -> None:
        self._nConfigs = nConfigs

    @property
    def nConfigs(self) -> int:
        """Number of pipeline configurations"""
        return self._nConfigs

    @abstractmethod
    def run(self, i: int) -> List[hp.Command]:
        """
        Creates a list of commands responsible for running this pipeline stage
        using the i-th configuration.
        """
        pass

    def update(self, i: int) -> None:
        """
        Updates the i-th configuration of this pipeline stage using the local
        state, thus applying recent changes.
        """
        pass


def runPipelineStage(stage: PipelineStage, i: int = 0, *, update: bool = True) -> None:
    """
    Helper function for running a single pipeline stage isolated using the i-th
    configuration. Blocks the calling code until the stage finishes.

    Parameters
    ----------
    i: int, default=0
        Which configuration to run
    update: bool, default=True
        Whether to update the i-th configuration before the run
    """
    if update:
        stage.update(i)
    hp.beginSequence().AndList(stage.run(i)).Submit().wait()


def runPipeline(
    stages: List[PipelineStage], i: int = 0, *, update: bool = True
) -> None:
    """
    Helper function for running an intermediate pipeline and wait for it to finish.

    Parameters
    ----------
    i: int, default=0
        Which configuration to run
    update: bool, default=True
        Whether to update the i-th configuration before the run
    """
    if update:
        for stage in stages:
            stage.update(i)
    hp.beginSequence().AndList(
        list(chain.from_iterable(stage.run(i) for stage in stages))
    ).Submit().wait()


class CachedPipelineStage(PipelineStage):
    """
    Base class for configurable pipeline stages used in schedulers.
    Contains code for cached updates on configurations (e.g. UBOs).
    """

    def __init__(self, params: Dict[str, Type[Structure]], nConfigs: int = 2) -> None:
        super().__init__(nConfigs)
        # create a local config
        self._cache = {name: c() for name, c in params.items()}
        # create mapped tensors containing configs
        self._configs = [
            {name: hp.StructureTensor(c, True) for name, c in params.items()}
            for _ in range(nConfigs)
        ]
        # check tensors are mapped
        if any(not t.isMapped for c in self._configs for t in c.values()):
            raise RuntimeError("Cannot create mapped tensors!")

    def _getParam(self, param: str, prop: str) -> Any:
        """Returns the cached value of the given property"""
        return getattr(self._cache[param], prop)

    def _setParam(self, param: str, prop: str, value: Any) -> None:
        """Sets the property of the given parameter set"""
        setattr(self._cache[param], prop, value)

    def _bindConfigs(self, program: hp.Program, i: int) -> None:
        """Binds the i-th configuration to the program"""
        program.bindParams(**self._configs[i])

    def update(self, i: int) -> None:
        for name, config in self._cache.items():
            memmove(self._configs[i][name].memory, addressof(config), sizeof(config))


class RetrieveTensorStage(PipelineStage):
    """Helper stage for copying a tensor to a buffer"""

    def __init__(self, src: hp.Tensor, nBuffers: int = 2) -> None:
        super().__init__(nBuffers)
        self._src = src
        # allocate buffers
        self._buffers = [hp.RawBuffer(src.size_bytes) for _ in range(nBuffers)]

    @property
    def src(self) -> hp.Tensor:
        """Source tensor"""
        return self._src

    def address(self, i: int = 0) -> int:
        """Returns the memory address of the i-th buffer"""
        return self.buffer(i).address

    def buffer(self, i: int = 0) -> hp.RawBuffer:
        """Returns the i-th buffer"""
        return self._buffers[i]

    def view(self, dtype: DTypeLike, i: int = 0) -> NDArray:
        """Interprets the retrieved tensor as array of given type"""
        data = (c_uint8 * self.src.size_bytes).from_address(self.address(i))
        return np.frombuffer(data, dtype)

    def run(self, i: int) -> List[hp.Command]:
        return [hp.retrieveTensor(self.src, self._buffers[i])]


class Pipeline:
    """
    Pipelines contain a sequence of named pipeline stages, manages their states
    as well as providing subroutines to be used for scheduling tasks to be fed
    into the pipeline.
    If no name is provided, a stage will be named "stage{i}" where i is the
    position of the stage within the sequence.

    Changes to the stages can be issued via a dict assigning a parameter path
    "stage_name.parameter" to change the parameter in a specific stage or
    just "parameter" to apply the change to all stages with that property.

    Parameters
    ----------
    stages: (PipelineStage | (name, PipelineStage))[]
        sequence of stages the pipeline consists of. Each element can optionally
        specify a name used for updating properties. If no name is provided
        (i.e. not a tuple), it get the name "stage{i}" where i is the stage's
        position in the pipeline
    nConfigs: int | None, default=None
        Optionally specify the number of parallel configuration the pipeline
        should manage. If None, chooses the minimum of all stages.
    """

    def __init__(
        self,
        stages: List[Union[PipelineStage, Tuple[str, PipelineStage]]],
        *,
        nConfigs: Optional[int] = None,
    ) -> None:
        # create stage list
        self._stages = [
            s if isinstance(s, tuple) else (f"stage{i}", s)
            for i, s in enumerate(stages)
        ]
        # create stage dict
        self._stageDict = {name: stage for name, stage in self._stages}

        # nConfigs is min value of stages' nConfigs
        self._nConfigs = min(stage.nConfigs for _, stage in self._stages)
        if nConfigs is not None:
            if self._nConfigs < nConfigs:
                raise RuntimeError(
                    "Not all stages support as many configurations as requested!"
                )
            else:
                self._nConfigs = nConfigs

        # create subroutines (can't rely on dict ordering)
        self._subroutines = [
            hp.createSubroutine(
                list(
                    chain.from_iterable(
                        [(s[1] if isinstance(s, tuple) else s).run(i) for s in stages]
                    )
                ),
                simultaneous=True,
            )
            for i in range(self._nConfigs)
        ]

    @property
    def nConfigs(self) -> int:
        """Number of parallel configurations supported"""
        return self._nConfigs

    @property
    def stages(self) -> List[Tuple[str, PipelineStage]]:
        """
        Sequence of named pipeline stages.
        Altering this list results in undefined behavior
        """
        return self._stages

    def getSubroutine(self, i: int) -> hp.Subroutine:
        """
        Returns the subroutine responsible for running the pipeline using
        the i-th configuration
        """
        return self._subroutines[i]

    def apply(self, **params) -> None:
        """
        Applies the given parameter changes to the corresponding stages.

        Changes to the stages can be issued via a dict assigning a parameter path
        "stage_name.parameter" to change the parameter in a specific stage or
        just "parameter" to apply the change to all stages with that property.
        """
        for path, value in params.items():
            # apply to all or specific stage?
            if "." in path:
                # specific stage
                stage, path = path.split(".", 1)
                if stage in self._stageDict and hasattr(self._stageDict[stage], path):
                    setattr(self._stageDict[stage], path, value)
            else:
                # apply to all stages
                for _, stage in self._stages:
                    if hasattr(stage, path):
                        setattr(stage, path, value)

    def update(self, i: int) -> None:
        """
        Updates the i-th configuration of all stages using their current state
        """
        for _, stage in self._stages:
            stage.update(i)

    def runAsync(self, i: int) -> hp.Submission:
        """
        Runs the i-th configuration of the pipeline and returns a Submission
        which can be waited on.
        """
        return hp.beginSequence().And(self.getSubroutine(i)).Submit()

    def run(self, i: int) -> None:
        """
        Runs the i-th configuration of the pipeline and blocks the calling code
        until it finishes
        """
        self.runAsync(i).wait()


class PipelineScheduler:
    """
    Schedules tasks into a pipeline and orchestrates the processing of the
    pipeline results. Bundles tasks into a single batch submission making it
    more efficient than repeatedly calling run(Async) on the pipeline.

    Scheduling tasks happens completely in the background using multithreading
    allowing the calling code to do other work, e.g. preparing the next
    submission to be issued to the scheduler, but also provides function to wait
    on the completion of a certain or all tasks.

    New tasks can be issued while previous ones are still processed.

    The scheduler requires exclusive access to two configurations, but may
    otherwise share the pipeline.

    Parameters
    ----------
    pipeline: Pipeline
        Pipeline onto which to schedule tasks
    baseConfig: int, default=0
        Index of the config to use for scheduling. Scheduler will use
        baseConfig and baseConfig+1
    queueSize: int, default=0
        Size of the task queue. Size of 0 means infinite.
    processFn: Callable( (config: int) -> None ), default=None
        Function to be called after each finished task on the pipeline.
        The scheduler ensures tasks using the same configuration to wait until
        processing finished. Will run in its own thread.
    daemon: bool, default=True
        Whether the worker threads should be daemons. If False, the worker
        threads will still process issued work even after the main thread (i.e.
        the calling code) finishes. Might be useful for scripts.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        baseConfig: int = 0,
        queueSize: int = 0,
        processFn: Optional[Callable[[int], None]],
        daemon: bool = True,
    ) -> None:
        # save params
        self._pipeline = pipeline
        self._baseConfig = baseConfig
        self._queueSize = queueSize
        self._config = 0
        self._totalTasks = 0
        self._daemon = daemon
        # create pipeline timeline
        self._pipelineTimeline = hp.Timeline()
        # create update timeline and queue
        self._updateTimeline = hp.Timeline()
        self._updateQueue = Queue(queueSize)
        self._updateThread = None  # lazily created
        self._updateCounter = 0
        # create process stuff if needed
        self._processFn = processFn
        self._processTimeline = hp.Timeline() if processFn is not None else None
        self._processThread = None
        self._processCounter = 0

    @property
    def pipeline(self) -> Pipeline:
        """Underlying pipeline tasks are scheduled onto"""
        return self._pipeline

    @property
    def baseConfig(self) -> int:
        """Index of the pipeline's config to be exclusively used"""
        return self._baseConfig

    @property
    def queueSize(self) -> int:
        """Capacity of the internal queue for scheduling tasks"""
        return self._queueSize

    @property
    def totalTasks(self) -> int:
        """Total number of tasks scheduled"""
        return self._totalTasks

    @property
    def tasksScheduled(self) -> int:
        """Approximate number of tasks scheduled"""
        return self._updateQueue.qsize()

    @property
    def tasksFinished(self) -> int:
        """Returns the number of tasks processed by the pipeline"""
        return self._pipelineTimeline.value

    def schedule(
        self, tasks: Iterable[Dict[str, Any]], *, timeout: Optional[float] = None
    ) -> int:
        """
        Schedules the list of tasks to be processed by the pipeline after
        previous submissions have been finished.

        Parameters
        ----------
        tasks: Iterable[Dict[str, Any]]
            Sequence of tasks to schedule onto the pipeline
        timeout: float | None, default=None
            Timeout in seconds for waiting on free space in the queue.
            If None, waits indefinitely.

        Returns
        -------
        nSubmitted: int
            Number of tasks actually submitted
        """
        # put tasks onto queue
        n = 0
        builder = hp.beginSequence(self._pipelineTimeline, self._totalTasks)
        for task in tasks:
            # try to enlist tasks
            try:
                self._updateQueue.put(task, timeout=timeout)
            except:
                break

            # wait on previous task
            builder.waitFor(self._updateTimeline, self._totalTasks + 1)
            if self._processFn is not None and self._totalTasks > 2:
                # double buffered -> wait for the processing of the
                # task two earlier to finish (need to check if that one exists)
                builder.waitFor(self._processTimeline, self._totalTasks - 2)
            # add submission
            i = self._config + self._baseConfig
            self.And(self._pipeline.getSubroutine(i))

            # update counters
            n += 1
            self._totalTasks += 1
            self._config = (self._config + 1) % 2  # toggle between [0,1]
        # Submit work
        submission = builder.Submit()
        # just to ensure we're not breaking things with later changes
        assert submission.forgettable

        # check if we have to start update thread
        if self._updateThread is None or not self._updateThread.is_alive():
            # start update thread
            self._updateThread = Thread(target=self._update, daemon=self._daemon)
            self._updateThread.start()
        # check if we have to start process thread
        if (
            self._processFn is not None
            and self._processThread is None
            or self._processThread.is_alive()
        ):
            # start process thread
            self._processThread = Thread(target=self._process, daemon=self._daemon)
            self._processThread.start()

        # return number of added tasks
        return n

    def wait(self) -> None:
        """Waits on the pipeline to finish all tasks"""
        self._pipelineTimeline.wait(self._totalTasks)

    def _update(self) -> None:
        """Internal update thread body"""
        # processing loop
        while True:
            # fetch next task
            task: Optional[Dict[str, Any]] = None
            try:
                task = self._updateQueue.get(timeout=0.05)  # 50 ms
            except:
                # no more tasks -> end thread
                break
            # update pipeline
            config = (self._updateCounter % 2) + self._baseConfig
            self._pipeline.apply(**task)
            # wait on previous task to finish so it's safe to update
            if self._updateCounter >= 2:
                self._pipelineTimeline.wait(self._updateCounter - 1)
            # update config
            self._pipeline.update(config)
            # advance timeline
            self._updateCounter += 1
            self._updateTimeline.value = self._updateCounter

    def _process(self) -> None:
        """Internal process thread body"""
        # processing loop
        while self._processCounter < self._totalTasks:
            config = self._processCounter % 2 + self._baseConfig
            self._processCounter += 1
            # wait on task to finish
            self._pipelineTimeline.wait(self._processCounter)
            # process
            self._processFn(config)
            # advance timeline
            self._processTimeline.value = self._processCounter
