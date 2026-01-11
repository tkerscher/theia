from __future__ import annotations

import hephaistos as hp
from hephaistos import Program
from hephaistos.pipeline import SourceCodeMixin

from ctypes import Structure, c_float, c_int32, c_uint32, c_uint64
from ctypes import addressof, memset, sizeof
from numpy.ctypeslib import as_array

from theia.util import ShaderLoader, createPreamble

from enum import IntEnum
from numpy.typing import NDArray


__all__ = [
    "EmptyEventCallback",
    "EventResultCode",
    "EventStatisticCallback",
    "TraceEventCallback",
    "TrackRecordCallback",
]


def __dir__():
    return __all__


class TraceEventCallback(SourceCodeMixin):
    """
    Base class for callbacks a tracer will call on tracing events
    (scatter, hit, detected, lost, decayed) of the form

    void onEvent(const Ray ray, uint type, uint idx, uint i)
    """

    name = "Trace Event Callback"

    def __init__(
        self,
        *,
        params: dict[str, type[Structure]] = {},
        extra: set[str] = set(),
    ) -> None:
        super().__init__(params, extra)


class EmptyEventCallback(TraceEventCallback):
    """Callback ignoring all events"""

    def __init__(self) -> None:
        super().__init__()

    sourceCode = ShaderLoader("callback/empty.glsl")


class EventStatisticCallback(TraceEventCallback):
    """
    Callback recording statistics about the event types, i.e. increments a
    global counter for each event type. Current statistic are read directly from
    the device, but may have a bad impact on currently running tracing.
    """

    name = "Event Statistic"

    class Statistic(Structure):
        _fields_ = [
            ("created", c_uint64),
            ("scattered", c_uint64),
            ("hit", c_uint64),
            ("detected", c_uint64),
            ("volume", c_uint64),
            ("lost", c_uint64),
            ("decayed", c_uint64),
            ("absorbed", c_uint64),
            ("missed", c_uint64),
            ("maxIter", c_uint64),
            ("error", c_uint64),
            ("mismatch", c_uint64),
        ]

    def __init__(self) -> None:
        super().__init__()
        # allocate mapped tensor for statistics
        self._tensor = hp.Tensor(sizeof(self.Statistic), mapped=True)
        if not self._tensor.isMapped:
            raise RuntimeError("Could not create mapped tensor")
        self._stat = self.Statistic.from_address(self._tensor.memory)
        self.reset()

    # sourceCode via descriptor
    sourceCode = ShaderLoader("callback/stat.glsl")

    @property
    def absorbed(self) -> int:
        """Number of absorbed rays"""
        return self._stat.absorbed

    @property
    def created(self) -> int:
        """Number of rays created"""
        return self._stat.created

    @property
    def hit(self) -> int:
        """Number of hit events"""
        return self._stat.hit

    @property
    def detected(self) -> int:
        """Number of rays detected"""
        return self._stat.detected

    @property
    def maxIter(self) -> int:
        """Number of rays that reached the maximum path length"""
        return self._stat.maxIter

    @property
    def mismatch(self) -> int:
        """Number of rays aborted due to hitting unexpected media"""
        return self._stat.mismatch

    @property
    def missed(self) -> int:
        """Number of rays that missed the target"""
        return self._stat.missed

    @property
    def scattered(self) -> int:
        """Number of scatter events"""
        return self._stat.scattered

    @property
    def lost(self) -> int:
        """Number of rays that left the tracing boundary box"""
        return self._stat.lost

    @property
    def decayed(self) -> int:
        """Number of rays exceeding the max time"""
        return self._stat.decayed

    @property
    def volume(self) -> int:
        """Number of volume boundary hits"""
        return self._stat.volume

    @property
    def error(self) -> int:
        """Number of traces aborted due to an error"""
        return self._stat.error

    def reset(self) -> None:
        """Resets the statistic"""
        memset(addressof(self._stat), 0, sizeof(self.Statistic))

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(Statistics=self._tensor)

    def __str__(self) -> str:
        fields = sorted([f[0] for f in self.Statistic._fields_])
        just = max(len(f) for f in fields)
        lines = [f"{f.rjust(just)}: {getattr(self, f):,}" for f in fields]
        return "\n".join(lines)


class TrackRecordCallback(TraceEventCallback):
    """
    Callback recording position of events allowing to later reconstruct complete
    path. Only considers first sample for time coordinate.

    Parameters
    ----------
    capacity: int
        Number of rays that can be recorded
    length: int
        Max number of points per track
    polarized: bool, default=False
        Whether to save polarization state. Save unpolarized state and zero
        vector as reference frame if ray contains no polarization state.
    retrieve: bool, default=True
        Whether to retrieve the tracks from the device
    """

    name = "Track Recorder"

    def __init__(
        self,
        capacity: int,
        length: int,
        *,
        polarized: bool = False,
        retrieve: bool = True,
    ) -> None:
        super().__init__()
        # save params
        self._capacity = capacity
        self._length = length
        self._retrieve = retrieve
        self._polarized = polarized

        # allocate memory
        self._cols = 11 if polarized else 4
        words = capacity * length * self._cols + capacity * 2  # track + length + codes
        self._tensor = hp.Tensor(words * 4)
        self._buffer = [hp.Buffer(words * 4) for _ in range(2)]

    @property
    def capacity(self) -> int:
        """Number of rays that can be recorded"""
        return self._capacity

    @property
    def length(self) -> int:
        """Max number of points per track"""
        return self._length

    @property
    def polarized(self) -> bool:
        """Whether polarization state is recorded"""
        return self._polarized

    @property
    def retrieve(self) -> bool:
        """Wether to retrieve the tracks from the device"""
        return self._retrieve

    # sourceCode via descriptor
    _sourceCode = ShaderLoader("callback/track.glsl")

    @property
    def sourceCode(self) -> str:
        preamble = createPreamble(
            TRACK_COUNT=self.capacity,
            TRACK_LENGTH=self.length,
            TRACK_POLARIZED=self.polarized,
        )
        return preamble + self._sourceCode

    @property
    def tensor(self) -> hp.Tensor:
        """Tensor containing the tracks"""
        return self._tensor

    def result(self, i: int) -> tuple[NDArray, NDArray, NDArray]:
        """
        Returns the recorded tracks saved using the i-th pipeline configuration.
        First tuple are the tracks, the second one the length of each track, and
        the last one is the last recorded result code. Positions after each
        track length may contain garbage data.

        Returns
        -------
        tracks: NDArray
            Recorded tracks stored in a numpy array of shape (track,pos,4)
        lengths: NDArray
            Length of each recorded track stored in a numpy array of shape (track,)
        codes: NDArray
            Last recorded result code per track
        """
        # fetch each data structures
        adr = self._buffer[i].address
        pLengths = (c_uint32 * self.capacity).from_address(adr)
        pCodes = (c_int32 * self.capacity).from_address(adr + 4 * self.capacity)
        n = self.length * self.capacity * self._cols
        pTracks = (c_float * n).from_address(adr + 8 * self.capacity)
        # construct numpy array
        lengths = as_array(pLengths)
        codes = as_array(pCodes)
        tracks = as_array(pTracks).reshape((self._cols, self.length, self.capacity))
        # change from (coords, ray, event) -> (ray, event, coords)
        tracks = tracks.transpose(2, 1, 0)
        return tracks, lengths, codes

    def bindParams(self, program: Program, i: int) -> None:
        super().bindParams(program, i)
        program.bindParams(TrackBuffer=self._tensor)

    def run(self, i: int) -> list[hp.Command]:
        if self.retrieve:
            return [hp.retrieveTensor(self._tensor, self._buffer[i])]
        else:
            return []


class EventResultCode(IntEnum):
    """
    Enumeration of result codes the tracing algorithm can encounter and pass
    on to the event callback. Negative codes indicate the tracer to stop.
    """

    SUCCESS = 0
    """Operation successful without further information"""
    RAY_CREATED = 1
    """Ray was sampled"""
    RAY_SCATTERED = 2
    """Ray scattered"""
    RAY_HIT = 3
    """Ray hit a geometry other than target"""
    RAY_DETECTED = 4
    """Ray hit target"""
    VOLUME_HIT = 5
    """Ray hit volume border"""
    RAY_LOST = -1
    """Ray left tracing boundary box"""
    RAY_DECAYED = -2
    """Ray reached max life time"""
    RAY_ABSORBED = -3
    """Ray hit absorber"""
    RAY_MISSED = -4
    """Ray meant to create a hit missed target"""
    MAX_ITER = -5
    """Tracing reached max iteration"""
    ERROR_CODE_MAX_VALUE = -10
    """Max value for an error code"""
    ERROR_UNKNOWN = -10
    """Tracer encountered an unexpected error indicating a bug in the tracer"""
    ERROR_MEDIA_MISMATCH = -11
    """Tracer encountered unexpected media"""
    ERROR_TRACE_ABORT = -12
    """Tracer reached a state it can't proceed from"""
