from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type


class Benchmark(ABC):
    """Base class for benchmarks"""

    _benches: list[tuple[str, Type[Benchmark]]] = []

    def __init_subclass__(cls, name: str = "", skip: bool = False, **kwargs) -> None:
        # register new benchmark
        super().__init_subclass__()
        if not skip:
            if not name:
                raise ValueError("Benchmarks must be named!")
            Benchmark._benches.append((name, cls))

    def setup(self) -> None:
        """Called one before all iterations"""
        pass

    @abstractmethod
    def run(self) -> None:
        """Called multiple times, once per iteration"""
        pass

    def finish(self) -> None:
        """Called once after all iterations"""
        pass
