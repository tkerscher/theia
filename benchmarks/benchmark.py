from __future__ import annotations

import importlib.util
import sys
import hephaistos as hp
import numpy as np

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from rich.console import Console
from rich.table import Table
from time import monotonic_ns

from common import Benchmark


def loadBenchmarks(console: Console, files: list[str]):
    """loads all given python files containing benchmarks"""
    for file in files:
        path = Path(file)
        moduleName = path.stem
        spec = importlib.util.spec_from_file_location(moduleName, path)
        if spec is None:
            console.print(f"[bold red] Unable to load file {file}")
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[moduleName] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)


@dataclass
class BenchmarkResult:
    """Result of a benchmark"""

    setup: float
    """Time spent on creating benchmark"""
    mean: float
    """Mean time of single benchmark runs"""
    std: float
    """Standard deviation of single benchmark runs"""


TIME_UNITS = ["ns", "µs", "ms", "s"]


def formatTime(delta: int | float):
    value = float(delta)
    unit_idx = 0
    while value > 900.0 and unit_idx < len(TIME_UNITS) - 1:
        value *= 1e-3
        unit_idx += 1
    return f"{value:.2f} {TIME_UNITS[unit_idx]}"


def printBenchmarks(console: Console):
    n = len(Benchmark._benches)
    console.print(f"[bold]Collected {n} benchmarks:\n")
    for name, bench in Benchmark._benches:
        console.print(name)


def formatResults(results: dict[str, BenchmarkResult]) -> Table:
    table = Table(expand=True)

    table.add_column("Name", justify="full")
    table.add_column("Setup")
    table.add_column("Mean")
    table.add_column("Std")

    for name, result in results.items():
        setup = formatTime(result.setup)
        mean = formatTime(result.mean)
        std = formatTime(result.std)
        table.add_row(name, setup, mean, std)

    return table


def runBenchmark(
    console: Console,
    name: str,
    bench: Benchmark,
    runs: int,
) -> BenchmarkResult:
    """runs the given benchmark `runs` times"""
    # setup benchmark
    time_a = monotonic_ns()
    bench.setup()
    time_b = monotonic_ns()
    setup = time_b - time_a
    console.print(f"{name}: Setup in {formatTime(setup)}")

    # run benchmark
    run_times: list[int] = []
    for i in range(runs):
        time_a = monotonic_ns()
        bench.run()
        time_b = monotonic_ns()
        delta = time_b - time_a
        run_times.append(delta)
        console.print(f"{name}: Run {i + 1} in {formatTime(delta)}")

    # clean up
    bench.finish()
    mean = np.mean(run_times).item()
    std = np.std(run_times).item()
    console.print(f"[bold]{name}: {formatTime(mean)} ± {formatTime(std)}")
    return BenchmarkResult(float(setup), mean, std)


def main():
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("files", nargs="+", help="One or multiple Python files containing the benchmarks")
    parser.add_argument("-l", "--list", action="store_true", help="Lists all registered benchmarks")
    parser.add_argument("-d", "--device", default=-1, help="device ID to run the benchmark on")
    parser.add_argument("-r", "--runs", default=6, help="Number of runs per benchmark")
    # TODO: add filter/regex pattern to select benchmarks
    # fmt: on
    args = parser.parse_args()
    console = Console()

    # set device before loading benchmarks
    if args.device != -1:
        hp.selectDevice(args.device)
    # load benchmarks
    loadBenchmarks(console, args.files)
    # print benchmarks
    if args.list:
        printBenchmarks(console)
        sys.exit(0)
    # only now print selected device as it forces creation of context
    console.print(f"Selected Device: {hp.getCurrentDevice()}\n")

    # run benchmarks
    results = {}
    N = len(Benchmark._benches)
    start_time = monotonic_ns()
    with console.status(f"[bold green]Running benchmark (1/{N})") as status:
        for i, (name, Bench) in enumerate(Benchmark._benches):
            status.update(f"[bold green]({i}/{N}) {name}")
            result = runBenchmark(console, name, Bench(), args.runs)
            results[name] = result
    end_time = monotonic_ns()
    run_time = end_time - start_time
    # print results
    console.print("")
    console.rule(f"{N} benchmarks in {formatTime(run_time)}")
    console.print("")
    console.print(formatResults(results))


if __name__ == "__main__":
    main()
