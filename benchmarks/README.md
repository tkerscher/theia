# Benchmarks

This directory contains benchmarks for comparing performance between different
versions. These are usually tailored towards specific parts of the framework,
for instance focusing on the ray tracing part only, and do not necessarily
model real world use cases.

The benchmark framework is currently considered work in progress and **will**
change in the future. Once it has matured enough, it will likely be moved to
`hephaistos` as it's not specific to `theia`.

## Basic Usage

To run all benchmarks open a shell in this directory and run:

```bash
python benchmark.py all.py
```

The results will be summarized in a table printed in the console once all
benchmarks have finished.
