# Pipeline Overview

The Monte Carlo simulation is organized as pipeline consisting of multiple
sub-components. This serves two main reasons:

1. With the GPU being responsible for the heavy lifting in the simulation we
   want to ensure that it does not have to wait for the CPU keeping it from
   doing some actual work. Pipelines allow to prepare the next batch and process
   the result of the previous one while the GPU is still busy.
2. Organizing the code into loosely coupled components allows easy reuse and
   extension of existing code.

Pipelines are a concept inherited from `hephaistos`, the underlying framework
used for interacting with the GPU. The general idea behind it is explained in
[Pipeline Components](pipeline.md), whereas the components that make up the Monte
Carlo simulation are explained in [Simulation Components](components.md).
