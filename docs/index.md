# Documentation

!!! info

    We assume you already know what theia is [about](about.md). Here we explain
    how it works and how you can use it.

Ignoring the internal machinery, the user facing part of theia is mostly split
into two main parts:

- **Scene**:  
Scenes describe the environment in which to simulate light, e.g. a detector.
They contain multiple mesh surfaces subdividing its volume. Optical properties
and models can be assigned to both the surfaces and volumes to define and
control their interaction with the simulated light.
- **Simulation**:  
Simulations quite literary bring light into scenes as they define how and where
light is created and detected. They use the scene to sample light paths and
process the resulting hits to produce a final result such as a light yield
estimate.

You can select a specific topic on the left to learn more about it or press
_Next_ on the bottom to read the documentation from front to back.

!!! note

    This part of the documentation gives a higher level description of the design
    and concepts in theia. If you are looking for the API documentation, you can
    find them [here](api/index.md).
