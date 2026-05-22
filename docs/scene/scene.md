---
icon: lucide/theater
---

# Scene

Scenes describe the environment within light is traced. It at least consists of
a single medium describing the optical properties such as absorption and
scattering, but can also contain arbitrary geometries to simulate accurate
reflection and refraction between multiple media.

!!! info
    The creation of scenes require ray tracing capable hardware. You can however
    always create media for use in volume tracers.

## World and Object Space

The simulation uses two different coordinate system: The world space is the one
within light rays are traced whereas the object space is used when interacting
with meshes such as targets. These are related through an (affine)
transformation matrix. The idea behind this is to make the reported hit position
and direction independent of the exact position and orientation of the
corresponding target. This way we do not have to create a distinct
[hit response](../pipeline/components.md#hit-response) for each target or change it
when a target moves. It further allows to to reuse the same mesh by giving
different instances different transformations.

## Creating a Scene

Before a scene can be created its building blocks needs to be uploaded to the
GPU. `MaterialStore` handles the uploading of media and material, while
`MeshStore` handles the preparation of meshes. Both of them are immutable in the
sense that you cannot add additional entries after they are created

Once the building blocks are available on the GPU, they can be used to build a
scene. To do so you create instances of your meshes, assign a material to them
and place them in space with a dedicated transformation. This way multiple
instances can share mesh and/or materials between them. Both materials and
meshes can be referenced by their assigned name. Finally, the scene requires a
boundary box used to determine when rays are "lost", i.e. to far away to be
worth further tracing.

The following shows a code example on how the creation of a scene may look like:

```python title="How to create a scene"
from theia.material import BK7Model, Material, MaterialStore, WaterBaseModel
from theia.model import BK7Model, PureWaterModel
from theia.scene import MeshStore, RectBBox, Scene, Transformation
from theia.surface import DielectricSurface
from theia.volume import Attenuating
import theia.units as u # (1)!

# create materials
water = WaterModel(4.0, 10_000, 35.0).createMedium(physicModel=Attenuating())
glass = BK7Model().createMedium() # defaults to transparent physic model
mat = Material("mat", glass, water, DielectricSurface(), flags=("DR", "B")) # (2)!
# make material available to GPU
matStore = MaterialStore([mat])

# load meshes
meshStore = MeshStore({
  "cube": "meshes/cube.ply", # (3)!
  "sphere": "meshes/sphere.stl",
})

# create mesh instances
det0 = meshStore.createInstance(
  "sphere", "mat", # (4)!
  Transform.TRS(scale=0.4 * u.m, translate=(10.0, 0.0, 50.0) * u.m),
  detectorId=0,
)
det1 = meshStore.createInstance(
  "sphere", "mat",
  Transform.TRS(scale=0.4 * u.m, translate(10.0, 0.0, -50.0) * u.m),
  detectorId=1, # (5)!
)

# create scene
scene = Scene(
  [det0, det1],
  matStore.material,
  bbox=RectBBox((-1.0 * u.km, ) * 3, (1.0 * u.km, ) * 3),
)
```

1. Theia provides a unit system for better readability.
2. These flags denote, that from the outside it acts as detector and only allows
   reflection, whereas from the inside it absorbs all light.
3. You can load meshes by simply passing the path to the corresponding file.
4. We reference both the mesh and the material by their assigned name. `Scene`
   will later resolve these.
5. We assigned each detector its own id that can later be used to select which
   detector becomes active during tracing.

