---
icon: lucide/mirror-rectangular
---

# Materials

While [meshes](mesh.md) define the boundaries between volumes, `Materials`
must be assigned to their instances to define the optical properties and
interaction of both volumes on each side and the interface itself.
Materials fulfill three tasks:

- **Storing Optical Properties**  
Materials define the optical properties of both the surface itself and the
volumes on each side. The optical properties of volumes bundled into `Medium`
and referenced by `Material` to allow reuse between different material.
- **Implementing Optical Processes**  
Both `Material` and `Medium` expect a `SurfaceModel` and `VolumeModel`
respectively, each containing the implementation of a specific physical
interaction model, such as diffuse reflection on surfaces or scattering in
volumes.
- **Guiding Path Sampling**  
Flags can be assigned to guide the path sampling
process. This can be used for instance to mark surfaces as detectors or absorber
as well as limiting surfaces to only transmit or reflect light.

## Property Tables

Optical properties are stored on the GPU using two separate _property tables_,
one for media and on for materials. Each optical property, as well as each
medium and material separately get a zero based index. When looking up a optical
property, these define the row and columns respectively pointing to the cell
containing the data. Each cell is exactly 8 bytes large and can either store
the property directly or a pointer to it.

Subclasses of `theia.property.Property` are responsible for populating these
cells and any potential additional data needed. They also handle the
[serialization and deserialization](material.md#storing-and-loading-from-disk).
The two most important types of properties are explained in the following:

### Scalar Properties

The simplest properties contain scalar values of either 32 bit floats
(`FloatProperty`) or 32 bit integers (`IntProperty`). Since two of those fit
into 8 bytes, up to two can be stored per property.

### Look Up Tables

The more complex but also more usefull properties are look up tables. They
approximate arbitrary functions in one or two dimensions by storing discrete
samples placed in regular intervals and interpolating between them. Currently
the following interpolation methods are implemented:

- linear
- cubic
- monotonic (Steffen)

!!! info
    Theia requires equidistant sampled look up tables for faster lookups.

## Medium

A `Medium` defines the optical properties of a volume. It consists of a
dictionary containing the values of its optical properties and a
`VolumeModel` defining its interaction with light. It further contains a name
for easier look up and implicit referencing.

Creating a medium is straight forward:

```python
from theia.lookup import Table
from theia.material import Medium
from theia.property import TableProperty
from theia.volume import Attenuating
import theia.units as u

mu_a = Table(np.array([...]))
mu_s = Table(np.array([...]))

props = {
    "refractive_index": TableProperty.createConstTable(1.33),
    "absorption_coef": TableProperty(mu_a),
    "scattering_coef": TableProperty(mu_s),
}
medium = Medium(
    name="water",
    wavelength_range=(400., 750.0) * u.nm,
    properties = props,
    physicModel = Attenuating(),    
)
```

!!! info
    Which optical properties are necessary and how they need to be named are
    defined by the physical models of the medium and any material using it.

## Material

`Material` are similar to `Medium`, but specify the optical properties of
surfaces. They, too, consist of a name, a dictionary with the values of the
optical properties and a physic model, but now derived from `SurfaceModel`.
Additionally, the reference two `Medium` (potentially the same) either directly
or by name and assign them to the inside and outside volume.

Flags can be used to steer the path sampling of a tracer as it encounters a
surface with a specific material. They can be assigned for both directions,
from inside to outside(outward) or from outside to inside(inward) separately.
Currently, the following flags are defined:

- `BLACK_BODY` (`B`)
- `DETECTOR` (`D`)
- `LIGHT_SOURCE` (`L`)
- `NO_REFLECT`
- `NO_TRANSMIT`
- `SKIP_MEDIA_MISMATCH_TEST` (`*`)
- `TRANSMIT_HIT` (`t`)

Check [`theia.material.MaterialFlags`](../api/material.md#theia.material.MaterialFlags)
and [`theia.material.parseMaterialFlag`](../api/material.md#theia.material.parseMaterialFlags)
for more details.

!!! warning
    The `LIGHT_SOURCE` flag is a special flag only used by `SceneBackwardTargetTracer`
    and does not make a surface a light source!

Creating a material works in the same way as with medium:

```python
from theia.lookup import Table
from theia.material import Material
from theia.property import FloatProperty, TableProperty
from theia.surface import LambertReflectingSurface

water = Medium(...)
metal = Medium(...)

props = {
    "reflectivity": TableProperty.createConstProperty(0.8),
}
material = Material(
    "casing",                   # name
    metal,                      # inside
    water,                      # outside
    LambertReflectingSurface(), # physic model
    flags="R",                  # reflect only
    properties=props,
)
```

!!! info
    Which optical properties are necessary and how they need to be named are
    defined by the physical model.

## Optical Models

`OpticalModel` allows the creation of both `Medium` and `Material` from analytic
optical models after deriving from it. Properties further decorated with
[`@optical_property`](../api/model.md#theia.model.optical_property) will be stored as
[scalar properties](material.md#scalar-properties). Similar decorated functions
will be sampled to create [look up tables](material.md#look-up-tables).

If not specified, the name of the class member is used as the property name.
For functions, the sampling range and sample count can be specified. The range
defaults to `wavelengthRange` specified by the class.

There exist some derived functions from `optical_property` that specify a
different range:

- `angular_property`: (-1.0, 1.0)
- `sampler_property`: (0.0, 1.0)

In addition, [`numerical_sampler_property`](../api/model.md#theia.model.numerical_sampler_property)
numerically creates the inverse cumulative function to create a sampler.

```python
from theia.model import OpticalModel, optical_property, angular_property
import theia.units as u

class Model(OpticalModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) # (1)!
        self.restrictWavelengthRange((200.0, 1000.0) * u.nm) # (2)!
    
    @optical_property
    @property
    def reflectivity(self) -> float:
        return 1.0
    
    @optical_property
    def refractive_index(self, lam: float) -> float:
        return 1.0 + lam / 100.0
    
    @optical_property(name="extra_prop", range=(5.0, 10.0))
    def funky_func(self, x: float) -> float:
        return 10.0 + x ** 2

medium = Model.createMedium(name="myMedium")
```

1. This pattern is known as _cooperative multiple inheritance_ and allows to
   combine multiple models to create new
2. Restricts the range over which optical properties will be sampled. If
   multiple models are combined through (multiple) inheritance, only the
   intersection of all ranges will be sampled.

There are already a few optical models that can even be combined through
multiple inheritance:

```python
from theia.model import DispersionFreeMedium, HenyeyGreensteinPhaseFunction
from theia.volume import Attenuating

class Model(DispersionFreeMedium, HenyeyGreensteinPhaseFunction):
    def __init__(self) -> None:
        super().__init__(
            n=1.33, ng=1.33, mu_a=1.0 / u.m, mu_s=1.0 / u.m, g = 0.9 # (1)!
        )

medium = Model.createMedium(physicModel=Attenuating())
```

1. Thanks to cooperative multiple inheritance we can pass all arguments the
   individual constructors need in one go.

## Physical Models

Physical models contain the implementation describing the interaction of light
within a medium (`VolumeModel`) or with a surface (`SurfaceModel`) using the
optical properties. They can be assigned to each medium and surface separately
allowing to use multiple ones in the same simulation.

## Material Store

After creating all media and materials, they still need to be uploaded to the
GPU. This is done by passing them to a `MaterialStore`:

```python
from theia.material import Material, MaterialStore, Medium

water = Medium(...)
glass = Medium(...)
steel = Medium(...)

window = Material(...)
casing = Material(...)

store = MaterialStore([window, casing], media=[steel])
```

Any media referenced by the materials passed to `MaterialStore` will be also
added alongside any additional specified media in the `media` parameter.

Both media and material are referenced by their indices in the respective
[property table](material.md#property-tables). These can be looked up using
their previously specified name:

```python
waterIdx = store.media["water"]
casing = store.material["casing"]
```

## Storing and Loading from Disk

To reduce boiler-plate code, both medium and material can be saved to and loaded
from disk using `saveMaterials` and `loadMaterials` respectively. The latter
returns two dictionaries for media and material respectively, each indexed by
name:

```python
from theia.material import loadMaterial, saveMaterial

path = "~/some/path/materials"

saveMaterial(path, [window, casing], media=[steel])
materials, media = loadMaterials(path)
```

Like with `MaterialStore`, any medium referenced by a material will also be
saved to disk.
