---
icon: lucide/ruler
---

# Units

Internally, numbers in theia do not have units. Instead we define what a value
of `1.0` means in their specific context depending on the parameters dimension:

| Dimension  | Unit  |
| ---------- | ----- |
| Length     | `m`   |
| Time       | `ns`  |
| Wavelength | `nm`  |
| Angle      | `rad` |
| Energy     | `GeV` |

!!! warning

    Note that length and wavelength are two separate dimensions and are not
    interchangeable!

To make this easier for the user, theia provides means to use the units directly
in code:

```python
import numpy as np
import theia.units as u

width = 30.0 * u.cm
length = 2.0 * u.m
diameter = 14.0 * u.inch

position = (12.5, -3.0, 17.0) * u.m

lam = np.linspace(400.0, 800.0) * u.nm

E = np.logspace(3.0, 7.0) * u.GeV
```

!!! warning

    The correct usage of units at runtime is neither checked nor enforced!
