# Units Module

This module defines useful SI units that can be used for e.g. material and
tracers. Its usage is recommended as it enhances the readability. For example:

```python
import theia.units as u

dist = 10.0 * u.m
lamb = 500.0 * u.nm
position = (0.2, 0.3, 0.5) * u.km
mu_s = 0.25 / u.m
v_g = 0.8 * u.c
```

::: theia.units
