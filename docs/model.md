# Mathematical Model

On this page we will introduce and discuss the underlying physical model theia
uses in its simulation. Perhaps the most surprising design choice is to not
trace ballistic photons but rather solve the radiance field at the detector.
This not only allows us to utilize advanced Monte Carlo methods to improve
performance but also gives us the complete light distribution instead of a
specific sample. If desired, one can use the latter to sample discrete photons.

## Light Model

Underlying the simulation of light is the model of light it uses. This section
briefly introduces how we describe light and its interaction with media.

### Photons and Radiance

Instead of discrete photons we will use radiance $L$, i.e. the energy per area
and solid angle. Since the physical model only considers monochromatic light,[^1]
these are actually equivalent descriptions:[^2]

[^1]: Wavelength is a parameter of the model, but to unclutter the notation a bit
      we usually do not state it explicitly in the formulas.

[^2]: $A_\bot = A\cos\theta$ describes the projected area onto a surface. In
      volume we assume the surface to be normal to the light ray.

$$
L = \frac{\partial^2 E}{\partial A_\bot\partial\Omega}
  = \frac{hc}{\lambda} \frac{\partial^2 N}{\partial A_\bot\partial\Omega}
$$

Following the same principle, the expected radiance our Monte Carlo simulation
calculates is equivalent to an expected photon count.

### Attenuation

As light propagates through media in a straight line it gets attenuated due to
absorption and scattering each described by their corresponding coefficient
$\mu_a$ and $\mu_s$.[^3] These are often combined into a single attenuation or
extinction coefficient $\mu_e = \mu_a + \mu_s$. The attenuation of the initial
radiance $L_0$ after it has traveled a distance $d$ is then described by the
**Beer-Lambert law**:

[^3]: One also often finds the corresponding length $\lambda_i = 1/\mu_i$ instead.

$$
L = L_0e^{-\mu_e\cdot d}
$$

### Volume Scattering

While absorption makes portion of the light unavailable, volume scattering simply
changes it directions. As stated earlier, the amount of light scattered per
distance traveled is denoted by the scattering coefficient $\mu_s$. Missing is
the angular distribution of the scattered light per scatter event, modelled by
the _phase function_ $f(\cos\theta)$, where $\cos\theta$ is the angle between
the incident and scattered light ray.[^4] The scattered light can then be
described using the following formula:

[^4]: More generally $f$ can depend on the exact incident and scattered direction.
      However, in the isotropic media we consider it only depends on the angle
      between them.

$$
L_s = \mu_s \int_\Omega f(\cos\theta)L_i \mathrm{d}\omega
$$

In order to be able to unify volume scattering with surface scattering, in the
following we will use a generalized phase function $\hat{f}(\cos\theta) =
\mu_sf(\cos\theta)$. Note that since the phase function is used in an integral
over solid angles its normalization constrain is actually:

$$
2\pi\int_0^\pi f(\cos\theta)\sin\theta \mathrm{d}\theta = 1
$$

### Reflection and Transmission

At interfaces between media of different refractive index reflection and
transmission occurs. As of now, we only consider specular surfaces as governed
by [**Snell's law**](https://en.wikipedia.org/wiki/Snell%27s_law) and the
[**law of reflection**](https://en.wikipedia.org/wiki/Specular_reflection#Law_of_reflection).
The ratios of the corresponding radiances $\mathcal{R}$ and $\mathcal{T}$ are
described by the [**Fresnel equations**](https://en.wikipedia.org/wiki/Fresnel_equations).
To unify this with the scattering model we can write down a corresponding phase
function using dirac deltas:

$$
\hat{f}(\omega_i, \omega_o) = \mathcal{R}\frac{\delta(\omega_o-r(\omega_i))}{|\cos\theta_r|} +
\frac{n_t^2}{n_i^2}\mathcal{T}\frac{\delta(\omega_o-t(\omega_i))}{|\cos\theta_t|}
$$

where we explicitly depend on the incident $\omega_i$ and outgoing $\omega_o$
direction. The cosines are needed to cancel the corresponding ones in the
projected area in the definition of radiance to ensure energy conservation.
Additionally, the squared ratio of refractive indices of the media on each side
of the interface account for the change in variable of $\mathrm{d}\omega$
caused by refraction.[^5] For more details see e.g.
[Chapter 8.2](https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission)
of Physically Based Rendering 3rd Edition by M. Pharr et al.

[^5]: When transmitted into a denser media the cone of directions becomes narrower.

Such phase functions handling both reflection and transmission at surfaces are
called _bidirectional scattering distribution function_ (BSDF). Using this
seemingly unnecessary complex description allows us in future to use more
realistic models or even measurements of surfaces.

### Propagation Speed

Perfect sine waves travel through media at the _phase velocity_ $v_p = c/n$
whereas physically more accurate wave packets (i.e. photons) do so at the
_group velocity_ $v_g$. These only agree in non-dispersive media, where the
refractive index $n$ does not depend on the wavelength. More generally the
following holds:

$$
v_g = v_p - \lambda\frac{\partial v_p}{\partial \lambda}
$$

## Propagation Model

Here we first introduce the physical model describing light propagation and then
follow its common reformulation to make it more suitable for Monte Carlo.

### Equation of Transfer

At the heart of the propagation model lies the **Equation of Transfer** as
introduced by Chandrasekhar (1960): First we promote radiance to a vector field
$L(p,\omega)$ parameterized on both position $p$ and direction $\omega$. Light
traveling along a ray $p+\omega\cdot s$ then follows the following
integro-differential equation:

$$
\frac{\partial}{\partial s}L(p,\omega) = -\mu_eL(p,\omega) + \int_\Omega L(p,\omega')\hat{f}(\cos\theta)\mathrm{d}\omega'
$$

If "scattered" on a surface, we pick up an extra factor $|\omega\cdot\omega'$|
inside the integral effectively canceling the cosines introduced in the
reflection and transmission phase function.

Integrating it along the light ray this gives us:

$$
L(p(s_1), \omega) = \int_{s_0}^{s_1} \frac{\partial}{\partial s}L(p(s),\omega)\mathrm{d}s
$$

Light sources can be introduced as boundary conditions where we a priori define
the emitted radiance $L_e$ for specific positions and directions.

To make notations more readable we adopt the common path notation, where we
describe a light path $\bar{p}_N$ consisting of straight elements between points
$p_i$ with $p_1\rightarrow p_2\rightarrow ... \rightarrow p_{N-1} \rightarrow p_N$.
Explicitly incorporating light sources the Equation of Transfer than becomes:

$$\begin{align*}
L(p'\rightarrow p) &= L_e(p'\rightarrow p)T(p'\rightarrow p) \\
+& \int \hat{f}(p''\rightarrow p'\rightarrow p)L(p''\rightarrow p')T(p''\rightarrow p')G(p''\rightarrow p')d\mu(p'')
\end{align*}$$

where $T$ describes the attenuation and $G$ contains geometric factors such as
the visibility indicator function denoting whether two points are mutual
visible.

### Path Integrals

By iteratively inserting the Equation of Transfer into itself one gets an
infinite sum of regular integrals over paths of increasing length suitable for
Monte Carlo simulation:

$$\begin{align*}
L(p_1\rightarrow p_0) = \sum^\infty_{n=1} \underbrace{\int...\int}_\text{n - 1}
L_e(p_n\rightarrow p_{n-1})T(\bar{p}_n)d\mu(\bar{p}_n) \\
T(\bar{p}_n) = \prod^{n-1}_{i=1}\hat{f}(p_{i+1}\rightarrow p_i\rightarrow p_{i-1})T(p_{i+1}\rightarrow p_i)G(p_{i+1}\rightarrow p_i)
\end{align*}$$

This form has several advantages:

- Paths can be build iteratively, reusing shorter paths for longer ones.
- It does not matter how we sample the paths. We can start tracing at the light
  source, the detector or event from both sides and connect the subpaths.[^6]
- For each path length we can create an "alternative" path were we deliberately
  complete the connection between light source and detector ensuring all
  simulated paths contribute.
- Since scattering also attenuates the light ray longer paths contribute less.
  Ignoring paths above a certain length thus only introduce minimal bias.

[^6]: There is a slight pit fall if one changes the direction paths are sampled.
      The Equation of Transfer starts at the detector, thus the integral
      responsible for scattering goes over incident direction. However, in the
      opposite direction we need to integrate over the outgoing direction
      causing a change of variable noticeable at surfaces.

### Elapsed Time

Finally, the time elapsed as the light completes a path is simply its distance
divided by the corresponding group velocity. Since the media can change for
each path segment we write:

$$
t(\bar{p}_N) = \sum_{i=1}^N \frac{||p_i - p_{i-1}||}{v_g(p_i\rightarrow p_{i-1})}
$$

## Simulation Model

At the end one is usually not interested in the radiance but in the signal $I$
it causes in the detector modelled by the detector response $W$. Both the
light source and detector response, as well as the light path $\bar{p}_N$ were
sampled from their corresponding probability distributions. To obtain the
Monte Carlo estimator, we finally need to divide by their probabilities:[^7]

[^7]: Some tracers use [multiple importance sampling](https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency#MultipleImportanceSampling)
      that introduce additional weights.

$$
I(\bar{p}_N) = \frac{L_e(p_N\rightarrow p_{N-1})}{p_L(p_N\rightarrow p_{N-1})}
            \cdot\frac{T(\bar{p}_N)}{p_T(\bar{p}_N)}
            \cdot\frac{W(p_1\rightarrow p_0)}{p_W(p_1\rightarrow p_0)}
$$

In the code we call the values of such fractions _contribution_.
