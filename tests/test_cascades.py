import pytest

import theia.cascades
import theia.units as u


@pytest.mark.parametrize("pType", list(theia.cascades.ParticleType))
def test_createParamsFromParticle(pType: theia.cascades.ParticleType, rng) -> None:
    # create random particle
    particle = theia.cascades.Particle(
        pType,
        tuple(rng.random(3) * 10.0) * u.m,
        tuple(rng.random(3)),
        (rng.random() * 100.0 + 10.0) * u.GeV,
        rng.random() * 500.0 * u.ns,
        (rng.random() * 500.0 + 100.0) * u.m,
        0.8 * u.c,
    )

    # we wont be able to use 'UNKNOWN' type
    if pType == theia.cascades.ParticleType.UNKNOWN:
        with pytest.raises(ValueError):
            theia.cascades.createParamsFromParticle(particle)
        return

    # check whether conversion works
    source, params, lightYield = theia.cascades.createParamsFromParticle(
        particle, lightSourceName=""
    )
    s = source(**params)
    # check common parameters
    assert params["startTime"] == particle.startTime
    assert params["startPosition"] == particle.position
