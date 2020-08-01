import pytest

from expertsystem.data import (
    Parity,
    Particle,
    Spin,
)


def test_parity():
    with pytest.raises(ValueError):
        Parity(1.2)
    parity = Parity(+1)
    assert parity == +1
    assert int(parity) == +1


def test_spin():
    with pytest.raises(ValueError):
        Spin(1, -2)
    isospin = Spin(1.5, -0.5)
    assert isospin == 1.5
    assert float(isospin) == 1.5
    assert isospin.magnitude == 1.5
    assert isospin.projection == -0.5


def test_particle():
    particle = Particle("J/psi", 443, charge=0, spin=1, mass=3.0969)
    assert particle.mass == 3.0969
    assert particle.bottomness == 0
