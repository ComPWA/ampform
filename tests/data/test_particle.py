import pytest

from expertsystem.data import (
    Parity,
    Particle,
    ParticleCollection,
    QuantumState,
    Spin,
)


@pytest.mark.parametrize(
    "instance",
    [
        ParticleCollection(),
        Spin(2.5, -0.5),
        Parity(1),
        Particle(
            name="J/psi",
            pid=443,
            mass=3.0969,
            width=9.29e-05,
            state=QuantumState[float](spin=1, charge=0),
        ),
    ],
)
def test_repr(instance):
    copy_from_repr = eval(repr(instance))  # pylint: disable=eval-used
    assert copy_from_repr == instance


def test_parity():
    with pytest.raises(ValueError):
        Parity(1.2)
    parity = Parity(+1)
    assert parity == +1
    assert int(parity) == +1
    flipped_parity = -parity
    assert flipped_parity.value == -parity.value


def test_spin():
    with pytest.raises(ValueError):
        Spin(1, -2)
    isospin = Spin(1.5, -0.5)
    assert isospin == 1.5
    assert float(isospin) == 1.5
    assert isospin.magnitude == 1.5
    assert isospin.projection == -0.5

    flipped_spin = -isospin
    assert flipped_spin.magnitude == isospin.magnitude
    assert flipped_spin.projection == -isospin.projection


def test_particle():
    particle = Particle(
        "J/psi", 443, mass=3.0969, state=QuantumState[float](charge=0, spin=1)
    )
    assert particle.mass == 3.0969
    assert particle.width == 0.0
    assert particle.state.bottomness == 0
