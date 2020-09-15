from dataclasses import FrozenInstanceError

import pytest

from expertsystem.data import Particle, Spin


@pytest.mark.parametrize(
    "instance",
    [
        Particle(
            name="jpsi",
            pid=1234,
            mass=3.0969,
            width=9.29e-05,
            spin=1,
            charge=0,
        ),
    ],
)
def test_repr(instance):
    copy_from_repr = eval(repr(instance))  # pylint: disable=eval-used
    assert copy_from_repr == instance


def test_immutability():
    with pytest.raises(FrozenInstanceError):
        test_state = Particle(
            "MyParticle",
            123,
            mass=1.2,
            width=0.1,
            spin=1,
            charge=0,
            isospin=Spin(1, 0),
        )
        test_state.charge = 1  # type: ignore


def test_complex_energy_equality():
    with pytest.raises(AssertionError):
        assert Particle(
            "MyParticle", pid=123, mass=1.5, width=0.1, spin=1,
        ) == Particle("MyParticle", pid=123, mass=1.5, width=0.2, spin=1)

    assert Particle(
        "MyParticle",
        123,
        mass=1.2,
        width=0.1,
        spin=1,
        charge=0,
        isospin=Spin(1, 0),
    ) == Particle(
        "MyParticle",
        123,
        mass=1.2,
        width=0.1,
        spin=1,
        charge=0,
        isospin=Spin(1, 0),
    )


@pytest.mark.parametrize(
    "magnitude, projection", [(0.3, 0.3), (1.0, 0.5), (0.5, 0.0), (-0.5, 0.5)],
)
def test_spin_exceptions(magnitude, projection):
    with pytest.raises(ValueError):
        print(Spin(magnitude, projection))
