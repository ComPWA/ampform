from dataclasses import FrozenInstanceError

import pytest

from expertsystem.data import (
    ComplexEnergy,
    ComplexEnergyState,
    QuantumState,
    Spin,
)


@pytest.mark.parametrize(
    "instance",
    [
        QuantumState(spin=Spin(1, 1), charge=0),
        ComplexEnergy(complex(3.0969, 9.29e-05)),
        ComplexEnergyState(
            complex(3.0969, 9.29e-05),
            state=QuantumState(spin=Spin(1, 1), charge=0),
        ),
    ],
)
def test_repr(instance):
    copy_from_repr = eval(repr(instance))  # pylint: disable=eval-used
    assert copy_from_repr == instance


def test_immutability():
    with pytest.raises(FrozenInstanceError):
        test_state = QuantumState(
            spin=Spin(1, 1), charge=0, isospin=Spin(1, 0)
        )
        test_state.charge = 1  # type: ignore

    with pytest.raises(FrozenInstanceError):
        test_state2 = ComplexEnergyState(
            complex(1.2, 0.1),
            state=QuantumState(spin=Spin(1, 1), charge=0, isospin=Spin(1, 0)),
        )
        test_state2.state.charge = 1  # type: ignore


def test_complex_energy_equality():
    with pytest.raises(AssertionError):
        assert ComplexEnergy(complex(1.5, 0.1)) == ComplexEnergy(
            complex(1.5, 0.2)
        )

    assert ComplexEnergy(complex(1.5, 0.2)) == ComplexEnergy(complex(1.5, 0.2))
    assert ComplexEnergyState(
        complex(1.2, 0.1),
        state=QuantumState(spin=Spin(1, 1), charge=0, isospin=Spin(1, 0)),
    ) == ComplexEnergyState(
        complex(1.2, 0.1),
        state=QuantumState(spin=Spin(1, 1), charge=0, isospin=Spin(1, 0)),
    )
