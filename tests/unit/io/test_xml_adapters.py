"""These tests can be removed when the internal `dict` have been removed."""

from expertsystem import io
from expertsystem.data import (
    ComplexEnergyState,
    QuantumState,
    Spin,
)


def test_complex_energy_state():
    state = ComplexEnergyState(
        energy=complex(2.5, 0.3),
        state=QuantumState[Spin](spin=Spin(1.5, -0.5), charge=-1),
    )
    converted_dict = io.xml.object_to_dict(state)
    quantum_numbers = converted_dict["QuantumNumber"]
    spin_dict = quantum_numbers[0]
    charge_dict = quantum_numbers[1]
    assert spin_dict["Value"] == 1.5
    assert spin_dict["Projection"] == -0.5
    assert charge_dict["Value"] == -1
