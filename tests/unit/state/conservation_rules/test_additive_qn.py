import pytest

from expertsystem.state.conservation_rules import ChargeConservation
from expertsystem.state.particle import StateQuantumNumberNames


@pytest.mark.parametrize(
    "graph_input, expected_value",
    [
        (
            (
                [{StateQuantumNumberNames.Charge: 0},],
                [
                    {StateQuantumNumberNames.Charge: -1},
                    {StateQuantumNumberNames.Charge: 1},
                ],
                {},
            ),
            True,
        ),
    ],
)
def test_charge_conservation(graph_input, expected_value):
    rule = ChargeConservation()

    assert rule(*graph_input) == expected_value
