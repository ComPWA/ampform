import pytest

from expertsystem.reaction.conservation_rules import ChargeConservation


@pytest.mark.parametrize(
    "graph_input, expected_value",
    [(([0], [-1, 1]), True), (([0], [1, 1]), False)],
)
def test_charge_conservation(graph_input, expected_value):
    assert ChargeConservation()(*graph_input) == expected_value
