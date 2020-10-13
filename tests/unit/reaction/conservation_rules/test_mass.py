import pytest

from expertsystem.reaction.conservation_rules import (
    MassConservation,
    MassEdgeInput,
)
from expertsystem.reaction.quantum_numbers import EdgeQuantumNumbers

# Currently need to cast to the proper Edge/NodeQuantumNumber type, see
# https://github.com/ComPWA/expertsystem/issues/255


@pytest.mark.parametrize(
    "rule_input, expected",
    [
        # we assume a two charged pion final state here
        # units are always in GeV
        (
            (
                [
                    MassEdgeInput(
                        mass=EdgeQuantumNumbers.mass(energy[0]),
                        width=EdgeQuantumNumbers.width(energy[1]),
                    )
                ],
                [
                    MassEdgeInput(EdgeQuantumNumbers.mass(0.139)),
                    MassEdgeInput(EdgeQuantumNumbers.mass(0.139)),
                ],
            ),
            expected,
        )
        for energy, expected in zip(
            [
                (0.280, 0.0),
                (0.260, 0.010),
                (0.300, 0.05),
                (0.270, 0.0),
                (0.250, 0.005),
                (0.200, 0.01),
            ],
            [True] * 3 + [False] * 3,
        )
    ],
)
def test_mass_two_body_decay_stable_outgoing(rule_input, expected):
    mass_rule = MassConservation(5)

    assert mass_rule(*rule_input) is expected
