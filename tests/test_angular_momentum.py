from typing import Optional

import pytest
from qrules.particle import Particle
from qrules.quantum_numbers import InteractionProperties
from qrules.topology import Edge, Topology
from qrules.transition import State, StateTransition

from ampform.helicity import _extract_angular_momentum


def _create_dummy_transition(
    l_magnitude: Optional[int], spin_magnitude: float
) -> StateTransition:
    topology = Topology(
        nodes={0},  # type: ignore
        edges={
            0: Edge(None, 0),
            1: Edge(0, None),
            2: Edge(0, None),
        },
    )
    particle = Particle(name="dummy", pid=123, spin=spin_magnitude, mass=1.0)
    state = State(particle, spin_magnitude)
    return StateTransition(
        topology,
        interactions={0: InteractionProperties(l_magnitude=l_magnitude)},
        states={0: state, 1: state, 2: state},
    )


@pytest.mark.parametrize(
    ("transition", "expected_l"),
    [
        (_create_dummy_transition(1, 0.5), 1),
        (_create_dummy_transition(0, 1.0), 0),
        (_create_dummy_transition(2, 1.0), 2),
        (_create_dummy_transition(None, 0.0), 0),
        (_create_dummy_transition(None, 1.0), 1),
    ],
)
def test_extract_angular_momentum(
    transition: StateTransition, expected_l: int
) -> None:
    assert expected_l == _extract_angular_momentum(transition, 0)


@pytest.mark.parametrize(
    "transition",
    [
        _create_dummy_transition(None, 0.5),
        _create_dummy_transition(None, 1.5),
    ],
)
def test_invalid_angular_momentum(
    transition: StateTransition,
) -> None:
    with pytest.raises(ValueError, match="not integral"):
        _extract_angular_momentum(transition, 0)
