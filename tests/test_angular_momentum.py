from typing import Optional

import pytest
from qrules.particle import Particle, ParticleWithSpin
from qrules.topology import (
    Edge,
    InteractionProperties,
    StateTransitionGraph,
    Topology,
)

from ampform.helicity import _extract_angular_momentum


def _create_graph_dummy(
    l_magnitude: Optional[int], spin_magnitude: float
) -> StateTransitionGraph[ParticleWithSpin]:
    topology = Topology(
        nodes={0},  # type: ignore
        edges={
            0: Edge(None, 0),
            1: Edge(0, None),
            2: Edge(0, None),
        },
    )
    return StateTransitionGraph[ParticleWithSpin](
        topology,
        node_props={0: InteractionProperties(l_magnitude=l_magnitude)},
        edge_props={
            0: (
                Particle(name="dummy", pid=123, spin=spin_magnitude, mass=1.0),
                spin_magnitude,
            )
        },
    )


@pytest.mark.parametrize(
    ("graph", "expected_l"),
    [
        (_create_graph_dummy(1, 0.5), 1),
        (_create_graph_dummy(0, 1.0), 0),
        (_create_graph_dummy(2, 1.0), 2),
        (_create_graph_dummy(None, 0.0), 0),
        (_create_graph_dummy(None, 1.0), 1),
    ],
)
def test_extract_angular_momentum(
    graph: StateTransitionGraph[ParticleWithSpin], expected_l: int
) -> None:
    assert expected_l == _extract_angular_momentum(graph, 0)


@pytest.mark.parametrize(
    "graph",
    [
        _create_graph_dummy(None, 0.5),
        _create_graph_dummy(None, 1.5),
    ],
)
def test_invalid_angular_momentum(
    graph: StateTransitionGraph[ParticleWithSpin],
) -> None:
    with pytest.raises(ValueError, match="not integral"):
        _extract_angular_momentum(graph, 0)
