# pylint: disable=redefined-outer-name

import pytest

from expertsystem.state.particle import initialize_graph
from expertsystem.topology import (
    InteractionNode,
    SimpleStateTransitionTopologyBuilder,
    Topology,
)


@pytest.fixture(scope="package")
def three_body_decay() -> Topology:
    two_body_decay_node = InteractionNode("TwoBodyDecay", 1, 2)
    simple_builder = SimpleStateTransitionTopologyBuilder(
        [two_body_decay_node]
    )
    all_graphs = simple_builder.build_graphs(1, 3)
    return all_graphs[0]


@pytest.mark.parametrize(
    "final_state_groupings", [[[["pi0", "pi0"]]], [[["gamma", "pi0"]]],],
)
def test_initialize_graph(
    final_state_groupings, three_body_decay, particle_database
):
    graphs = initialize_graph(
        three_body_decay,
        initial_state=[("J/psi(1S)", [-1, +1])],
        final_state=["gamma", "pi0", "pi0"],
        particles=particle_database,
        final_state_groupings=final_state_groupings,
    )
    assert len(graphs) == 4
