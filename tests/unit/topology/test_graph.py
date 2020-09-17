# pylint: disable=redefined-outer-name

import pytest

from expertsystem import io
from expertsystem import ui
from expertsystem.state.particle import initialize_graph
from expertsystem.topology import (
    InteractionNode,
    StateTransitionGraph,
)


class TestInteractionNode:
    @staticmethod
    def test_constructor_exceptions():
        dummy_type_name = "type_name"
        with pytest.raises(TypeError):
            assert InteractionNode(
                dummy_type_name,
                number_of_ingoing_edges="has to be int",  # type: ignore
                number_of_outgoing_edges=2,
            )
        with pytest.raises(TypeError):
            assert InteractionNode(
                dummy_type_name,
                number_of_outgoing_edges="has to be int",  # type: ignore
                number_of_ingoing_edges=2,
            )
        with pytest.raises(ValueError):
            assert InteractionNode(
                dummy_type_name,
                number_of_outgoing_edges=0,
                number_of_ingoing_edges=1,
            )
        with pytest.raises(ValueError):
            assert InteractionNode(
                dummy_type_name,
                number_of_outgoing_edges=1,
                number_of_ingoing_edges=0,
            )


def create_dummy_topology() -> StateTransitionGraph:
    topology = StateTransitionGraph()
    topology.add_node(0)
    topology.add_node(1)
    topology.add_edges([0, 1, 2, 3, 4])
    topology.attach_edges_to_node_ingoing([0], 0)
    topology.attach_edges_to_node_ingoing([1], 1)
    topology.attach_edges_to_node_outgoing([1, 2], 0)
    topology.attach_edges_to_node_outgoing([3, 4], 1)
    return topology


@pytest.fixture(scope="package")
def dummy_topology():
    return create_dummy_topology()


def test_initialize_graph(  # pylint: disable=unused-argument
    dummy_topology, particle_database
):
    graphs = initialize_graph(
        dummy_topology,
        initial_state=[("J/psi(1S)", [-1, +1])],
        final_state=["gamma", "pi0", "pi0"],
        particles=particle_database,
    )
    assert len(graphs) == 8
    return graphs


def visualize_graphs():
    """Render graphs when running this file directly."""
    ui.load_default_particles()
    topology = create_dummy_topology()
    graphs = test_initialize_graph(topology, io.load_pdg())
    io.write(graphs, "jpsi_to_gamma_pi0_pi0.gv")


if __name__ == "__main__":
    visualize_graphs()
