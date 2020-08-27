# pylint: disable=redefined-outer-name

import pytest

from expertsystem import io
from expertsystem import ui
from expertsystem.state.particle import initialize_graph
from expertsystem.topology import StateTransitionGraph


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
    graphs = test_initialize_graph(topology, None)
    try:
        # pylint: disable=import-error,import-outside-toplevel
        import graphviz  # type: ignore

        dot_source = io.dot.convert_to_dot(graphs)
        vis = graphviz.Source(dot_source)
        vis.view()
    except ModuleNotFoundError:
        pass


if __name__ == "__main__":
    visualize_graphs()
