# pylint: disable=no-self-use, redefined-outer-name

from copy import deepcopy

import pytest

from expertsystem.reaction.topology import (
    Edge,
    InteractionNode,
    SimpleStateTransitionTopologyBuilder,
    Topology,
)


@pytest.fixture(scope="session")
def two_to_three_decay() -> Topology:
    r"""Create a dummy `Topology`.

    Has the following shape:

    .. code-block::

        e0 -- (N0) -- e2 -- (N1) -- e3 -- (N2) -- e6
              /               \             \
            e1                 e4            e5
    """
    topology = Topology(
        nodes={0, 1, 2},
        edges={
            0: Edge(None, 0),
            1: Edge(None, 0),
            2: Edge(0, 1),
            3: Edge(1, 2),
            4: Edge(1, None),
            5: Edge(2, None),
            6: Edge(2, None),
        },
    )
    return topology


class TestEdge:
    @staticmethod
    def test_get_connected_nodes():
        edge = Edge(1, 2)
        assert edge.get_connected_nodes() == {1, 2}
        edge = Edge(originating_node_id=3)
        assert edge.get_connected_nodes() == {3}
        edge = Edge(ending_node_id=4)
        assert edge.get_connected_nodes() == {4}


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


class TestSimpleStateTransitionTopologyBuilder:
    @staticmethod
    def test_two_body_states():
        four_body_decay_node = InteractionNode("TwoBodyDecay", 1, 2)

        simple_builder = SimpleStateTransitionTopologyBuilder(
            [four_body_decay_node]
        )

        all_graphs = simple_builder.build_graphs(1, 3)

        assert len(all_graphs) == 1


class TestTopology:
    @pytest.mark.parametrize(
        "nodes, edges",
        [
            (None, None),
            ({1}, None),
            (
                {0, 1},
                {
                    0: Edge(None, 0),
                    1: Edge(0, 1),
                    2: Edge(1, None),
                    3: Edge(1, None),
                },
            ),
            (
                {0, 1, 2},
                {
                    0: Edge(None, 0),
                    1: Edge(0, 1),
                    2: Edge(0, 2),
                    3: Edge(1, None),
                    4: Edge(1, None),
                    5: Edge(2, None),
                    6: Edge(2, None),
                },
            ),
        ],
    )
    def test_constructor(self, nodes, edges):
        topology = Topology(nodes=nodes, edges=edges)
        if nodes is None:
            nodes = set()
        if edges is None:
            edges = dict()
        assert topology.nodes == nodes
        assert topology.edges == edges

    @pytest.mark.parametrize(
        "nodes, edges",
        [
            (None, {0: Edge()}),
            (None, {0: Edge(None, 1)}),
            ({0}, {0: Edge(1, None)}),
            (None, {0: Edge(1, None)}),
            ({0, 1}, {0: Edge(0, None), 1: Edge(None, 1)}),
        ],
    )
    def test_constructor_exceptions(self, nodes, edges):
        with pytest.raises(ValueError):
            assert Topology(nodes=nodes, edges=edges)

    @staticmethod
    def test_repr_and_eq(two_to_three_decay):
        topology = eval(str(two_to_three_decay))  # pylint: disable=eval-used
        assert topology == two_to_three_decay
        with pytest.raises(NotImplementedError):
            assert topology == float()

    @staticmethod
    def test_add_and_attach(two_to_three_decay):
        topology = deepcopy(two_to_three_decay)
        topology.add_node(3)
        topology.add_edges([7, 8])
        topology.attach_edges_to_node_outgoing([7, 8], 3)
        with pytest.raises(ValueError):
            topology.verify()
        topology.attach_edges_to_node_ingoing([6], 3)
        assert topology.verify() is None

    @staticmethod
    def test_add_exceptions(two_to_three_decay):
        with pytest.raises(ValueError):
            two_to_three_decay.add_node(0)
        with pytest.raises(ValueError):
            two_to_three_decay.add_edges([0])
        with pytest.raises(ValueError):
            two_to_three_decay.attach_edges_to_node_ingoing([0], 0)
        with pytest.raises(ValueError):
            two_to_three_decay.attach_edges_to_node_ingoing([7], 2)
        with pytest.raises(ValueError):
            two_to_three_decay.attach_edges_to_node_outgoing([6], 2)
        with pytest.raises(ValueError):
            two_to_three_decay.attach_edges_to_node_outgoing([7], 2)

    @staticmethod
    def test_getters(two_to_three_decay):
        topology: Topology = two_to_three_decay  # shorter name
        assert topology.get_originating_node_list([0]) == []
        assert topology.get_originating_node_list([5, 6]) == [2, 2]
        assert topology.get_initial_state_edge_ids() == [0, 1]
        assert topology.get_final_state_edge_ids() == [4, 5, 6]
        assert topology.get_intermediate_state_edge_ids() == [2, 3]

    @staticmethod
    def test_swap(two_to_three_decay):
        topology = deepcopy(two_to_three_decay)
        topology.swap_edges(0, 1)
        assert topology == two_to_three_decay
        topology.swap_edges(5, 6)
        assert topology == two_to_three_decay
        topology.swap_edges(4, 6)
        assert topology != two_to_three_decay
