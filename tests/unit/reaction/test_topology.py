# flake8: noqa
# pylint: disable=no-self-use, redefined-outer-name, too-many-arguments

import typing

import attr
import pytest

from expertsystem.reaction.topology import (
    FrozenDict,  # pyright: reportUnusedImport=false
)
from expertsystem.reaction.topology import (
    Edge,
    InteractionNode,
    SimpleStateTransitionTopologyBuilder,
    Topology,
    _MutableTopology,
    create_isobar_topologies,
    create_n_body_topology,
    get_originating_node_list,
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

    @typing.no_type_check
    def test_immutability(self):
        edge = Edge(1, 2)
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            edge.originating_node_id = None
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            edge.originating_node_id += 1
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            edge.ending_node_id = None
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            edge.ending_node_id += 1


class TestInteractionNode:
    @staticmethod
    def test_constructor_exceptions():
        with pytest.raises(TypeError):
            assert InteractionNode(
                number_of_ingoing_edges="has to be int",  # type: ignore
                number_of_outgoing_edges=2,
            )
        with pytest.raises(TypeError):
            assert InteractionNode(
                number_of_outgoing_edges="has to be int",  # type: ignore
                number_of_ingoing_edges=2,
            )
        with pytest.raises(ValueError):
            assert InteractionNode(
                number_of_outgoing_edges=0,
                number_of_ingoing_edges=1,
            )
        with pytest.raises(ValueError):
            assert InteractionNode(
                number_of_outgoing_edges=1,
                number_of_ingoing_edges=0,
            )


class TestMutableTopology:
    @staticmethod
    def test_add_and_attach(two_to_three_decay: Topology):
        topology = _MutableTopology(
            edges=two_to_three_decay.edges,
            nodes=two_to_three_decay.nodes,  # type: ignore
        )
        topology.add_node(3)
        topology.add_edges([7, 8])
        topology.attach_edges_to_node_outgoing([7, 8], 3)
        with pytest.raises(ValueError):
            topology.freeze()
        topology.attach_edges_to_node_ingoing([6], 3)
        assert isinstance(topology.freeze(), Topology)

    @staticmethod
    def test_add_exceptions(two_to_three_decay: Topology):
        topology = _MutableTopology(
            edges=two_to_three_decay.edges,
            nodes=two_to_three_decay.nodes,  # type: ignore
        )
        with pytest.raises(ValueError):
            topology.add_node(0)
        with pytest.raises(ValueError):
            topology.add_edges([0])
        with pytest.raises(ValueError):
            topology.attach_edges_to_node_ingoing([0], 0)
        with pytest.raises(ValueError):
            topology.attach_edges_to_node_ingoing([7], 2)
        with pytest.raises(ValueError):
            topology.attach_edges_to_node_outgoing([6], 2)
        with pytest.raises(ValueError):
            topology.attach_edges_to_node_outgoing([7], 2)


class TestSimpleStateTransitionTopologyBuilder:
    @staticmethod
    def test_two_body_states():
        two_body_decay_node = InteractionNode(1, 2)
        simple_builder = SimpleStateTransitionTopologyBuilder(
            [two_body_decay_node]
        )
        all_graphs = simple_builder.build(1, 3)
        assert len(all_graphs) == 1


class TestTopology:
    @pytest.mark.parametrize(
        "nodes, edges",
        [
            ({1}, dict()),
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
            ([], {0: Edge()}),
            ([], {0: Edge(None, 1)}),
            ({0}, {0: Edge(1, None)}),
            ([], {0: Edge(1, None)}),
            ({0, 1}, {0: Edge(0, None), 1: Edge(None, 1)}),
        ],
    )
    def test_constructor_exceptions(self, nodes, edges):
        with pytest.raises(ValueError):
            assert Topology(nodes=nodes, edges=edges)

    @staticmethod
    def test_repr_and_eq(two_to_three_decay: Topology):
        topology = eval(str(two_to_three_decay))  # pylint: disable=eval-used
        assert topology == two_to_three_decay
        assert topology != float()

    @staticmethod
    def test_getters(two_to_three_decay: Topology):
        topology = two_to_three_decay  # shorter name
        assert get_originating_node_list(topology, edge_ids=[0]) == []
        assert get_originating_node_list(topology, edge_ids=[5, 6]) == [2, 2]
        assert topology.incoming_edge_ids == {0, 1}
        assert topology.outgoing_edge_ids == {4, 5, 6}
        assert topology.intermediate_edge_ids == {2, 3}

    @staticmethod
    @typing.no_type_check
    def test_immutability(two_to_three_decay: Topology):
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            two_to_three_decay.edges = {0: Edge(None, None)}
        with pytest.raises(TypeError):
            two_to_three_decay.edges[0] = Edge(None, None)
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            two_to_three_decay.edges[0].ending_node_id = None
        with pytest.raises(attr.exceptions.FrozenInstanceError):
            two_to_three_decay.nodes = {0, 1}
        with pytest.raises(AttributeError):
            two_to_three_decay.nodes.add(2)
        for node in two_to_three_decay.nodes:
            node += 666
        assert two_to_three_decay.nodes == {0, 1, 2}

    @staticmethod
    def test_organize_edge_ids(two_to_three_decay: Topology):
        topology = two_to_three_decay.organize_edge_ids()
        assert topology.incoming_edge_ids == frozenset({-1, -2})
        assert topology.outgoing_edge_ids == frozenset({0, 1, 2})
        assert topology.intermediate_edge_ids == frozenset({3, 4})

    @staticmethod
    def test_swap_edges(two_to_three_decay: Topology):
        original_topology = two_to_three_decay
        topology = original_topology.swap_edges(0, 1)
        assert topology == original_topology
        topology = topology.swap_edges(5, 6)
        assert topology == original_topology
        topology = topology.swap_edges(4, 6)
        assert topology != original_topology


@pytest.mark.parametrize(
    "n_final, n_topologies, exception",
    [
        (0, None, ValueError),
        (1, None, ValueError),
        (2, 1, None),
        (3, 1, None),
        (4, 2, None),
        (5, 5, None),
        (6, 16, None),
        (7, 61, None),
        (8, 272, None),
    ],
)
def test_create_isobar_topologies(
    n_final: int,
    n_topologies: int,
    exception,
):
    if exception is not None:
        with pytest.raises(exception):
            create_isobar_topologies(n_final)
    else:
        topologies = create_isobar_topologies(n_final)
        assert len(topologies) == n_topologies
        n_expected_nodes = n_final - 1
        n_intermediate_edges = n_final - 2
        for topology in topologies:
            assert len(topology.outgoing_edge_ids) == n_final
            assert len(topology.intermediate_edge_ids) == n_intermediate_edges
            assert len(topology.nodes) == n_expected_nodes


@pytest.mark.parametrize(
    "n_initial, n_final, exception",
    [
        (1, 0, ValueError),
        (0, 1, ValueError),
        (0, 0, ValueError),
        (1, 1, None),
        (2, 1, None),
        (3, 1, None),
        (1, 2, None),
        (1, 3, None),
        (2, 4, None),
    ],
)
def test_create_n_body_topology(n_initial: int, n_final: int, exception):
    if exception is not None:
        with pytest.raises(exception):
            create_n_body_topology(n_initial, n_final)
    else:
        topology = create_n_body_topology(n_initial, n_final)
        assert len(topology.incoming_edge_ids) == n_initial
        assert len(topology.outgoing_edge_ids) == n_final
        assert len(topology.intermediate_edge_ids) == 0
        assert len(topology.nodes) == 1
