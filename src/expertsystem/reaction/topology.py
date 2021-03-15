"""All modules related to topology building.

Responsible for building all possible topologies bases on basic user input:

- number of initial state particles
- number of final state particles

The main interface is the `.StateTransitionGraph`.
"""

import copy
import itertools
import logging
from collections import abc
from typing import (
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Generic,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    ValuesView,
)

import attr

from .quantum_numbers import InteractionProperties

_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenDict(  # pylint: disable=too-many-ancestors
    Generic[_K, _V], abc.Hashable, abc.Mapping
):
    def __init__(self, mapping: Optional[Mapping] = None):
        self.__mapping: Dict[_K, _V] = {}
        if mapping is not None:
            self.__mapping = dict(mapping)
        self.__hash = hash(None)
        if len(self.__mapping) != 0:
            self.__hash = 0
            for key_value_pair in self.items():
                self.__hash ^= hash(key_value_pair)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__mapping})"

    def __iter__(self) -> Iterator[_K]:
        return iter(self.__mapping)

    def __len__(self) -> int:
        return len(self.__mapping)

    def __getitem__(self, key: _K) -> _V:
        return self.__mapping[key]

    def __hash__(self) -> int:
        return self.__hash

    def keys(self) -> KeysView[_K]:
        return self.__mapping.keys()

    def items(self) -> ItemsView[_K, _V]:
        return self.__mapping.items()

    def values(self) -> ValuesView[_V]:
        return self.__mapping.values()


def _to_optional_int(optional_int: Optional[int]) -> Optional[int]:
    if optional_int is None:
        return None
    return int(optional_int)


@attr.s(frozen=True)
class Edge:
    """Struct-like definition of an edge, used in `Topology`."""

    originating_node_id: Optional[int] = attr.ib(
        default=None, converter=_to_optional_int
    )
    ending_node_id: Optional[int] = attr.ib(
        default=None, converter=_to_optional_int
    )

    def get_connected_nodes(self) -> Set[int]:
        connected_nodes = {self.ending_node_id, self.originating_node_id}
        connected_nodes.discard(None)
        return connected_nodes  # type: ignore


def _to_frozenset(iterable: Iterable[int]) -> FrozenSet[int]:
    if not all(map(lambda i: isinstance(i, int), iterable)):
        raise TypeError(f"Not all items in {iterable} are of type int")
    return frozenset(iterable)


@attr.s(frozen=True)
class Topology:
    """Directed Feynman-like graph without edge or node properties.

    Forms the underlying topology of `StateTransitionGraph`. The graphs are
    directed, meaning the edges are ingoing and outgoing to specific nodes
    (since feynman graphs also have a time axis).
    Note that a `Topology` is not strictly speaking a graph from graph theory,
    because it allows open edges, like a Feynman-diagram.
    """

    nodes: FrozenSet[int] = attr.ib(converter=_to_frozenset)
    edges: FrozenDict[int, Edge] = attr.ib(converter=FrozenDict)

    incoming_edge_ids: FrozenSet[int] = attr.ib(init=False, repr=False)
    outgoing_edge_ids: FrozenSet[int] = attr.ib(init=False, repr=False)
    intermediate_edge_ids: FrozenSet[int] = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self.__verify()
        object.__setattr__(
            self,
            "incoming_edge_ids",
            frozenset(
                edge_id
                for edge_id, edge in self.edges.items()
                if edge.originating_node_id is None
            ),
        )
        object.__setattr__(
            self,
            "outgoing_edge_ids",
            frozenset(
                edge_id
                for edge_id, edge in self.edges.items()
                if edge.ending_node_id is None
            ),
        )
        object.__setattr__(
            self,
            "intermediate_edge_ids",
            frozenset(self.edges)
            ^ self.incoming_edge_ids
            ^ self.outgoing_edge_ids,
        )

    def __verify(self) -> None:
        """Verify if there are no dangling edges or nodes."""
        for edge_id, edge in self.edges.items():
            connected_nodes = edge.get_connected_nodes()
            if not connected_nodes:
                raise ValueError(
                    f"Edge nr. {edge_id} is not connected to any node ({edge})"
                )
            if not connected_nodes <= self.nodes:
                raise ValueError(
                    f"{edge} (ID: {edge_id}) has non-existing node IDs.\n"
                    f"Available node IDs: {self.nodes}"
                )
        self.__check_isolated_nodes()

    def __check_isolated_nodes(self) -> None:
        if len(self.nodes) < 2:
            return
        for node_id in self.nodes:
            surrounding_nodes = self.__get_surrounding_nodes(node_id)
            if not surrounding_nodes:
                raise ValueError(
                    f"Node {node_id} is unconnected to any other node"
                )

    def __get_surrounding_nodes(self, node_id: int) -> Set[int]:
        surrounding_nodes = set()
        for edge in self.edges.values():
            connected_nodes = edge.get_connected_nodes()
            if node_id in connected_nodes:
                surrounding_nodes |= connected_nodes
        surrounding_nodes.discard(node_id)
        return surrounding_nodes

    def is_isomorphic(self, other: "Topology") -> bool:
        """Check if two graphs are isomorphic.

        Returns:
            bool:
                True if the two graphs have a one-to-one mapping of the node IDs
                and edge IDs.
        """
        raise NotImplementedError

    def get_edge_ids_ingoing_to_node(self, node_id: int) -> Set[int]:
        return {
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.ending_node_id == node_id
        }

    def get_edge_ids_outgoing_from_node(self, node_id: int) -> Set[int]:
        return {
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.originating_node_id == node_id
        }

    def get_originating_final_state_edge_ids(self, node_id: int) -> Set[int]:
        fs_edges = self.outgoing_edge_ids
        edge_ids = set()
        temp_edge_list = self.get_edge_ids_outgoing_from_node(node_id)
        while temp_edge_list:
            new_temp_edge_list = set()
            for edge_id in temp_edge_list:
                if edge_id in fs_edges:
                    edge_ids.add(edge_id)
                else:
                    new_node_id = self.edges[edge_id].ending_node_id
                    if new_node_id is not None:
                        new_temp_edge_list.update(
                            self.get_edge_ids_outgoing_from_node(new_node_id)
                        )
            temp_edge_list = new_temp_edge_list
        return edge_ids

    def get_originating_initial_state_edge_ids(self, node_id: int) -> Set[int]:
        is_edges = self.incoming_edge_ids
        edge_ids: Set[int] = set()
        temp_edge_list = self.get_edge_ids_ingoing_to_node(node_id)
        while temp_edge_list:
            new_temp_edge_list = set()
            for edge_id in temp_edge_list:
                if edge_id in is_edges:
                    edge_ids.add(edge_id)
                else:
                    new_node_id = self.edges[edge_id].originating_node_id
                    if new_node_id is not None:
                        new_temp_edge_list.update(
                            self.get_edge_ids_ingoing_to_node(new_node_id)
                        )
            temp_edge_list = new_temp_edge_list
        return edge_ids

    def organize_edge_ids(self) -> "Topology":
        """Create a new topology with edge IDs in range :code:`[-m, n+i]`.

        where :code:`m` is the number of `.incoming_edge_ids`, :code:`n` is the
        number of `.outgoing_edge_ids`, and :code:`i` is the number of
        `.intermediate_edge_ids`.

        In other words, relabel the edges so that:

        - `.incoming_edge_ids` lies in the range :code:`[-1, -2, ...]`
        - `.outgoing_edge_ids` lies in the range :code:`[0, 1, ..., n]`
        - `.intermediate_edge_ids` lies in the range :code:`[n+1, n+2, ...]`
        """
        new_edges = dict()
        # Relabel so that initial edge IDs are [-1, -2, ...]
        for new_edge_id, edge_id in zip(
            range(-1, -len(self.incoming_edge_ids) - 1, -1),
            self.incoming_edge_ids,
        ):
            new_edges[new_edge_id] = self.edges[edge_id]
        # Relabel so that
        # outgoing edge IDs are [0, 1, 2, ..., n]
        # intermediate edge IDs are [n+1, n+2, ...]
        for new_edge_id, edge_id in enumerate(
            tuple(self.outgoing_edge_ids) + tuple(self.intermediate_edge_ids)
        ):
            new_edges[new_edge_id] = self.edges[edge_id]
        return attr.evolve(self, edges=new_edges)

    def swap_edges(self, edge_id1: int, edge_id2: int) -> "Topology":
        new_edges = dict(self.edges.items())
        new_edges.update(
            {
                edge_id1: self.edges[edge_id2],
                edge_id2: self.edges[edge_id1],
            }
        )
        return attr.evolve(self, edges=FrozenDict(new_edges))


def get_originating_node_list(
    topology: Topology, edge_ids: Iterable[int]
) -> List[int]:
    """Get list of node ids from which the supplied edges originate from.

    Args:
        edge_ids ([int]): list of edge ids for which the origin node is
            searched for

    Returns:
        [int]: a list of node ids
    """

    def __get_originating_node(edge_id: int) -> Optional[int]:
        return topology.edges[edge_id].originating_node_id

    return [
        node_id for node_id in map(__get_originating_node, edge_ids) if node_id
    ]


@attr.s(kw_only=True)
class _MutableTopology:
    edges: Dict[int, Edge] = attr.ib(factory=dict, converter=dict)
    nodes: Set[int] = attr.ib(factory=set, converter=set)

    def freeze(self) -> Topology:
        return Topology(
            edges=self.edges,
            nodes=self.nodes,
        )

    def add_node(self, node_id: int) -> None:
        """Adds a node with id node_id.

        Raises:
            ValueError: if node_id already exists
        """
        if node_id in self.nodes:
            raise ValueError(f"Node with id {node_id} already exists!")
        self.nodes.add(node_id)

    def add_edges(self, edge_ids: List[int]) -> None:
        """Add edges with the ids in the edge_ids list."""
        for edge_id in edge_ids:
            if edge_id in self.edges:
                raise ValueError(f"Edge with id {edge_id} already exists!")
            self.edges[edge_id] = Edge()

    def attach_edges_to_node_ingoing(
        self, ingoing_edge_ids: Iterable[int], node_id: int
    ) -> None:
        """Attach existing edges to nodes.

        So that the are ingoing to these nodes.

        Args:
            ingoing_edge_ids ([int]): list of edge ids, that will be attached
            node_id (int): id of the node to which the edges will be attached

        Raises:
            ValueError: if an edge not doesn't exist.
            ValueError: if an edge ID is already an ingoing node.
        """
        # first check if the ingoing edges are all available
        for edge_id in ingoing_edge_ids:
            if edge_id not in self.edges:
                raise ValueError(f"Edge with id {edge_id} does not exist!")
            if self.edges[edge_id].ending_node_id is not None:
                raise ValueError(
                    f"Edge with id {edge_id} is already ingoing to"
                    f" node {self.edges[edge_id].ending_node_id}"
                )

        # update the newly connected edges
        for edge_id in ingoing_edge_ids:
            edge = self.edges[edge_id]
            self.edges[edge_id] = Edge(
                ending_node_id=node_id,
                originating_node_id=edge.originating_node_id,
            )

    def attach_edges_to_node_outgoing(
        self, outgoing_edge_ids: Iterable[int], node_id: int
    ) -> None:
        # first check if the ingoing edges are all available
        for edge_id in outgoing_edge_ids:
            if edge_id not in self.edges:
                raise ValueError(f"Edge with id {edge_id} does not exist!")
            if self.edges[edge_id].originating_node_id is not None:
                raise ValueError(
                    f"Edge with id {edge_id} is already outgoing from"
                    f" node {self.edges[edge_id].originating_node_id}"
                )

        # update the edges
        for edge_id in outgoing_edge_ids:
            edge = self.edges[edge_id]
            self.edges[edge_id] = Edge(
                ending_node_id=edge.ending_node_id,
                originating_node_id=node_id,
            )


@attr.s
class InteractionNode:
    """Helper class for the `.SimpleStateTransitionTopologyBuilder`."""

    number_of_ingoing_edges: int = attr.ib(
        validator=attr.validators.instance_of(int)
    )
    number_of_outgoing_edges: int = attr.ib(
        validator=attr.validators.instance_of(int)
    )

    def __attrs_post_init__(self) -> None:
        if self.number_of_ingoing_edges < 1:
            raise ValueError("NumberOfIngoingEdges has to be larger than 0")
        if self.number_of_outgoing_edges < 1:
            raise ValueError("NumberOfOutgoingEdges has to be larger than 0")


class SimpleStateTransitionTopologyBuilder:
    """Simple topology builder.

    Recursively tries to add the interaction nodes to available open end
    edges/lines in all combinations until the number of open end lines matches
    the final state lines.
    """

    def __init__(
        self, interaction_node_set: Iterable[InteractionNode]
    ) -> None:
        if not isinstance(interaction_node_set, list):
            raise TypeError("interaction_node_set must be a list")
        self.interaction_node_set: List[InteractionNode] = list(
            interaction_node_set
        )

    def build(
        self, number_of_initial_edges: int, number_of_final_edges: int
    ) -> Tuple[Topology, ...]:
        number_of_initial_edges = int(number_of_initial_edges)
        number_of_final_edges = int(number_of_final_edges)
        if number_of_initial_edges < 1:
            raise ValueError("number_of_initial_edges has to be larger than 0")
        if number_of_final_edges < 1:
            raise ValueError("number_of_final_edges has to be larger than 0")

        logging.info("building topology graphs...")
        # result list
        graph_tuple_list: List[Tuple[_MutableTopology, List[int]]] = []
        # create seed graph
        seed_graph = _MutableTopology()
        current_open_end_edges = list(range(number_of_initial_edges))
        seed_graph.add_edges(current_open_end_edges)
        extendable_graph_list: List[Tuple[_MutableTopology, List[int]]] = [
            (seed_graph, current_open_end_edges)
        ]

        while extendable_graph_list:
            active_graph_list = extendable_graph_list
            extendable_graph_list = []
            for active_graph in active_graph_list:
                # check if finished
                if (
                    len(active_graph[1]) == number_of_final_edges
                    and len(active_graph[0].nodes) > 0
                ):
                    active_graph[0].freeze()  # verify
                    graph_tuple_list.append(active_graph)
                    continue

                extendable_graph_list.extend(self._extend_graph(active_graph))

        logging.info("finished building topology graphs...")
        # strip the current open end edges list from the result graph tuples
        topologies = list()
        for graph_tuple in graph_tuple_list:
            topology = graph_tuple[0].freeze()
            topology = topology.organize_edge_ids()
            topologies.append(topology)
        return tuple(topologies)

    def _extend_graph(
        self, pair: Tuple[_MutableTopology, Sequence[int]]
    ) -> List[Tuple[_MutableTopology, List[int]]]:
        extended_graph_list: List[Tuple[_MutableTopology, List[int]]] = []

        topology, current_open_end_edges = pair

        # Try to extend the graph with interaction nodes
        # that have equal or less ingoing lines than active lines
        for interaction_node in self.interaction_node_set:
            if interaction_node.number_of_ingoing_edges <= len(
                current_open_end_edges
            ):
                # make all combinations
                combis = list(
                    itertools.combinations(
                        current_open_end_edges,
                        interaction_node.number_of_ingoing_edges,
                    )
                )
                # remove all combinations that originate from the same nodes
                for comb1, comb2 in itertools.combinations(combis, 2):
                    if get_originating_node_list(
                        topology, comb1  # type: ignore
                    ) == get_originating_node_list(
                        topology, comb2  # type: ignore
                    ):
                        combis.remove(comb2)

                for combi in combis:
                    new_graph = _attach_node_to_edges(
                        pair, interaction_node, combi
                    )
                    extended_graph_list.append(new_graph)

        return extended_graph_list


def create_isobar_topologies(
    number_of_final_states: int,
) -> Tuple[Topology, ...]:
    if number_of_final_states < 2:
        raise ValueError(
            "At least two final states required for an isobar decay"
        )
    builder = SimpleStateTransitionTopologyBuilder([InteractionNode(1, 2)])
    topologies = builder.build(
        number_of_initial_edges=1,
        number_of_final_edges=number_of_final_states,
    )
    return tuple(topologies)


def create_n_body_topology(
    number_of_initial_states: int, number_of_final_states: int
) -> Topology:
    n_in = number_of_initial_states
    n_out = number_of_final_states
    builder = SimpleStateTransitionTopologyBuilder(
        [
            InteractionNode(
                number_of_ingoing_edges=n_in,
                number_of_outgoing_edges=n_out,
            )
        ]
    )
    topologies = builder.build(
        number_of_initial_edges=n_in,
        number_of_final_edges=n_out,
    )
    decay_name = f"{n_in} to {n_out}"
    if len(topologies) == 0:
        raise ValueError(f"Could not create n-body decay for {decay_name}")
    if len(topologies) > 1:
        raise RuntimeError(f"Several n-body decays for {decay_name}")
    topology = next(iter(topologies))
    return topology


def _attach_node_to_edges(
    graph: Tuple[_MutableTopology, Sequence[int]],
    interaction_node: InteractionNode,
    ingoing_edge_ids: Iterable[int],
) -> Tuple[_MutableTopology, List[int]]:
    temp_graph = copy.deepcopy(graph[0])
    new_open_end_lines = list(copy.deepcopy(graph[1]))

    # add node
    new_node_id = len(temp_graph.nodes)
    temp_graph.add_node(new_node_id)

    # attach the edges to the node
    temp_graph.attach_edges_to_node_ingoing(ingoing_edge_ids, new_node_id)
    # update the newly connected edges
    for edge_id in ingoing_edge_ids:
        new_open_end_lines.remove(edge_id)

    # make new edges for the outgoing lines
    new_edge_start_id = len(temp_graph.edges)
    new_edge_ids = list(
        range(
            new_edge_start_id,
            new_edge_start_id + interaction_node.number_of_outgoing_edges,
        )
    )
    temp_graph.add_edges(new_edge_ids)
    temp_graph.attach_edges_to_node_outgoing(new_edge_ids, new_node_id)
    for edge_id in new_edge_ids:
        new_open_end_lines.append(edge_id)

    return (temp_graph, new_open_end_lines)


EdgeType = TypeVar("EdgeType")
"""A `~typing.TypeVar` representing the type of edge properties."""


class StateTransitionGraph(Generic[EdgeType]):
    """Graph class that resembles a frozen `.Topology` with properties.

    This class should contain the full information of a state transition from a
    initial state to a final state. This information can be attached to the
    nodes and edges via properties.
    In case not all information is provided, error can be raised on property
    retrieval.
    """

    def __init__(
        self,
        topology: Topology,
        node_props: Mapping[int, InteractionProperties],
        edge_props: Mapping[int, EdgeType],
    ):
        self.__node_props = dict(node_props)
        self.__edge_props = dict(edge_props)
        if not isinstance(topology, Topology):
            raise TypeError
        self.topology = topology

    def __post_init__(self) -> None:
        _assert_over_defined(self.topology.nodes, self.__node_props)
        _assert_over_defined(self.topology.edges, self.__edge_props)

    def __eq__(self, other: object) -> bool:
        """Check if two `.StateTransitionGraph` instances are **identical**."""
        if isinstance(other, StateTransitionGraph):
            if self.topology != other.topology:
                return False
            for i in self.topology.edges:
                if self.get_edge_props(i) != other.get_edge_props(i):
                    return False
            for i in self.topology.nodes:
                if self.get_node_props(i) != other.get_node_props(i):
                    return False
            return True
        raise NotImplementedError(
            f"Cannot compare {self.__class__.__name__}"
            f" with {other.__class__.__name__}"
        )

    def get_node_props(self, node_id: int) -> InteractionProperties:
        return self.__node_props[node_id]

    def get_edge_props(self, edge_id: int) -> EdgeType:
        return self.__edge_props[edge_id]

    def evolve(
        self,
        node_props: Optional[Dict[int, InteractionProperties]] = None,
        edge_props: Optional[Dict[int, EdgeType]] = None,
    ) -> "StateTransitionGraph[EdgeType]":
        """Changes the node and edge properties of a graph instance.

        Since a `.StateTransitionGraph` is frozen (cannot be modified), the
        evolve function will also create a shallow copy the properties.
        """
        new_node_props = copy.copy(self.__node_props)
        if node_props:
            _assert_over_defined(self.topology.nodes, node_props)
            for node_id, node_prop in node_props.items():
                new_node_props[node_id] = node_prop

        new_edge_props = copy.copy(self.__edge_props)
        if edge_props:
            _assert_over_defined(self.topology.edges, edge_props)
            for edge_id, edge_prop in edge_props.items():
                new_edge_props[edge_id] = edge_prop

        return StateTransitionGraph[EdgeType](
            topology=self.topology,
            node_props=new_node_props,
            edge_props=new_edge_props,
        )

    def compare(
        self,
        other: "StateTransitionGraph",
        edge_comparator: Optional[Callable[[EdgeType, EdgeType], bool]] = None,
        node_comparator: Optional[
            Callable[[InteractionProperties, InteractionProperties], bool]
        ] = None,
    ) -> bool:
        if self.topology != other.topology:
            return False
        if edge_comparator is not None:
            for i in self.topology.edges:
                if not edge_comparator(
                    self.get_edge_props(i), other.get_edge_props(i)
                ):
                    return False
        if node_comparator is not None:
            for i in self.topology.nodes:
                if not node_comparator(
                    self.get_node_props(i), other.get_node_props(i)
                ):
                    return False
        return True

    def swap_edges(self, edge_id1: int, edge_id2: int) -> None:
        self.topology = self.topology.swap_edges(edge_id1, edge_id2)
        value1: Optional[EdgeType] = None
        value2: Optional[EdgeType] = None
        if edge_id1 in self.__edge_props:
            value1 = self.__edge_props.pop(edge_id1)
        if edge_id2 in self.__edge_props:
            value2 = self.__edge_props.pop(edge_id2)
        if value1 is not None:
            self.__edge_props[edge_id2] = value1
        if value2 is not None:
            self.__edge_props[edge_id1] = value2


def _assert_over_defined(items: Collection, properties: Mapping) -> None:
    defined = set(properties)
    existing = set(items)
    over_defined = existing & defined ^ defined
    if over_defined:
        raise ValueError(
            "Properties have been defined for items that don't exist."
            f" Available items: {existing}, over-defined: {over_defined}"
        )
