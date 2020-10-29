"""All modules related to topology building.

Responsible for building all possible topologies bases on basic user input:

- number of initial state particles
- number of final state particles

The main interface is the `.StateTransitionGraph`.
"""

import copy
import itertools
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import attr

from .quantum_numbers import InteractionProperties


@attr.s
class Edge:
    """Struct-like definition of an edge, used in `Topology`."""

    ending_node_id: Optional[int] = attr.ib(default=None)
    originating_node_id: Optional[int] = attr.ib(default=None)

    def get_connected_nodes(self) -> Set[int]:
        connected_nodes = {self.ending_node_id, self.originating_node_id}
        connected_nodes.discard(None)
        return connected_nodes  # type: ignore


class Topology:
    """Directed Feynman-like graph for a state transition.

    Forms the underlying topology of `StateTransitionGraph`. Note that a
    `Topology` is not strictly speaking a graph from graph theory, because it
    allows open edges, like a Feynman-diagram.
    """

    def __init__(
        self,
        nodes: Optional[Set[int]] = None,
        edges: Optional[Dict[int, Edge]] = None,
    ) -> None:
        self.__nodes: Set[int] = set()
        self.__edges: Dict[int, Edge] = dict()
        if nodes is not None:
            self.__nodes = set(nodes)
        if edges is not None:
            self.__edges = edges
        self.verify()

    @property
    def nodes(self) -> Set[int]:
        return self.__nodes

    @property
    def edges(self) -> Dict[int, Edge]:
        return self.__edges

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{(self.nodes, self.edges)}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Topology):
            return self.nodes == other.nodes and self.edges == other.edges
        raise NotImplementedError

    def add_node(self, node_id: int) -> None:
        """Adds a node with id node_id.

        Raises:
            ValueError: if node_id already exists
        """
        if node_id in self.__nodes:
            raise ValueError(f"Node with id {node_id} already exists!")
        self.__nodes.add(node_id)

    def add_edges(self, edge_ids: List[int]) -> None:
        """Add edges with the ids in the edge_ids list."""
        for edge_id in edge_ids:
            if edge_id in self.__edges:
                raise ValueError(f"Edge with id {edge_id} already exists!")
            self.__edges[edge_id] = Edge()

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
            if edge_id not in self.__edges:
                raise ValueError(f"Edge with id {edge_id} does not exist!")
            if self.__edges[edge_id].ending_node_id is not None:
                raise ValueError(
                    f"Edge with id {edge_id} is already ingoing to"
                    f" node {self.__edges[edge_id].ending_node_id}"
                )

        # update the newly connected edges
        for edge_id in ingoing_edge_ids:
            self.__edges[edge_id].ending_node_id = node_id

    def attach_edges_to_node_outgoing(
        self, outgoing_edge_ids: Iterable[int], node_id: int
    ) -> None:
        # first check if the ingoing edges are all available
        for edge_id in outgoing_edge_ids:
            if edge_id not in self.__edges:
                raise ValueError(f"Edge with id {edge_id} does not exist!")
            if self.__edges[edge_id].originating_node_id is not None:
                raise ValueError(
                    f"Edge with id {edge_id} is already outgoing from"
                    f" node {self.__edges[edge_id].originating_node_id}"
                )

        # update the edges
        for edge_id in outgoing_edge_ids:
            self.__edges[edge_id].originating_node_id = node_id

    def verify(self) -> None:
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
        # EdgeIndexMapping = {}
        # NodeIndexMapping = {}

        # get start edges
        # CurrentEdges = [graph1.getInitial]

        # while(CurrentEdges):
        #    TempEdges = CurrentEdges
        #    CurrentEdges = []

        # check if the mapping is still valid and can be extended

    def get_originating_node_list(
        self, edge_ids: Iterable[int]
    ) -> List[Optional[int]]:
        """Get list of node ids from which the supplied edges originate from.

        Args:
            edge_ids ([int]): list of edge ids for which the origin node is
                searched for

        Returns:
            [int]: a list of node ids
        """
        return [
            self.edges[edge_id].originating_node_id for edge_id in edge_ids
        ]

    def get_initial_state_edges(self) -> List[int]:
        return sorted(
            [
                edge_id
                for edge_id, edge in self.edges.items()
                if edge.originating_node_id is None
            ]
        )

    def get_final_state_edges(self) -> List[int]:
        return sorted(
            [
                edge_id
                for edge_id, edge in self.edges.items()
                if edge.ending_node_id is None
            ]
        )

    def get_intermediate_state_edges(self) -> List[int]:
        return sorted(
            [
                edge_id
                for edge_id, edge in self.edges.items()
                if edge.ending_node_id is not None
                and edge.originating_node_id is not None
            ]
        )

    def get_edges_ingoing_to_node(self, node_id: Optional[int]) -> List[int]:
        return [
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.ending_node_id == node_id
        ]

    def get_edges_outgoing_from_node(
        self, node_id: Optional[int]
    ) -> List[int]:
        return [
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.originating_node_id == node_id
        ]

    def get_originating_final_state_edges(
        self, node_id: Optional[int]
    ) -> List[int]:
        fs_edges = self.get_final_state_edges()
        edge_list = []
        temp_edge_list = self.get_edges_outgoing_from_node(node_id)
        while temp_edge_list:
            new_temp_edge_list = []
            for edge_id in temp_edge_list:
                if edge_id in fs_edges:
                    edge_list.append(edge_id)
                else:
                    new_node_id = self.edges[edge_id].ending_node_id
                    new_temp_edge_list.extend(
                        self.get_edges_outgoing_from_node(new_node_id)
                    )
            temp_edge_list = new_temp_edge_list
        return edge_list

    def get_originating_initial_state_edges(self, node_id: int) -> List[int]:
        is_edges = self.get_initial_state_edges()
        edge_list = []
        temp_edge_list = self.get_edges_ingoing_to_node(node_id)
        while temp_edge_list:
            new_temp_edge_list = []
            for edge_id in temp_edge_list:
                if edge_id in is_edges:
                    edge_list.append(edge_id)
                else:
                    new_node_id = self.edges[edge_id].originating_node_id
                    new_temp_edge_list.extend(
                        self.get_edges_ingoing_to_node(new_node_id)
                    )
            temp_edge_list = new_temp_edge_list
        return edge_list

    def swap_edges(self, edge_id1: int, edge_id2: int) -> None:
        popped_edge_id1 = self.__edges.pop(edge_id1)
        popped_edge_id2 = self.__edges.pop(edge_id2)
        self.__edges[edge_id2] = popped_edge_id1
        self.__edges[edge_id1] = popped_edge_id2


_EdgeType = TypeVar("_EdgeType")


class StateTransitionGraph(Topology, Generic[_EdgeType]):
    """Graph class that contains edges and nodes.

    Similar to feynman graphs. The graphs are directed, meaning the edges are
    ingoing and outgoing to specific nodes (since feynman graphs also have a
    time axis) This class can contain the full information of a state
    transition from a initial state to a final state. This information can be
    attached to the nodes and edges via properties.
    """

    def __init__(
        self,
        nodes: Optional[Set[int]] = None,
        edges: Optional[Dict[int, Edge]] = None,
    ) -> None:
        super().__init__(nodes, edges)
        self.node_props: Dict[int, InteractionProperties] = {}
        self.edge_props: Dict[int, _EdgeType] = {}
        self.graph_node_properties_comparator: Optional[Callable] = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nodes={self.nodes}, edges={self.edges})"
            f"\n    node props: {self.node_props}"
            f"\n    edge props: {self.edge_props}"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StateTransitionGraph):
            if self.nodes != other.nodes:
                return False
            if self.edges != other.edges:
                return False
            if self.edge_props != other.edge_props:
                return False
            if self.graph_node_properties_comparator is not None:
                return self.graph_node_properties_comparator(
                    self.node_props, other.node_props
                )
            return self.node_props == other.node_props

        raise NotImplementedError

    def __copy__(self) -> "StateTransitionGraph[_EdgeType]":
        """Makes a *shallow* copy.

        The default shallow copy behavior was overwritten to also make copies
        one level deeper. In other words copies of the containers (`nodes`,
        `edges`, `node_props`, `edge_props`) are made as well. Just like the
        default copy behavior, the existing contents of the containers are
        **NOT** copied!
        """
        new_nodes = copy.copy(self.nodes)
        new_edges = copy.copy(self.edges)
        new_graph = type(self)(nodes=new_nodes, edges=new_edges)
        new_graph.node_props = copy.copy(self.node_props)
        new_graph.edge_props = copy.copy(self.edge_props)
        return new_graph

    @staticmethod
    def from_topology(topology: Topology) -> "StateTransitionGraph":
        """Create a `StateTransitionGraph` from a `Topology`."""
        return StateTransitionGraph(topology.nodes, topology.edges)

    def compare(
        self,
        other: "StateTransitionGraph",
        edge_comparator: Optional[Callable[[Any, Any], bool]] = None,
        node_comparator: Optional[Callable[[Any, Any], bool]] = None,
    ) -> bool:
        if self.nodes != other.nodes:
            return False
        if self.edges != other.edges:
            return False
        if edge_comparator is not None:
            for i in self.edges:
                if not edge_comparator(
                    self.edge_props[i], other.edge_props[i]
                ):
                    return False
        if node_comparator is not None:
            for i in self.nodes:
                if not node_comparator(
                    self.node_props[i], other.node_props[i]
                ):
                    return False
        return True

    def swap_edges(self, edge_id1: int, edge_id2: int) -> None:
        super().swap_edges(edge_id1, edge_id2)
        value1: Optional[_EdgeType] = None
        value2: Optional[_EdgeType] = None
        if edge_id1 in self.edge_props:
            value1 = self.edge_props.pop(edge_id1)
        if edge_id2 in self.edge_props:
            value2 = self.edge_props.pop(edge_id2)
        if value1 is not None:
            self.edge_props[edge_id2] = value1
        if value2 is not None:
            self.edge_props[edge_id1] = value2


class InteractionNode:  # pylint: disable=too-few-public-methods
    """Struct-like definition of an interaction node."""

    def __init__(
        self,
        type_name: str,
        number_of_ingoing_edges: int,
        number_of_outgoing_edges: int,
    ) -> None:
        if not isinstance(number_of_ingoing_edges, int):
            raise TypeError("NumberOfIngoingEdges must be an integer")
        if not isinstance(number_of_outgoing_edges, int):
            raise TypeError("NumberOfOutgoingEdges must be an integer")
        if number_of_ingoing_edges < 1:
            raise ValueError("NumberOfIngoingEdges has to be larger than 0")
        if number_of_outgoing_edges < 1:
            raise ValueError("NumberOfOutgoingEdges has to be larger than 0")
        self.type_name = str(type_name)
        self.number_of_ingoing_edges = int(number_of_ingoing_edges)
        self.number_of_outgoing_edges = int(number_of_outgoing_edges)


class SimpleStateTransitionTopologyBuilder:
    """Simple topology builder.

    Recursively tries to add the interaction nodes to available open end
    edges/lines in all combinations until the number of open end lines matches
    the final state lines.
    """

    def __init__(
        self, interaction_node_set: Sequence[InteractionNode]
    ) -> None:
        if not isinstance(interaction_node_set, list):
            raise TypeError("interaction_node_set must be a list")
        self.interaction_node_set = list(interaction_node_set)

    def build_graphs(
        self, number_of_initial_edges: int, number_of_final_edges: int
    ) -> List[Topology]:
        number_of_initial_edges = int(number_of_initial_edges)
        number_of_final_edges = int(number_of_final_edges)
        if number_of_initial_edges < 1:
            raise ValueError("number_of_initial_edges has to be larger than 0")
        if number_of_final_edges < 1:
            raise ValueError("number_of_final_edges has to be larger than 0")

        logging.info("building topology graphs...")
        # result list
        graph_tuple_list = []
        # create seed graph
        seed_graph = Topology()
        current_open_end_edges = list(range(number_of_initial_edges))
        seed_graph.add_edges(current_open_end_edges)
        extendable_graph_list = [(seed_graph, current_open_end_edges)]

        while extendable_graph_list:
            active_graph_list = extendable_graph_list
            extendable_graph_list = []
            for active_graph in active_graph_list:
                # check if finished
                if (
                    len(active_graph[1]) == number_of_final_edges
                    and len(active_graph[0].nodes) > 0
                ):
                    active_graph[0].verify()
                    graph_tuple_list.append(active_graph)
                    continue

                extendable_graph_list.extend(self.extend_graph(active_graph))

            # check if two topologies are the same
            for graph_index1, graph_index2 in itertools.combinations(
                range(len(extendable_graph_list)), 2
            ):
                if extendable_graph_list[graph_index1][0].is_isomorphic(
                    extendable_graph_list[graph_index2][0]
                ):
                    extendable_graph_list.remove(
                        extendable_graph_list[graph_index2]
                    )

        logging.info("finished building topology graphs...")
        # strip the current open end edges list from the result graph tuples
        result_graph_list = []
        for graph_tuple in graph_tuple_list:
            result_graph_list.append(graph_tuple[0])
        return result_graph_list

    def extend_graph(
        self, graph: Tuple[Topology, Sequence[int]]
    ) -> List[Tuple[Topology, List[int]]]:
        extended_graph_list: List[Tuple[Topology, List[int]]] = []

        current_open_end_edges = graph[1]

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
                    if graph[0].get_originating_node_list(comb1) == graph[
                        0
                    ].get_originating_node_list(comb2):
                        combis.remove(comb2)

                for combi in combis:
                    new_graph = _attach_node_to_edges(
                        graph, interaction_node, combi
                    )
                    extended_graph_list.append(new_graph)

        return extended_graph_list


def _attach_node_to_edges(
    graph: Tuple[Topology, Sequence[int]],
    interaction_node: InteractionNode,
    ingoing_edge_ids: Sequence[int],
) -> Tuple[Topology, List[int]]:
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
