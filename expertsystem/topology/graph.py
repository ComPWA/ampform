"""Graph module."""

from collections import OrderedDict
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
)


class Edge:
    """Struct-like definition of an edge."""

    def __init__(self) -> None:
        self.ending_node_id: Optional[int] = None
        self.originating_node_id: Optional[int] = None

    def __repr__(self) -> str:
        return f"<class 'Edge': {self.ending_node_id} -> {self.originating_node_id}>"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Edge):
            return (
                self.ending_node_id == other.ending_node_id
                and self.originating_node_id == other.originating_node_id
            )
        raise NotImplementedError


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


class StateTransitionGraph:
    """Graph class that contains edges and nodes.

    Similar to feynman graphs. The graphs are directed, meaning the edges are
    ingoing and outgoing to specific nodes (since feynman graphs also have a
    time axis) This class can contain the full information of a state
    transition from a initial state to a final state. This information can be
    attached to the nodes and edges via properties.
    """

    def __init__(self) -> None:
        self.nodes: List[int] = []
        self.edges: Dict[int, Edge] = {}
        self.node_props: Dict[int, dict] = {}
        self.edge_props: Dict[int, dict] = {}
        self.graph_element_properties_comparator: Optional[Callable] = None

    def set_graph_element_properties_comparator(
        self, comparator: Optional[Callable]
    ) -> None:
        self.graph_element_properties_comparator = comparator

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}()"
            f"\n    nodes: {self.nodes}"
            f"\n    edges: {self.edges}"
            f"\n    node props: {self.node_props}"
            f"\n    node props: {self.edge_props}"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StateTransitionGraph):
            if set(self.nodes) != set(other.nodes):
                return False
            if dicts_unequal(self.edges, other.edges):
                return False
            if self.graph_element_properties_comparator is not None:
                if not self.graph_element_properties_comparator(
                    self.node_props, other.node_props
                ):
                    return False
                return self.graph_element_properties_comparator(
                    self.edge_props, other.edge_props
                )
            raise NotImplementedError(
                "Graph element properties comparator is not set!"
            )
        raise NotImplementedError

    def add_node(self, node_id: int) -> None:
        """Adds a node with id node_id.

        Raises:
            ValueError: if node_id already exists
        """
        if node_id in self.nodes:
            raise ValueError(f"Node with id {node_id} already exists!")
        self.nodes.append(node_id)

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
            self.edges[edge_id].ending_node_id = node_id

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
            self.edges[edge_id].originating_node_id = node_id

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
        node_list: List[Optional[int]] = []
        for edge_id in edge_ids:
            node_list.append(self.edges[edge_id].originating_node_id)
        return node_list

    def swap_edges(self, edge_id1: int, edge_id2: int) -> None:
        popped_edge_id1 = self.edges.pop(edge_id1)
        popped_edge_id2 = self.edges.pop(edge_id2)
        self.edges[edge_id2] = popped_edge_id1
        self.edges[edge_id1] = popped_edge_id2
        value1: Optional[dict] = None
        value2: Optional[dict] = None
        if edge_id1 in self.edge_props:
            value1 = self.edge_props.pop(edge_id1)
        if edge_id2 in self.edge_props:
            value2 = self.edge_props.pop(edge_id2)
        if value1 is not None:
            self.edge_props[edge_id2] = value1
        if value2 is not None:
            self.edge_props[edge_id1] = value2

    def verify(self) -> bool:  # pylint: disable=no-self-use
        """Verify if the graph is connected.

        So that no dangling parts which are not connected.
        """
        return True


def are_graphs_isomorphic(  # pylint: disable=unused-argument
    graph1: StateTransitionGraph, graph2: StateTransitionGraph
) -> bool:
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


def get_initial_state_edges(graph: StateTransitionGraph) -> List[int]:
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    is_list: List[int] = []
    for edge_id, edge in graph.edges.items():
        if edge.originating_node_id is None:
            is_list.append(edge_id)
    return sorted(is_list)


def get_final_state_edges(graph: StateTransitionGraph) -> List[int]:
    fs_list: List[int] = []
    for edge_id, edge in graph.edges.items():
        if edge.ending_node_id is None:
            fs_list.append(edge_id)
    return sorted(fs_list)


def get_intermediate_state_edges(graph: StateTransitionGraph) -> List[int]:
    is_list: List[int] = []
    for edge_id, edge in graph.edges.items():
        if (
            edge.ending_node_id is not None
            and edge.originating_node_id is not None
        ):
            is_list.append(edge_id)
    return sorted(is_list)


def get_edges_ingoing_to_node(
    graph: StateTransitionGraph, node_id: Optional[int]
) -> List[int]:
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    edge_list: List[int] = []
    for edge_id, edge in graph.edges.items():
        if edge.ending_node_id == node_id:
            edge_list.append(edge_id)
    return edge_list


def get_edges_outgoing_to_node(
    graph: StateTransitionGraph, node_id: Optional[int]
) -> List[int]:
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    edge_list: List[int] = []
    for edge_id, edge in graph.edges.items():
        if edge.originating_node_id == node_id:
            edge_list.append(edge_id)
    return edge_list


def get_originating_final_state_edges(
    graph: StateTransitionGraph, node_id: Optional[int]
) -> List[int]:
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    fs_edges = get_final_state_edges(graph)
    edge_list = []
    temp_edge_list = get_edges_outgoing_to_node(graph, node_id)
    while temp_edge_list:
        new_temp_edge_list = []
        for edge_id in temp_edge_list:
            if edge_id in fs_edges:
                edge_list.append(edge_id)
            else:
                new_node_id = graph.edges[edge_id].ending_node_id
                new_temp_edge_list.extend(
                    get_edges_outgoing_to_node(graph, new_node_id)
                )
        temp_edge_list = new_temp_edge_list
    return edge_list


def get_originating_initial_state_edges(
    graph: StateTransitionGraph, node_id: int
) -> List[int]:
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    is_edges = get_initial_state_edges(graph)
    edge_list = []
    temp_edge_list = get_edges_ingoing_to_node(graph, node_id)
    while temp_edge_list:
        new_temp_edge_list = []
        for edge_id in temp_edge_list:
            if edge_id in is_edges:
                edge_list.append(edge_id)
            else:
                new_node_id = graph.edges[edge_id].originating_node_id
                new_temp_edge_list.extend(
                    get_edges_ingoing_to_node(graph, new_node_id)
                )
        temp_edge_list = new_temp_edge_list
    return edge_list


def dicts_unequal(dict1: dict, dict2: dict) -> bool:
    return OrderedDict(sorted(dict1.items())) != OrderedDict(
        sorted(dict2.items())
    )
