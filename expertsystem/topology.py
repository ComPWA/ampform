"""All modules related to topology building.

Responsible for building all possible topologies bases on basic user input:

  - number of initial state particles
  - number of final state particles
"""

import copy
import itertools
import logging
from collections import OrderedDict
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
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
            if _dicts_unequal(self.edges, other.edges):
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

    def is_isomorphic(self, other: "StateTransitionGraph") -> bool:
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

    def get_initial_state_edges(self) -> List[int]:
        is_list: List[int] = []
        for edge_id, edge in self.edges.items():
            if edge.originating_node_id is None:
                is_list.append(edge_id)
        return sorted(is_list)

    def get_final_state_edges(self) -> List[int]:
        fs_list: List[int] = []
        for edge_id, edge in self.edges.items():
            if edge.ending_node_id is None:
                fs_list.append(edge_id)
        return sorted(fs_list)

    def get_intermediate_state_edges(self) -> List[int]:
        is_list: List[int] = []
        for edge_id, edge in self.edges.items():
            if (
                edge.ending_node_id is not None
                and edge.originating_node_id is not None
            ):
                is_list.append(edge_id)
        return sorted(is_list)

    def get_edges_ingoing_to_node(self, node_id: Optional[int]) -> List[int]:
        edge_list: List[int] = []
        for edge_id, edge in self.edges.items():
            if edge.ending_node_id == node_id:
                edge_list.append(edge_id)
        return edge_list

    def get_edges_outgoing_to_node(self, node_id: Optional[int]) -> List[int]:
        edge_list: List[int] = []
        for edge_id, edge in self.edges.items():
            if edge.originating_node_id == node_id:
                edge_list.append(edge_id)
        return edge_list

    def get_originating_final_state_edges(
        self, node_id: Optional[int]
    ) -> List[int]:
        fs_edges = self.get_final_state_edges()
        edge_list = []
        temp_edge_list = self.get_edges_outgoing_to_node(node_id)
        while temp_edge_list:
            new_temp_edge_list = []
            for edge_id in temp_edge_list:
                if edge_id in fs_edges:
                    edge_list.append(edge_id)
                else:
                    new_node_id = self.edges[edge_id].ending_node_id
                    new_temp_edge_list.extend(
                        self.get_edges_outgoing_to_node(new_node_id)
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
    ) -> List[StateTransitionGraph]:
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
        seed_graph = StateTransitionGraph()
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
                    if active_graph[0].verify():
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
        self, graph: Tuple[StateTransitionGraph, Sequence[int]]
    ) -> List[Tuple[StateTransitionGraph, List[int]]]:
        extended_graph_list: List[Tuple[StateTransitionGraph, List[int]]] = []

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
                    new_graph = attach_node_to_edges(
                        graph, interaction_node, combi
                    )
                    extended_graph_list.append(new_graph)

        return extended_graph_list


def attach_node_to_edges(
    graph: Tuple[StateTransitionGraph, Sequence[int]],
    interaction_node: InteractionNode,
    ingoing_edge_ids: Sequence[int],
) -> Tuple[StateTransitionGraph, List[int]]:
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


def _dicts_unequal(dict1: dict, dict2: dict) -> bool:
    return OrderedDict(sorted(dict1.items())) != OrderedDict(
        sorted(dict2.items())
    )
