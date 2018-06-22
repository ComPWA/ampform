"""graph module - some description here."""

from collections import OrderedDict


def are_graphs_isomorphic(graph1, graph2):
    """Returns True if the two graphs have a one-to-one mapping
    of the node IDs and edge IDs"""
    # EdgeIndexMapping = {}
    # NodeIndexMapping = {}

    # get start edges
    # CurrentEdges = [graph1.getInitial]

    # while(CurrentEdges):
    #    TempEdges = CurrentEdges
    #    CurrentEdges = []

    # check if the mapping is still valid and can be extended
    return False


def get_initial_state_edges(graph):
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    is_list = []
    for edge_id, edge in graph.edges.items():
        if edge.originating_node_id is None:
            is_list.append(edge_id)
    return sorted(is_list)


def get_final_state_edges(graph):
    fs_list = []
    for edge_id, edge in graph.edges.items():
        if edge.ending_node_id is None:
            fs_list.append(edge_id)
    return sorted(fs_list)


def get_intermediate_state_edges(graph):
    is_list = []
    for edge_id, edge in graph.edges.items():
        if (edge.ending_node_id is not None and
                edge.originating_node_id is not None):
            is_list.append(edge_id)
    return sorted(is_list)


def get_edges_ingoing_to_node(graph, node_id):
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    edge_list = []
    for edge_id, edge in graph.edges.items():
        if edge.ending_node_id == node_id:
            edge_list.append(edge_id)
    return edge_list


def get_edges_outgoing_to_node(graph, node_id):
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    edge_list = []
    for edge_id, edge in graph.edges.items():
        if edge.originating_node_id == node_id:
            edge_list.append(edge_id)
    return edge_list


def get_originating_final_state_edges(graph, node_id):
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
                    get_edges_outgoing_to_node(graph, new_node_id))
        temp_edge_list = new_temp_edge_list
    return edge_list


def get_originating_initial_state_edges(graph, node_id):
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
                    get_edges_ingoing_to_node(graph, new_node_id))
        temp_edge_list = new_temp_edge_list
    return edge_list


def dicts_unequal(dict1, dict2):
    return (OrderedDict(sorted(dict1.items())) !=
            OrderedDict(sorted(dict2.items())))


class StateTransitionGraph:
    """
        Graph class which contains edges and nodes, similar to feynman graphs.
        The graphs are directed, meaning the edges are ingoing and outgoing
        to specific nodes (since feynman graphs also have a time axis)
        This class can contain the full information of a state transition from
        a initial state to a final state. This information can be attached to
        the nodes and edges via properties.
    """

    def __init__(self):
        self.nodes = []
        self.edges = {}
        self.node_props = {}
        self.edge_props = {}
        self.graph_element_properties_comparator = None

    def set_graph_element_properties_comparator(self, comparator):
        self.graph_element_properties_comparator = comparator

    def __repr__(self):
        return_string = "\nnodes: " + \
            str(self.nodes) + "\nedges: " + str(self.edges) + "\n"
        return_string = return_string + "node props: {\n"
        for x, y in self.node_props.items():
            return_string = return_string + str(x) + ": " + str(y) + "\n"
        return_string = return_string + "}\n"
        return_string = return_string + "edge props: {\n"
        for x, y in self.edge_props.items():
            return_string = return_string + str(x) + ": " + str(y) + "\n"
        return_string = return_string + "}\n"
        return return_string

    def __str__(self):
        return_string = "\nnodes: " + \
            str(self.nodes) + "\nedges: " + str(self.edges)
        return_string = return_string + "\nnode props: {\n"
        for x, y in self.node_props.items():
            return_string = return_string + str(x) + ": " + str(y) + "\n"
        return_string = return_string + "}\n"
        return_string = return_string + "\nedge props: {\n"
        for x, y in self.edge_props.items():
            return_string = return_string + str(x) + ": " + str(y) + "\n"
        return_string = return_string + "}\n"
        return return_string

    def __eq__(self, other):
        """
        defines the equal operator for the graph class
        """
        if isinstance(other, StateTransitionGraph):
            if set(self.nodes) != set(other.nodes):
                return False
            if dicts_unequal(self.edges, other.edges):
                return False
            if self.graph_element_properties_comparator is not None:
                if not self.graph_element_properties_comparator(
                        self.node_props, other.node_props):
                    return False
                return self.graph_element_properties_comparator(
                    self.edge_props, other.edge_props)
            else:
                raise NotImplementedError(
                    "Graph element properties comparator is not set!")
            return True
        else:
            return NotImplemented

    def add_node(self, node_id):
        """Adds a node with id node_id. Raises an value error,
        if node_id already exists"""
        if node_id in self.nodes:
            raise ValueError('Node with id ' +
                             str(node_id) + ' already exists!')
        self.nodes.append(node_id)

    def add_edges(self, edge_ids):
        """Adds edges with the ids in the edge_ids list"""
        for edge_id in edge_ids:
            if edge_id in self.edges:
                raise ValueError('Edge with id ' +
                                 str(edge_id) + ' already exists!')
            self.edges[edge_id] = Edge()

    def attach_edges_to_node_ingoing(self, ingoing_edge_ids, node_id):
        """Attaches existing edges to nodes, so that the are ingoing to these
        nodes

        Args:
            ingoing_edge_ids ([int]): list of edge ids, that will be attached
            node_id (int): id of the node to which the edges will be attached

        Raises:
            ValueError
        """
        # first check if the ingoing edges are all available
        for edge_id in ingoing_edge_ids:
            if edge_id not in self.edges:
                raise ValueError('Edge with id ' + str(edge_id)
                                 + ' does not exist!')
            if self.edges[edge_id].ending_node_id is not None:
                raise ValueError('Edge with id ' + str(edge_id)
                                 + ' is already ingoing to node '
                                 + str(self.edges[edge_id].ending_node_id))

        # update the newly connected edges
        for edge_id in ingoing_edge_ids:
            self.edges[edge_id].ending_node_id = node_id

    def attach_edges_to_node_outgoing(self, outgoing_edge_ids, node_id):
        # first check if the ingoing edges are all available
        for edge_id in outgoing_edge_ids:
            if edge_id not in self.edges:
                raise ValueError('Edge with id ' + str(edge_id)
                                 + ' does not exist!')
            if self.edges[edge_id].originating_node_id is not None:
                raise ValueError('Edge with id ' + str(edge_id)
                                 + ' is already outgoing from node '
                                 + str(
                                     self.edges[edge_id].originating_node_id))

        # update the edges
        for edge_id in outgoing_edge_ids:
            self.edges[edge_id].originating_node_id = node_id

    def get_originating_node_list(self, edge_ids):
        """ Get list of node ids from which the supplied edges originate from

        Args:
            edge_ids ([int]): list of edge ids for which the origin node is \
            searched for

        Returns:
            [int]: a list of node ids
        """
        node_list = []
        for edge_id in edge_ids:
            node_list.append(self.edges[edge_id].originating_node_id)
        return node_list

    def swap_edges(self, edge_id1, edge_id2):
        val1 = self.edges.pop(edge_id1)
        val2 = self.edges.pop(edge_id2)

        self.edges[edge_id2] = val1
        self.edges[edge_id1] = val2

        val1 = None
        val2 = None
        if edge_id1 in self.edge_props:
            val1 = self.edge_props.pop(edge_id1)
        if edge_id2 in self.edge_props:
            val2 = self.edge_props.pop(edge_id2)
        if val1:
            self.edge_props[edge_id2] = val1
        if val2:
            self.edge_props[edge_id1] = val2

    def verify(self):
        """ Verify the graph is connected,
        so no dangling parts which are not connected"""

        return True


class InteractionNode:
    """struct-like definition of an interaction node"""

    def __init__(self, type_name, number_of_ingoing_edges,
                 number_of_outgoing_edges):
        if not isinstance(number_of_ingoing_edges, int):
            raise TypeError("NumberOfIngoingEdges must be an integer")
        if not isinstance(number_of_outgoing_edges, int):
            raise TypeError("NumberOfOutgoingEdges must be an integer")
        if number_of_ingoing_edges < 1:
            raise ValueError("NumberOfIngoingEdges has to be larger than 0")
        if number_of_outgoing_edges < 1:
            raise ValueError("NumberOfOutgoingEdges has to be larger than 0")
        self.type_name = type_name
        self.number_of_ingoing_edges = number_of_ingoing_edges
        self.number_of_outgoing_edges = number_of_outgoing_edges


class Edge:
    """struct-like definition of an edge"""

    def __init__(self):
        self.ending_node_id = None
        self.originating_node_id = None

    def __str__(self):
        return str((self.ending_node_id, self.originating_node_id))

    def __repr__(self):
        return str((self.ending_node_id, self.originating_node_id))

    def __eq__(self, other):
        """
        defines the equal operator for the graph class
        """
        if isinstance(other, Edge):
            return (self.ending_node_id == other.ending_node_id and
                    self.originating_node_id == other.originating_node_id)
        else:
            return NotImplemented
