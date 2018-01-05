"""graph module - some description here."""


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
    return is_list


def get_final_state_edges(graph):
    fs_list = []
    for edge_id, edge in graph.edges.items():
        if edge.ending_node_id is None:
            fs_list.append(edge_id)
    return fs_list


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
        for x, y in self.node_props:
            return_string = return_string + str(x) + ": " + str(y) + "\n"
        return_string = return_string + "}\n"
        return_string = return_string + "\nedge props: {\n"
        for x, y in self.edge_props:
            return_string = return_string + str(x) + ": " + str(y) + "\n"
        return_string = return_string + "}\n"
        return return_string

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
                                 + str(self.edges[edge_id].originating_node_id))

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
