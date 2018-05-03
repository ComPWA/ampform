"""graph module - some description here."""

from copy import deepcopy
from collections import OrderedDict
from json import loads, dumps


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


def get_edge_groups_full_attached_node(graph, edge_ids):
    edge_id_groups = {}

    node_in_out_score = {}
    for edge_id in edge_ids:
        edge = graph.edges[edge_id]
        if edge.ending_node_id is not None:
            if edge.ending_node_id in node_in_out_score:
                node_in_out_score[edge.ending_node_id][0] += 1
            else:
                node_in_out_score[edge.ending_node_id] = [1, 0]
        if edge.originating_node_id is not None:
            if edge.originating_node_id in node_in_out_score:
                node_in_out_score[edge.originating_node_id][1] += 1
            else:
                node_in_out_score[edge.originating_node_id] = [0, 1]

    # check if nodes have fully attached outgoing or ingoing edges
    for node_id, in_out_edges in node_in_out_score.items():
        time_level = get_node_time_level(graph, node_id)
        if in_out_edges[0] > 0:
            in_edges = get_edges_ingoing_to_node(graph, node_id)
            if (len(in_edges) == in_out_edges[0]):
                if time_level not in edge_id_groups:
                    edge_id_groups[time_level] = []
                edge_id_groups[time_level].append(in_edges)
        if in_out_edges[1] > 0:
            out_edges = get_edges_outgoing_to_node(graph, node_id)
            if (len(out_edges) == in_out_edges[1]):
                if time_level not in edge_id_groups:
                    edge_id_groups[time_level] = []
                edge_id_groups[time_level].append(out_edges)

    return edge_id_groups


def get_node_time_level(graph, node_id):
    '''
    A graph is ordered in time, due to the hiearchy of the nodes
    This function return the time order of the requested node.
    A time order value of 0 corresponds to the topmost nodes.
    We assume the graph has no cycles...
    '''
    max_time_level = 100
    time_level = -1
    paths_to_top = [[node_id]]
    # we try to find our way up from that node_id
    while len(paths_to_top) > 0:
        if time_level > max_time_level:
            raise ValueError(
                "Reached maximum time level 100."
                " This graph must have a cycle.")
        temp_paths_to_top = paths_to_top
        paths_to_top = []
        for node_path in temp_paths_to_top:
            for edge_id, edge in graph.edges.items():
                if edge.ending_node_id == node_path[-1]:
                    if edge.originating_node_id is None:
                        if time_level < len(node_path):
                            time_level = len(node_path)
                    else:
                        new_path = deepcopy(node_path)
                        new_path.append(edge.originating_node_id)
                        paths_to_top.append(new_path)

    return time_level


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
            if set(self.edges) != set(other.edges):
                return False
            if set(self.node_props) != set(other.node_props):
                return False
            if loads(dumps(self.edge_props)) != loads(dumps(other.edge_props)):
                return False
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

    def __eq__(self, other):
        """
        defines the equal operator for the graph class
        """
        if isinstance(other, Edge):
            return (self.ending_node_id == other.ending_node_id and
                    self.originating_node_id == other.originating_node_id)
        else:
            return NotImplemented
