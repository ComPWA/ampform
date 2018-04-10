""" module topologybuilder
responsible for building all possible topologies base on basic user input
  -number of initial state particles
  -number of final state particles
"""

import copy
import itertools
import logging

from core.topology.graph import (
    StateTransitionGraph, InteractionNode, are_graphs_isomorphic)


class SimpleStateTransitionTopologyBuilder():
    """Simple topology builder. Recursively trys to add the interaction nodes
    to available open end edges/lines in all combinations until the number of
    open end lines matches the final state lines"""

    def __init__(self, interaction_node_set):
        if not isinstance(interaction_node_set, list):
            raise TypeError("interaction_node_set must be a list")
            if not all((isinstance(Item, InteractionNode)
                        for Item in interaction_node_set)):
                raise TypeError(
                    "interaction_node_set must be a list of InteractionNode"
                    " instances")
        self.interaction_node_set = interaction_node_set

    def build_graphs(self, number_of_intial_edges, number_of_final_edges):
        if not isinstance(number_of_intial_edges, int):
            raise TypeError("number_of_intial_edges must be an integer")
        if number_of_intial_edges < 1:
            raise ValueError("number_of_intial_edges has to be larger than 0")
        if not isinstance(number_of_final_edges, int):
            raise TypeError("number_of_final_edges must be an integer")
        if number_of_final_edges < 1:
            raise ValueError("number_of_final_edges has to be larger than 0")

        logging.info('building topology graphs...')
        # result list
        graph_tuple_list = []
        # create seed graph
        seed_graph = StateTransitionGraph()
        current_open_end_edges = list(range(number_of_intial_edges))
        seed_graph.add_edges(current_open_end_edges)
        extendable_graph_list = [(seed_graph, current_open_end_edges)]

        while extendable_graph_list:
            active_graph_list = extendable_graph_list
            extendable_graph_list = []
            for active_graph in active_graph_list:
                # check if finished
                if (len(active_graph[1]) == number_of_final_edges
                        and len(active_graph[0].nodes) > 0):
                    if active_graph[0].verify():
                        graph_tuple_list.append(active_graph)
                    continue

                extendable_graph_list.extend(self.extend_graph(active_graph))

            # check if two topologies are the same
            for graph_index1, graph_index2 in itertools.combinations(
                    range(len(extendable_graph_list)), 2):
                if are_graphs_isomorphic(
                        extendable_graph_list[graph_index1][0],
                        extendable_graph_list[graph_index2][0]):
                    extendable_graph_list.remove(
                        extendable_graph_list[graph_index2])

        logging.info('finished building topology graphs...')
        # strip the current open end edges list from the result graph tuples
        result_graph_list = []
        for graph_tuple in graph_tuple_list:
            result_graph_list.append(graph_tuple[0])
        return result_graph_list

    def extend_graph(self, graph):
        extended_graph_list = []

        current_open_end_edges = graph[1]

        # Try to extend the graph with interaction nodes
        # that have equal or less ingoing lines than active lines
        for interaction_node in self.interaction_node_set:
            if (interaction_node.number_of_ingoing_edges
                    <= len(current_open_end_edges)):
                # make all combinations
                combis = list(itertools.combinations(
                    current_open_end_edges,
                    interaction_node.number_of_ingoing_edges))
                # remove all combinations that originate from the same nodes
                for comb1, comb2 in itertools.combinations(combis, 2):
                    if (graph[0].get_originating_node_list(comb1)
                            == graph[0].get_originating_node_list(comb2)):
                        combis.remove(comb2)

                for combi in combis:
                    new_graph = attach_node_to_edges(
                        graph, interaction_node, combi)
                    extended_graph_list.append(new_graph)

        return extended_graph_list


def attach_node_to_edges(graph, interaction_node, ingoing_edge_ids):
    temp_graph = copy.deepcopy(graph[0])
    new_open_end_lines = copy.deepcopy(graph[1])

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
        range(new_edge_start_id,
              new_edge_start_id + interaction_node.number_of_outgoing_edges))
    temp_graph.add_edges(new_edge_ids)
    temp_graph.attach_edges_to_node_outgoing(new_edge_ids, new_node_id)
    for edge_id in new_edge_ids:
        new_open_end_lines.append(edge_id)

    return (temp_graph, new_open_end_lines)
