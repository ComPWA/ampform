""" sample script for the testing purposes using the decay
    JPsi -> gamma pi0 pi0
"""
import logging

from core.ui.system_control import StateTransitionManager

# logging.basicConfig(level=logging.DEBUG)

# initialize the graph edges (intial and final state)
initial_state = [("D0", [0])]
final_state = [("K_S0", [0]), ("K+", [0]), ("K-", [0])]

tbd_manager = StateTransitionManager(initial_state, final_state,
                                     [], 'helicity', 'weak')

graph_node_setting_pairs = tbd_manager.prepare_graphs()
(solutions, violated_rules) = tbd_manager.find_solutions(graph_node_setting_pairs)

print("found " + str(len(solutions)) + " solutions!")

for g in solutions:
    # print(g.node_props[0])
    # print(g.node_props[1])
    print(g.edge_props[1]['@Name'])
