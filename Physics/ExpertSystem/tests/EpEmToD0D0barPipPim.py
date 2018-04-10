""" sample script for the testing purposes using the decay
    e+e- -> D0 D0bar pi+ pi-
"""
import logging

from expertsystem.ui.system_control import StateTransitionManager

logging.basicConfig(level=logging.INFO)

# initialize the graph edges (intial and final state)
initial_state = [("EpEm", [-1, 1])]
final_state = [("D0", [0]), ("D0bar", [0]), ("pi+", [0]), ("pi-", [0])]

tbd_manager = StateTransitionManager(initial_state, final_state)

graph_node_setting_pairs = tbd_manager.prepare_graphs()
(solutions, violated_rules) = tbd_manager.find_solutions(graph_node_setting_pairs)

print("found " + str(len(solutions)) + " solutions!")

for g in solutions:
    #print(g.node_props[0])
    #print(g.node_props[1])
    print(g.edge_props[1]['@Name'])