import logging

from core.ui.system_control import (StateTransitionManager,
                                    HelicityDecayAmplitudeGeneratorXML)

logging.basicConfig(level=logging.INFO)

# initialize the graph edges (initial and final state)
initial_state = [("EpEm", [-1, 1])]
final_state = [("Chic1", [-1, 1]), ("pi0", [0]), ("pi0", [0])]

tbd_manager = StateTransitionManager(initial_state, final_state)
tbd_manager.add_final_state_grouping([["Chic1", "pi0"]])

graph_node_setting_pairs = tbd_manager.prepare_graphs()
(solutions, violated_rules) = tbd_manager.find_solutions(graph_node_setting_pairs)

print("found " + str(len(solutions)) + " solutions!")

print("intermediate states:")
for g in solutions:
    print(g.edge_props[1]['@Name'])
