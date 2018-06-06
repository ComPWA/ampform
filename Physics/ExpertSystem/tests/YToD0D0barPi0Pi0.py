""" sample script for the testing purposes using the decay
    Y -> D*0 D*0bar -> D0 D0bar pi0 pi0
"""
import logging

from expertsystem.ui.system_control import (
    StateTransitionManager, InteractionTypes
)
from expertsystem.amplitude.canonicaldecay import (
    CanonicalDecayAmplitudeGeneratorXML
)

logging.basicConfig(level=logging.INFO)

# initialize the graph edges (intial and final state)
initial_state = [("Y", [-1, 1])]
final_state = [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])]

tbd_manager = StateTransitionManager(initial_state, final_state, ['D*'])

tbd_manager.set_interaction_settings([InteractionTypes.Strong])
tbd_manager.add_final_state_grouping([['D0', 'pi0'], ['D0bar', 'pi0']])
tbd_manager.number_of_threads = 1

graph_node_setting_pairs = tbd_manager.prepare_graphs()
print(graph_node_setting_pairs)
(solutions, violated_rules) = tbd_manager.find_solutions(
    graph_node_setting_pairs)

print("found " + str(len(solutions)) + " solutions!")

for g in solutions:
    # print(g.node_props[0])
    # print(g.node_props[1])
    print(g.edge_props[1]['@Name'])

xml_generator = CanonicalDecayAmplitudeGeneratorXML()
xml_generator.generate(solutions)
xml_generator.write_to_file('YToD0D0barPi0Pi0.xml')
