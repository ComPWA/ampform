""" sample script for the testing purposes using the decay
    JPsi -> gamma pi0 pi0
"""
import logging

from expertsystem.ui.system_control import StateTransitionManager

from expertsystem.amplitude.helicitydecay import (
    HelicityDecayAmplitudeGeneratorXML)

logging.basicConfig(level=logging.DEBUG)

# initialize the graph edges (initial and final state)
initial_state = [("J/psi", [-1, 1])]
final_state = [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])]

tbd_manager = StateTransitionManager(initial_state, final_state, ['omega(782)'])

graph_node_setting_pairs = tbd_manager.prepare_graphs()
(solutions, violated_rules) = tbd_manager.find_solutions(
    graph_node_setting_pairs)

print("found " + str(len(solutions)) + " solutions!")

print("intermediate states:")
for g in solutions:
    print(g.edge_props[1]['@Name'])

xml_generator = HelicityDecayAmplitudeGeneratorXML(solutions)
xml_generator.write_to_xml('output.xml')
