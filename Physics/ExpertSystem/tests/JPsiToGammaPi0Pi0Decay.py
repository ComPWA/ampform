""" sample script for the testing purposes using the decay
    JPsi -> gamma pi0 pi0
"""
import logging

from expertsystem.ui.system_control import (
    StateTransitionManager, InteractionTypes)

from expertsystem.amplitude.helicitydecay import (
    HelicityDecayAmplitudeGeneratorXML)

logging.basicConfig(level=logging.INFO)

# initialize the graph edges (initial and final state)
initial_state = [("J/psi", [-1, 1])]
final_state = [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])]

tbd_manager = StateTransitionManager(initial_state, final_state,
                                     ['f0', 'f2', 'omega'])
#tbd_manager.number_of_threads = 1
tbd_manager.set_interaction_settings(
    [InteractionTypes.Strong, InteractionTypes.EM])
graph_interaction_settings_groups = tbd_manager.prepare_graphs()

(solutions, violated_rules) = tbd_manager.find_solutions(
    graph_interaction_settings_groups)

print("found " + str(len(solutions)) + " solutions!")

print("intermediate states:")
for g in solutions:
    print(g.edge_props[1]['@Name'])

xml_generator = HelicityDecayAmplitudeGeneratorXML()
xml_generator.generate(solutions)
xml_generator.write_to_file('JPsiToGammaPi0Pi0.xml')
