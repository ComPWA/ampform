""" sample script for the testing purposes using the decay
    JPsi -> pi0 pi+ pi-
"""
import logging

from expertsystem.ui.system_control import (
    StateTransitionManager, InteractionTypes)

from expertsystem.amplitude.helicitydecay import (
    HelicityDecayAmplitudeGeneratorXML)

logging.basicConfig(level=logging.INFO)

# initialize the graph edges (initial and final state)
initial_state = [("J/psi", [1])]
final_state = [("pi0", [0]), ("pi+", [0]), ("pi-", [0])]

tbd_manager = StateTransitionManager(initial_state, final_state,
                                     ['rho'])
#tbd_manager.number_of_threads = 1
tbd_manager.add_final_state_grouping(['pi+', 'pi-'])
tbd_manager.set_allowed_interaction_types(
    [InteractionTypes.EM])
graph_interaction_settings_groups = tbd_manager.prepare_graphs()

(solutions, violated_rules) = tbd_manager.find_solutions(
    graph_interaction_settings_groups)

print("found " + str(len(solutions)) + " solutions!")

print("intermediate states:")
for g in solutions:
    print(g.edge_props[1]['@Name'])

xml_generator = HelicityDecayAmplitudeGeneratorXML()
xml_generator.generate(solutions)
xml_generator.write_to_file('test.xml')
