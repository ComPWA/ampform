""" sample script for the testing purposes using the decay
    D0 -> K_S0 K+ K-
"""
import logging

from pycompwa.expertsystem.ui.system_control import StateTransitionManager

from pycompwa.expertsystem.amplitude.helicitydecay import (
    HelicityAmplitudeGeneratorXML)


def test_script():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    # initialize the graph edges (intial and final state)
    initial_state = [("D0", [0])]
    final_state = [("K_S0", [0]), ("K+", [0]), ("K-", [0])]

    tbd_manager = StateTransitionManager(initial_state, final_state,
                                         ['a0', 'phi', 'a2(1320)-'])
    tbd_manager.number_of_threads = 2

    graph_interaction_settings_groups = tbd_manager.prepare_graphs()
    (solutions, violated_rules) = tbd_manager.find_solutions(
        graph_interaction_settings_groups)

    print("found " + str(len(solutions)) + " solutions!")
    assert len(solutions) == 5

    # print intermediate state particle names
    for g in solutions:
        print(g.edge_props[1]['@Name'])

    xml_generator = HelicityAmplitudeGeneratorXML()
    xml_generator.generate(solutions)
    xml_generator.write_to_file('D0ToKs0KpKm.xml')


if __name__ == '__main__':
    test_script()
