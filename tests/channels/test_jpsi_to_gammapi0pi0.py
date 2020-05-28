""" sample script for the testing purposes using the decay
    JPsi -> gamma pi0 pi0
"""

import logging

import pytest

from expertsystem.amplitude.helicitydecay import HelicityAmplitudeGeneratorXML
from expertsystem.topology.graph import (
    get_final_state_edges,
    get_initial_state_edges,
    get_intermediate_state_edges,
)
from expertsystem.ui.system_control import (
    InteractionTypes,
    StateTransitionManager,
    create_edge_id_particle_mapping,
)


@pytest.mark.slow
def test_script():
    logging.basicConfig(level=logging.INFO)
    # initialize the graph edges (initial and final state)
    initial_state = [("J/psi", [-1, 1])]
    final_state = [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])]

    tbd_manager = StateTransitionManager(
        initial_state, final_state, ["f0", "f2", "omega"]
    )
    tbd_manager.number_of_threads = 2
    tbd_manager.set_allowed_interaction_types(
        [InteractionTypes.Strong, InteractionTypes.EM]
    )
    graph_interaction_settings_groups = tbd_manager.prepare_graphs()

    solutions, _ = tbd_manager.find_solutions(
        graph_interaction_settings_groups
    )

    print("found " + str(len(solutions)) + " solutions!")
    assert len(solutions) == 48

    ref_mapping_fs = create_edge_id_particle_mapping(
        solutions[0], get_final_state_edges
    )
    ref_mapping_is = create_edge_id_particle_mapping(
        solutions[0], get_initial_state_edges
    )
    for solution in solutions[1:]:
        assert ref_mapping_fs == create_edge_id_particle_mapping(
            solution, get_final_state_edges
        )
        assert ref_mapping_is == create_edge_id_particle_mapping(
            solution, get_initial_state_edges
        )

    print("intermediate states:")
    intermediate_states = set()
    for solution in solutions:
        int_edge_id = get_intermediate_state_edges(solution)[0]
        intermediate_states.add(solution.edge_props[int_edge_id]["Name"])
    print(intermediate_states)

    xml_generator = HelicityAmplitudeGeneratorXML()
    xml_generator.generate(solutions)
    xml_generator.write_to_file("JPsiToGammaPi0Pi0.xml")


if __name__ == "__main__":
    test_script()
