""" sample script for the testing purposes using the decay
    Y -> D*0 D*0bar -> D0 D0bar pi0 pi0
"""

import logging

import pytest

from expertsystem.amplitude.canonicaldecay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicitydecay import HelicityAmplitudeGenerator
from expertsystem.state.particle import (
    InteractionQuantumNumberNames,
    create_spin_domain,
)
from expertsystem.ui.default_settings import (
    create_default_interaction_settings,
)
from expertsystem.ui.system_control import (
    InteractionTypes,
    StateTransitionManager,
    change_qn_domain,
)

logging.basicConfig(level=logging.INFO)


def test_script_simple():
    # initialize the graph edges (initial and final state)
    initial_state = [("Y", [-1, 1])]
    final_state = [("D*(2007)0", [-1, 0, 1]), ("D*(2007)0bar", [-1, 0, 1])]

    # because the amount of solutions is too big we change the default domains
    formalism_type = "canonical-helicity"
    int_settings = create_default_interaction_settings(formalism_type)
    change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.L,
        create_spin_domain([0, 1, 2, 3], True),
    )
    change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.S,
        create_spin_domain([0, 1, 2], True),
    )

    tbd_manager = StateTransitionManager(
        initial_state,
        final_state,
        ["D*"],
        interaction_type_settings=int_settings,
        formalism_type=formalism_type,
    )

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    # tbd_manager.add_final_state_grouping([['D0', 'pi0'], ['D0bar', 'pi0']])
    tbd_manager.number_of_threads = 2

    graph_node_setting_pairs = tbd_manager.prepare_graphs()

    solutions, _ = tbd_manager.find_solutions(graph_node_setting_pairs)

    print("found " + str(len(solutions)) + " solutions!")

    canonical_xml_generator = CanonicalAmplitudeGenerator()
    canonical_xml_generator.generate(solutions)

    # because the amount of solutions is too big we change the default domains
    formalism_type = "helicity"
    int_settings = create_default_interaction_settings(formalism_type)
    change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.L,
        create_spin_domain([0, 1, 2, 3], True),
    )
    change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.S,
        create_spin_domain([0, 1, 2], True),
    )

    tbd_manager = StateTransitionManager(
        initial_state,
        final_state,
        ["D*"],
        interaction_type_settings=int_settings,
        formalism_type=formalism_type,
    )

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    tbd_manager.number_of_threads = 2

    graph_node_setting_pairs = tbd_manager.prepare_graphs()
    solutions, _ = tbd_manager.find_solutions(graph_node_setting_pairs)
    print("found " + str(len(solutions)) + " solutions!")

    helicity_xml_generator = HelicityAmplitudeGenerator()
    helicity_xml_generator.generate(solutions)

    assert len(helicity_xml_generator.get_fit_parameters()) == len(
        canonical_xml_generator.get_fit_parameters()
    )


@pytest.mark.slow
def test_script_full():
    # initialize the graph edges (initial and final state)
    initial_state = [("Y", [-1, 1])]
    final_state = ["D0", "D0bar", "pi0", "pi0"]

    # because the amount of solutions is too big we change the default domains
    formalism_type = "canonical-helicity"
    int_settings = create_default_interaction_settings(formalism_type)
    change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.L,
        create_spin_domain([0, 1, 2, 3], True),
    )
    change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.S,
        create_spin_domain([0, 1, 2], True),
    )

    tbd_manager = StateTransitionManager(
        initial_state,
        final_state,
        ["D*"],
        interaction_type_settings=int_settings,
        formalism_type=formalism_type,
    )

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    tbd_manager.add_final_state_grouping([["D0", "pi0"], ["D0bar", "pi0"]])
    tbd_manager.number_of_threads = 2

    graph_node_setting_pairs = tbd_manager.prepare_graphs()

    solutions, _ = tbd_manager.find_solutions(graph_node_setting_pairs)

    print("found " + str(len(solutions)) + " solutions!")

    canonical_xml_generator = CanonicalAmplitudeGenerator()
    canonical_xml_generator.generate(solutions)

    # because the amount of solutions is too big we change the default domains
    formalism_type = "helicity"
    int_settings = create_default_interaction_settings(formalism_type)
    change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.L,
        create_spin_domain([0, 1, 2, 3], True),
    )
    change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.S,
        create_spin_domain([0, 1, 2], True),
    )

    tbd_manager = StateTransitionManager(
        initial_state,
        final_state,
        ["D*"],
        interaction_type_settings=int_settings,
        formalism_type=formalism_type,
    )

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    tbd_manager.add_final_state_grouping([["D0", "pi0"], ["D0bar", "pi0"]])
    tbd_manager.number_of_threads = 2

    graph_node_setting_pairs = tbd_manager.prepare_graphs()
    solutions, _ = tbd_manager.find_solutions(graph_node_setting_pairs)
    print("found " + str(len(solutions)) + " solutions!")

    helicity_xml_generator = HelicityAmplitudeGenerator()
    helicity_xml_generator.generate(solutions)

    assert len(helicity_xml_generator.get_fit_parameters()) == len(
        canonical_xml_generator.get_fit_parameters()
    )


if __name__ == "__main__":
    test_script_simple()
    test_script_full()
