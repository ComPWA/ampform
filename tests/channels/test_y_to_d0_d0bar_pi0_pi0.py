""" sample script for the testing purposes using the decay
    Y(4260) -> D*0 D*0bar -> D0 D0bar pi0 pi0
"""

import logging
from typing import List

import pytest

from expertsystem.amplitude.canonical_decay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.ui import (
    InteractionTypes,
    StateDefinition,
    StateTransitionManager,
)

logging.basicConfig(level=logging.INFO)


def test_script_simple(particle_database):
    # initialize the graph edges (initial and final state)
    initial_state: List[StateDefinition] = [("Y(4260)", [-1, 1])]
    final_state: List[StateDefinition] = [
        ("D*(2007)0", [-1, 0, 1]),
        ("D*(2007)~0", [-1, 0, 1]),
    ]

    # because the amount of solutions is too big we change the default domains
    formalism_type = "canonical-helicity"

    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        allowed_intermediate_particles=["D*"],
        formalism_type=formalism_type,
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    # stm.add_final_state_grouping([['D0', 'pi0'], ['D0bar', 'pi0']])
    stm.number_of_threads = 2

    graph_node_setting_pairs = stm.prepare_graphs()

    result = stm.find_solutions(graph_node_setting_pairs)

    print("found " + str(len(result.solutions)) + " solutions!")

    canonical_xml_generator = CanonicalAmplitudeGenerator()
    canonical_xml_generator.generate(result.solutions)

    # because the amount of solutions is too big we change the default domains
    formalism_type = "helicity"

    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        ["D*"],
        formalism_type=formalism_type,
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    stm.number_of_threads = 2

    graph_node_setting_pairs = stm.prepare_graphs()
    result = stm.find_solutions(graph_node_setting_pairs)
    print("found " + str(len(result.solutions)) + " solutions!")

    helicity_xml_generator = HelicityAmplitudeGenerator()
    helicity_xml_generator.generate(result.solutions)

    assert len(helicity_xml_generator.fit_parameters) == len(
        canonical_xml_generator.fit_parameters
    )


@pytest.mark.skip(
    reason="Test takes too long. Can be enabled again after Rule refactoring"
)
@pytest.mark.slow
def test_script_full(particle_database):
    # initialize the graph edges (initial and final state)
    initial_state: List[StateDefinition] = [("Y(4260)", [-1, 1])]
    final_state: List[StateDefinition] = ["D0", "D~0", "pi0", "pi0"]

    # because the amount of solutions is too big we change the default domains
    formalism_type = "canonical-helicity"

    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        ["D*"],
        formalism_type=formalism_type,
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    stm.add_final_state_grouping([["D0", "pi0"], ["D~0", "pi0"]])
    stm.number_of_threads = 2

    graph_node_setting_pairs = stm.prepare_graphs()

    result = stm.find_solutions(graph_node_setting_pairs)

    print("found " + str(len(result.solutions)) + " solutions!")

    canonical_xml_generator = CanonicalAmplitudeGenerator()
    canonical_xml_generator.generate(result.solutions)

    # because the amount of solutions is too big we change the default domains
    formalism_type = "helicity"

    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        ["D*"],
        formalism_type=formalism_type,
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    stm.add_final_state_grouping([["D0", "pi0"], ["D~0", "pi0"]])
    stm.number_of_threads = 2

    graph_node_setting_pairs = stm.prepare_graphs()
    result = stm.find_solutions(graph_node_setting_pairs)
    print("found " + str(len(result.solutions)) + " solutions!")

    helicity_xml_generator = HelicityAmplitudeGenerator()
    helicity_xml_generator.generate(result.solutions)

    assert len(helicity_xml_generator.fit_parameters) == len(
        canonical_xml_generator.fit_parameters
    )
