""" sample script for the testing purposes using the decay
    Y(4260) -> D*0 D*0bar -> D0 D0bar pi0 pi0
"""

import logging
from typing import List

import pytest

from expertsystem.amplitude.canonical_decay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.nested_dicts import InteractionQuantumNumberNames
from expertsystem.state.properties import create_spin_domain
from expertsystem.ui import (
    InteractionTypes,
    StateDefinition,
    StateTransitionManager,
)
from expertsystem.ui._default_settings import (
    create_default_interaction_settings,
)
from expertsystem.ui._system_control import _change_qn_domain

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
    int_settings = create_default_interaction_settings(formalism_type)
    _change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.L,
        create_spin_domain([0, 1, 2, 3], True),
    )
    _change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.S,
        create_spin_domain([0, 1, 2], True),
    )

    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        allowed_intermediate_particles=["D*"],
        interaction_type_settings=int_settings,
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
    int_settings = create_default_interaction_settings(formalism_type)
    _change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.L,
        create_spin_domain([0, 1, 2, 3], True),
    )
    _change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.S,
        create_spin_domain([0, 1, 2], True),
    )

    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        ["D*"],
        interaction_type_settings=int_settings,
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


@pytest.mark.slow
def test_script_full(particle_database):
    # initialize the graph edges (initial and final state)
    initial_state: List[StateDefinition] = [("Y(4260)", [-1, 1])]
    final_state: List[StateDefinition] = ["D0", "D~0", "pi0", "pi0"]

    # because the amount of solutions is too big we change the default domains
    formalism_type = "canonical-helicity"
    int_settings = create_default_interaction_settings(formalism_type)
    _change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.L,
        create_spin_domain([0, 1, 2, 3], True),
    )
    _change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.S,
        create_spin_domain([0, 1, 2], True),
    )

    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        ["D*"],
        interaction_type_settings=int_settings,
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
    int_settings = create_default_interaction_settings(formalism_type)
    _change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.L,
        create_spin_domain([0, 1, 2, 3], True),
    )
    _change_qn_domain(
        int_settings[InteractionTypes.Strong],
        InteractionQuantumNumberNames.S,
        create_spin_domain([0, 1, 2], True),
    )

    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        ["D*"],
        interaction_type_settings=int_settings,
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
