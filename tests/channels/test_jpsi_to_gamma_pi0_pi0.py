""" sample script for the testing purposes using the decay
    JPsi -> gamma pi0 pi0
"""

import logging

import pytest

from expertsystem.ui import (
    InteractionTypes,
    StateTransitionManager,
)
from expertsystem.ui._system_control import _create_edge_id_particle_mapping


logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


@pytest.mark.parametrize(
    "allowed_intermediate_particles, number_of_solutions",
    [
        (["f(0)(1500)"], 4),
        (["f(0)(980)", "f(0)(1500)"], 8),
        (["f(2)(1270)"], 12),
        (["omega(782)"], 16),
        (
            [
                "f(0)(980)",
                "f(0)(1500)",
                "f(2)(1270)",
                "f(2)(1950)",
                "omega(782)",
            ],
            48,
        ),
    ],
)
@pytest.mark.slow
def test_number_of_solutions(
    allowed_intermediate_particles, number_of_solutions
):
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=[("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
        allowed_intermediate_particles=allowed_intermediate_particles,
        number_of_threads=1,
    )
    stm.set_allowed_interaction_types(
        [InteractionTypes.Strong, InteractionTypes.EM]
    )
    graph_interaction_settings_groups = stm.prepare_graphs()
    solutions, _ = stm.find_solutions(graph_interaction_settings_groups)
    assert len(solutions) == number_of_solutions


def test_id_to_particle_mappings():
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=[("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
        allowed_intermediate_particles=["f(0)(980)"],
        number_of_threads=1,
    )
    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    graph_interaction_settings_groups = stm.prepare_graphs()
    solutions, _ = stm.find_solutions(graph_interaction_settings_groups)
    assert len(solutions) == 4
    ref_mapping_fs = _create_edge_id_particle_mapping(
        solutions[0], "get_final_state_edges"
    )
    ref_mapping_is = _create_edge_id_particle_mapping(
        solutions[0], "get_initial_state_edges"
    )
    for solution in solutions[1:]:
        assert ref_mapping_fs == _create_edge_id_particle_mapping(
            solution, "get_final_state_edges"
        )
        assert ref_mapping_is == _create_edge_id_particle_mapping(
            solution, "get_initial_state_edges"
        )
