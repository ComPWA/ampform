import pytest

import expertsystem as es
from expertsystem.reaction.combinatorics import (
    _create_edge_id_particle_mapping,
)


@pytest.mark.parametrize(
    "allowed_intermediate_particles, number_of_solutions",
    [
        (["f(0)(1500)"], 4),
        (["f(0)(980)", "f(0)(1500)"], 8),
        (["f(2)(1270)"], 12),
        (["omega(782)"], 8),
        (
            [
                "f(0)(980)",
                "f(0)(1500)",
                "f(2)(1270)",
                "f(2)(1950)",
                "omega(782)",
            ],
            40,
        ),
    ],
)
@pytest.mark.slow
def test_number_of_solutions(
    particle_database, allowed_intermediate_particles, number_of_solutions
):
    result = es.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        particles=particle_database,
        allowed_interaction_types="strong and EM",
        allowed_intermediate_particles=allowed_intermediate_particles,
        number_of_threads=1,
    )
    assert len(result.solutions) == number_of_solutions
    assert result.get_intermediate_particles().names == set(
        allowed_intermediate_particles
    )


def test_id_to_particle_mappings(particle_database):
    result = es.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        particles=particle_database,
        allowed_interaction_types="strong",
        allowed_intermediate_particles=["f(0)(980)"],
        number_of_threads=1,
    )
    assert len(result.solutions) == 4
    ref_mapping_fs = _create_edge_id_particle_mapping(
        result.solutions[0], result.solutions[0].get_final_state_edge_ids()
    )
    ref_mapping_is = _create_edge_id_particle_mapping(
        result.solutions[0], result.solutions[0].get_initial_state_edge_ids()
    )
    for solution in result.solutions[1:]:
        assert ref_mapping_fs == _create_edge_id_particle_mapping(
            solution, solution.get_final_state_edge_ids()
        )
        assert ref_mapping_is == _create_edge_id_particle_mapping(
            solution, solution.get_initial_state_edge_ids()
        )
