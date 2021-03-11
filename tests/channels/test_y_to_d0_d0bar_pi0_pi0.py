import pytest

import expertsystem as es
from expertsystem.reaction import InteractionTypes, StateTransitionManager


@pytest.mark.parametrize(
    "formalism_type, n_solutions",
    [
        ("helicity", 14),
        ("canonical-helicity", 28),  # two different LS couplings 2*14 = 28
    ],
)
def test_simple(formalism_type, n_solutions, particle_database):
    result = es.generate_transitions(
        initial_state=[("Y(4260)", [-1, +1])],
        final_state=["D*(2007)0", "D*(2007)~0"],
        particles=particle_database,
        formalism_type=formalism_type,
        allowed_interaction_types="strong",
        number_of_threads=1,
    )
    assert len(result.transitions) == n_solutions
    model_builder = es.amplitude.get_builder(result)
    model = model_builder.generate()
    assert len(model.parameters) == 4


@pytest.mark.slow
@pytest.mark.parametrize(
    "formalism_type, n_solutions",
    [
        ("helicity", 14),
        ("canonical-helicity", 28),  # two different LS couplings 2*14 = 28
    ],
)
def test_full(formalism_type, n_solutions, particle_database):
    stm = StateTransitionManager(
        initial_state=[("Y(4260)", [-1, +1])],
        final_state=["D0", "D~0", "pi0", "pi0"],
        particles=particle_database,
        allowed_intermediate_particles=["D*"],
        formalism_type=formalism_type,
        number_of_threads=1,
    )
    stm.set_allowed_interaction_types([InteractionTypes.STRONG])
    stm.add_final_state_grouping([["D0", "pi0"], ["D~0", "pi0"]])
    problem_sets = stm.create_problem_sets()
    result = stm.find_solutions(problem_sets)
    assert len(result.transitions) == n_solutions
    model_builder = es.amplitude.get_builder(result)
    model = model_builder.generate()
    assert len(model.parameters) == 4
