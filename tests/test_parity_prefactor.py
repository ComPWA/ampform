from typing import NamedTuple

import pytest
import qrules

from ampform import get_builder


class Input(NamedTuple):
    initial_state: list
    final_state: list
    intermediate_states: list
    final_state_grouping: list


@pytest.mark.parametrize(
    ("test_input", "parameter_count"),
    [
        (
            Input(
                initial_state=[("Lambda(c)+", [0.5])],
                final_state=["p", "K-", "pi+"],
                intermediate_states=["Lambda(1405)"],
                final_state_grouping=[],
            ),
            2,
        ),
        (
            Input(
                initial_state=[("Lambda(c)+", [0.5])],
                final_state=["p", "K-", "pi+"],
                intermediate_states=["Delta(1232)++"],
                final_state_grouping=[],
            ),
            2,
        ),
        (
            Input(
                initial_state=[("Lambda(c)+", [0.5])],
                final_state=["p", "K-", "pi+"],
                intermediate_states=["K*(892)0"],
                final_state_grouping=[],
            ),
            4,
        ),
    ],
)
def test_parity_amplitude_coupling(
    test_input: Input,
    parameter_count: int,
) -> None:
    stm = qrules.StateTransitionManager(
        test_input.initial_state,
        test_input.final_state,
        allowed_intermediate_particles=test_input.intermediate_states,
        number_of_threads=1,
    )
    problem_sets = stm.create_problem_sets()
    reaction = stm.find_solutions(problem_sets)

    model_builder = get_builder(reaction)
    amplitude_model = model_builder.generate()
    assert len(amplitude_model.parameter_defaults) == parameter_count
