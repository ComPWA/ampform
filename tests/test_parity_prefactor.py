from typing import NamedTuple

import pytest
import qrules as q

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
                [("Lambda(c)+", [0.5])],
                ["p", "K-", "pi+"],
                ["Lambda(1405)"],
                [],
            ),
            2,
        ),
        (
            Input(
                [("Lambda(c)+", [0.5])],
                ["p", "K-", "pi+"],
                ["Delta(1232)++"],
                [],
            ),
            2,
        ),
        (
            Input(
                [("Lambda(c)+", [0.5])],
                ["p", "K-", "pi+"],
                ["K*(892)0"],
                [],
            ),
            4,
        ),
    ],
)
def test_parity_amplitude_coupling(
    test_input: Input,
    parameter_count: int,
) -> None:
    stm = q.StateTransitionManager(
        test_input.initial_state,
        test_input.final_state,
        allowed_intermediate_particles=test_input.intermediate_states,
        number_of_threads=1,
    )
    problem_sets = stm.create_problem_sets()
    result = stm.find_solutions(problem_sets)

    model_builder = get_builder(result)
    amplitude_model = model_builder.generate()
    assert len(amplitude_model.parameter_defaults) == parameter_count
