from typing import NamedTuple, Tuple

import pytest

import expertsystem as es
from expertsystem.reaction import InteractionTypes, StateTransitionManager


class Input(NamedTuple):
    initial_state: list
    final_state: list
    intermediate_states: list
    final_state_grouping: list


@pytest.mark.parametrize(
    "test_input, ingoing_state, related_component_names, relative_parity_prefactor",
    [
        (
            Input(
                [("J/psi(1S)", [1])],
                [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
                ["f(0)(980)"],
                ["pi0", "pi0"],
            ),
            "J/psi(1S)",
            (
                "J/psi(1S)_1_to_f(0)(980)_0+gamma_1;f(0)(980)_0_to_pi0_0+pi0_0;",
                "J/psi(1S)_1_to_f(0)(980)_0+gamma_-1;f(0)(980)_0_to_pi0_0+pi0_0;",
            ),
            1.0,
        ),
        (
            Input(
                [("J/psi(1S)", [1])],
                [("pi0", [0]), ("pi+", [0]), ("pi-", [0])],
                ["rho(770)"],
                ["pi+", "pi-"],
            ),
            "J/psi(1S)",
            (
                "J/psi(1S)_1_to_pi0_0+rho(770)0_1;rho(770)0_1_to_pi+_0+pi-_0;",
                "J/psi(1S)_1_to_pi0_0+rho(770)0_-1;rho(770)0_-1_to_pi+_0+pi-_0;",
            ),
            -1.0,
        ),
    ],
)
def test_parity_prefactor(
    test_input: Input,
    ingoing_state: str,
    related_component_names: Tuple[str, str],
    relative_parity_prefactor: float,
    output_dir,
) -> None:
    stm = StateTransitionManager(
        test_input.initial_state,
        test_input.final_state,
        allowed_intermediate_particles=test_input.intermediate_states,
        number_of_threads=1,
    )
    stm.add_final_state_grouping(test_input.final_state_grouping)
    stm.set_allowed_interaction_types([InteractionTypes.EM])
    problem_sets = stm.create_problem_sets()

    result = stm.find_solutions(problem_sets)

    for solution in result.solutions:
        in_edge = [
            edge_id
            for edge_id in solution.edges
            if solution.get_edge_props(edge_id)[0].name == ingoing_state
        ]
        assert len(in_edge) == 1
        node_id = solution.edges[in_edge[0]].ending_node_id

        assert isinstance(node_id, int)

        assert (
            relative_parity_prefactor
            == solution.get_node_props(node_id).parity_prefactor
        )

    amplitude_model = es.generate_amplitudes(result)
    es.io.write(
        instance=amplitude_model,
        filename=output_dir
        + f'amplitude_model_prefactor_{"-".join(test_input.intermediate_states)}.xml',
    )

    prefactor1 = extract_prefactor(amplitude_model, related_component_names[0])
    prefactor2 = extract_prefactor(amplitude_model, related_component_names[1])

    assert prefactor1 == relative_parity_prefactor * prefactor2


def extract_prefactor(node, coefficient_amplitude_name):
    if hasattr(node, "component"):
        if node.component == coefficient_amplitude_name:
            if hasattr(node, "prefactor") and node.prefactor is not None:
                return node.prefactor
            return 1.0
    if hasattr(node, "intensity"):
        return extract_prefactor(node.intensity, coefficient_amplitude_name)
    if hasattr(node, "intensities"):
        for amp in node.intensities:
            prefactor = extract_prefactor(amp, coefficient_amplitude_name)
            if prefactor is not None:
                return prefactor
    if hasattr(node, "amplitudes"):
        for amp in node.amplitudes:
            prefactor = extract_prefactor(amp, coefficient_amplitude_name)
            if prefactor is not None:
                return prefactor
    return None


@pytest.mark.parametrize(
    "test_input, parameter_count",
    [
        (
            Input(
                [("Lambda(c)+", [0.5])],
                ["p", "K-", "pi+"],
                ["Lambda(1405)"],
                [],
            ),
            5,
        ),
        (
            Input(
                [("Lambda(c)+", [0.5])],
                ["p", "K-", "pi+"],
                ["Delta(1232)++"],
                [],
            ),
            5,
        ),
        (
            Input(
                [("Lambda(c)+", [0.5])],
                ["p", "K-", "pi+"],
                ["K*(892)0"],
                [],
            ),
            9,
        ),
    ],
)
def test_parity_amplitude_coupling(
    test_input: Input,
    parameter_count: int,
) -> None:
    stm = StateTransitionManager(
        test_input.initial_state,
        test_input.final_state,
        allowed_intermediate_particles=test_input.intermediate_states,
        number_of_threads=1,
    )
    problem_sets = stm.create_problem_sets()
    result = stm.find_solutions(problem_sets)

    amplitude_model = es.generate_amplitudes(result)
    assert len(amplitude_model.parameters) == parameter_count
