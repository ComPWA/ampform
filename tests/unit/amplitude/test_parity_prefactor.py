from typing import NamedTuple, Tuple

import pytest

from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.state.particle import (
    InteractionQuantumNumberNames,
    get_interaction_property,
)
from expertsystem.ui import (
    InteractionTypes,
    StateTransitionManager,
)


class Input(NamedTuple):
    """Helper tuple for tests."""

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
        ),  # pylint: disable=too-many-locals
    ],
)
def test_parity_prefactor(
    test_input: Input,
    ingoing_state: str,
    related_component_names: Tuple[str, str],
    relative_parity_prefactor: float,
) -> None:
    stm = StateTransitionManager(
        test_input.initial_state,
        test_input.final_state,
        allowed_intermediate_particles=test_input.intermediate_states,
    )
    # stm.number_of_threads = 1
    stm.add_final_state_grouping(test_input.final_state_grouping)
    stm.set_allowed_interaction_types([InteractionTypes.EM])
    graph_interaction_settings_groups = stm.prepare_graphs()

    solutions, _ = stm.find_solutions(graph_interaction_settings_groups)

    for solution in solutions:
        in_edge = [
            k
            for k, v in solution.edge_props.items()
            if v["Name"] == ingoing_state
        ]
        assert len(in_edge) == 1
        node_id = solution.edges[in_edge[0]].ending_node_id

        assert isinstance(node_id, int)

        prefactor = get_interaction_property(
            solution.node_props[node_id],
            InteractionQuantumNumberNames.ParityPrefactor,
        )

        assert relative_parity_prefactor == prefactor

    amp_gen = HelicityAmplitudeGenerator()
    amp_gen.generate(solutions)
    amp_dict = amp_gen.helicity_amplitudes

    prefactor1 = extract_prefactor(amp_dict, related_component_names[0])
    prefactor2 = extract_prefactor(amp_dict, related_component_names[1])

    assert prefactor1 == relative_parity_prefactor * prefactor2


def extract_prefactor(amplitude_dict, coefficient_amplitude_name):
    dict_element_stack = [amplitude_dict]

    while dict_element_stack:
        element = dict_element_stack.pop()

        if (
            "Component" in element
            and element["Component"] == coefficient_amplitude_name
        ):
            # we found what we are looking for, extract the prefactor
            if "PreFactor" in element:
                return element["PreFactor"]["Real"]
            return 1.0

        if "Intensity" in element:
            sub_intensity = element["Intensity"]
            if isinstance(sub_intensity, dict):
                dict_element_stack.append(sub_intensity)
            elif isinstance(sub_intensity, list):
                for sub_intensity_dict in sub_intensity:
                    dict_element_stack.append(sub_intensity_dict)
        elif "Amplitude" in element:
            sub_amplitude = element["Amplitude"]
            if isinstance(sub_amplitude, dict):
                dict_element_stack.append(sub_amplitude)
            elif isinstance(sub_amplitude, list):
                for sub_amplitude_dict in sub_amplitude:
                    dict_element_stack.append(sub_amplitude_dict)

    return None
