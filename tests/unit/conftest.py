# pylint: disable=redefined-outer-name
import logging

import pytest

from expertsystem.amplitude.canonical_decay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.ui import InteractionTypes, StateTransitionManager

logging.basicConfig(level=logging.ERROR)


@pytest.fixture(scope="module")
def jpsi_to_gamma_pi_pi_canonical_solutions():
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        formalism_type="canonical-helicity",
    )
    stm.set_allowed_interaction_types([InteractionTypes.EM])
    graph_interaction_settings_groups = stm.prepare_graphs()
    result = stm.find_solutions(graph_interaction_settings_groups)
    return result.solutions


@pytest.fixture(scope="module")
def jpsi_to_gamma_pi_pi_helicity_solutions():
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
    )
    stm.set_allowed_interaction_types(
        [InteractionTypes.Strong, InteractionTypes.EM]
    )
    graph_interaction_settings_groups = stm.prepare_graphs()
    result = stm.find_solutions(graph_interaction_settings_groups)
    return result.solutions


@pytest.fixture(scope="module")
def jpsi_to_gamma_pi_pi_canonical_amplitude_model(
    jpsi_to_gamma_pi_pi_canonical_solutions,
) -> AmplitudeModel:
    amplitude_generator = CanonicalAmplitudeGenerator()
    return amplitude_generator.generate(
        jpsi_to_gamma_pi_pi_canonical_solutions
    )


@pytest.fixture(scope="module")
def jpsi_to_gamma_pi_pi_helicity_amplitude_model(
    jpsi_to_gamma_pi_pi_helicity_solutions,
) -> AmplitudeModel:
    amplitude_generator = HelicityAmplitudeGenerator()
    return amplitude_generator.generate(jpsi_to_gamma_pi_pi_helicity_solutions)
