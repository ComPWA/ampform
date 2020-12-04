# pylint: disable=redefined-outer-name
import logging

import pytest

import expertsystem as es
from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.reaction import Result

logging.basicConfig(level=logging.ERROR)


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_canonical_solutions() -> Result:
    return es.reaction.generate(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types="strong only",
        formalism_type="canonical-helicity",
    )


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_helicity_solutions() -> Result:
    return es.reaction.generate(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types="strong only",
        formalism_type="helicity",
    )


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_canonical_amplitude_model(
    jpsi_to_gamma_pi_pi_canonical_solutions: Result,
) -> AmplitudeModel:
    return __create_model(jpsi_to_gamma_pi_pi_canonical_solutions)


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_helicity_amplitude_model(
    jpsi_to_gamma_pi_pi_helicity_solutions: Result,
) -> AmplitudeModel:
    return __create_model(jpsi_to_gamma_pi_pi_helicity_solutions)


def __create_model(result: Result) -> AmplitudeModel:
    model = es.amplitude.generate(result)
    for name in result.get_intermediate_particles().names:
        model.dynamics.set_breit_wigner(name)
    return model
