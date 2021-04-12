# pylint: disable=redefined-outer-name
import logging
from typing import Dict

import numpy as np
import pytest
import qrules as q
from qrules import ParticleCollection, Result, load_default_particles

from ampform import get_builder
from ampform.data import EventCollection
from ampform.dynamics.builder import create_relativistic_breit_wigner_with_ff
from ampform.helicity import HelicityModel

logging.getLogger().setLevel(level=logging.ERROR)


@pytest.fixture(scope="session")
def particle_database() -> ParticleCollection:
    return load_default_particles()


@pytest.fixture(scope="session")
def output_dir(pytestconfig) -> str:
    return f"{pytestconfig.rootpath}/tests/output/"


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_canonical_solutions() -> Result:
    return q.generate_transitions(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types="strong only",
        formalism_type="canonical-helicity",
    )


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_helicity_solutions() -> Result:
    return q.generate_transitions(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types="strong only",
        formalism_type="helicity",
    )


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_canonical_amplitude_model(
    jpsi_to_gamma_pi_pi_canonical_solutions: Result,
) -> HelicityModel:
    return __create_model(jpsi_to_gamma_pi_pi_canonical_solutions)


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_helicity_amplitude_model(
    jpsi_to_gamma_pi_pi_helicity_solutions: Result,
) -> HelicityModel:
    return __create_model(jpsi_to_gamma_pi_pi_helicity_solutions)


def __create_model(result: Result) -> HelicityModel:
    model_builder = get_builder(result)
    for name in result.get_intermediate_particles().names:
        model_builder.set_dynamics(
            name, create_relativistic_breit_wigner_with_ff
        )
    return model_builder.generate()


# https://github.com/ComPWA/tensorwaves/blob/3d0ec44/tests/physics/helicity_formalism/test_helicity_angles.py#L61-L98
@pytest.fixture(scope="session")
def data_sample_np() -> EventCollection:
    return EventCollection(
        {
            0: np.array(  # pi0
                [
                    (1.35527, 0.514208, -0.184219, 1.23296),
                    (0.841933, 0.0727385, -0.0528868, 0.826163),
                    (0.550927, -0.162529, 0.29976, -0.411133),
                    (0.425195, 0.0486171, 0.151922, 0.370309),
                    (0.186869, -0.0555915, -0.100214, -0.0597338),
                    (1.26375, 0.238921, 0.266712, -1.20442),
                    (0.737698, 0.450724, -0.439515, -0.360076),
                    (0.965809, 0.552298, 0.440006, 0.644927),
                    (0.397113, -0.248155, -0.158587, -0.229673),
                    (1.38955, 1.33491, 0.358535, 0.0457548),
                ]
            ),
            1: np.array(  # gamma
                [
                    (0.755744, -0.305812, 0.284, -0.630057),
                    (1.02861, 0.784483, 0.614347, -0.255334),
                    (0.356875, -0.20767, 0.272796, 0.0990739),
                    (0.70757, 0.404557, 0.510467, -0.276426),
                    (0.953902, 0.47713, 0.284575, -0.775431),
                    (0.220732, -0.204775, -0.0197981, 0.0799868),
                    (0.734602, 0.00590727, 0.709346, -0.190877),
                    (0.607787, 0.329157, -0.431973, 0.272873),
                    (0.626325, -0.201436, -0.534829, 0.256253),
                    (0.386432, -0.196357, 0.00211926, -0.33282),
                ]
            ),
            2: np.array(  # pi0
                [
                    (0.208274, -0.061663, -0.0211864, 0.144596),
                    (0.461193, -0.243319, -0.283044, -0.234866),
                    (1.03294, 0.82872, -0.0465425, -0.599834),
                    (0.752466, 0.263003, -0.089236, 0.686187),
                    (0.746588, 0.656892, -0.107848, 0.309898),
                    (0.692537, 0.521569, -0.0448683, 0.43283),
                    (0.865147, -0.517582, -0.676002, -0.0734335),
                    (1.35759, -0.975278, -0.0207817, -0.934467),
                    (0.852141, -0.41665, 0.237646, 0.691269),
                    (0.616162, -0.464203, -0.358114, 0.13307),
                ]
            ),
            3: np.array(  # pi0
                [
                    (0.777613, -0.146733, -0.0785946, -0.747499),
                    (0.765168, -0.613903, -0.278416, -0.335962),
                    (1.15616, -0.458522, -0.526014, 0.911894),
                    (1.21167, -0.716177, -0.573154, -0.780069),
                    (1.20954, -1.07843, -0.0765127, 0.525267),
                    (0.919879, -0.555715, -0.202046, 0.691605),
                    (0.759452, 0.0609506, 0.406171, 0.624387),
                    (0.165716, 0.0938229, 0.012748, 0.0166676),
                    (1.22132, 0.866241, 0.455769, -0.717849),
                    (0.704759, -0.674348, -0.0025409, 0.153994),
                ]
            ),
        }
    )


@pytest.fixture(scope="session")
def data_sample(data_sample_np: Dict[int, np.ndarray]) -> EventCollection:
    return EventCollection(data_sample_np)
