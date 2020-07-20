import json
import logging
from os.path import dirname, realpath

import pytest

import yaml

from expertsystem.amplitude.canonical_decay import CanonicalAmplitudeGenerator
from expertsystem.ui.system_control import (
    InteractionTypes,
    StateTransitionManager,
)

logging.basicConfig(level=logging.ERROR)

SCRIPT_PATH = dirname(realpath(__file__))


def create_amplitude_generator():
    tbd_manager = StateTransitionManager(
        initial_state=[("J/psi", [-1, 1])],
        final_state=[("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
        allowed_intermediate_particles=["f0"],
        formalism_type="canonical-helicity",
    )
    tbd_manager.set_allowed_interaction_types([InteractionTypes.EM])
    graph_interaction_settings_groups = tbd_manager.prepare_graphs()
    solutions, _ = tbd_manager.find_solutions(
        graph_interaction_settings_groups
    )

    amplitude_generator = CanonicalAmplitudeGenerator()
    amplitude_generator.generate(solutions)
    return amplitude_generator


def write_load_yaml() -> dict:
    amplitude_generator = create_amplitude_generator()
    output_filename = "JPsiToGammaPi0Pi0_cano_recipe.yml"
    amplitude_generator.write_to_file(output_filename)
    with open(output_filename, "rb") as input_file:
        imported_dict = yaml.load(input_file, Loader=yaml.FullLoader)
    return imported_dict


def equalize_dict(input_dict):
    output_dict = json.loads(json.dumps(input_dict, sort_keys=True))
    return output_dict


class TestCanonicalAmplitudeGeneratorYAML:
    amplitude_generator = create_amplitude_generator()
    imported_dict = write_load_yaml()

    def test_not_implemented_writer(self):
        with pytest.raises(NotImplementedError):
            self.amplitude_generator.write_to_file("JPsiToGammaPi0Pi0.csv")

    def test_particle_section(self):
        particle_list = self.imported_dict["ParticleList"]
        gamma = particle_list["gamma"]
        assert gamma["PID"] == 22
        assert gamma["Mass"] == 0.0
        gamma_qns = gamma["QuantumNumbers"]
        assert gamma_qns["CParity"] == -1
        f0_980 = particle_list["f0(980)"]
        assert f0_980["Width"] == 0.07
        pi0_qns = particle_list["pi0"]["QuantumNumbers"]
        assert pi0_qns["IsoSpin"]["Value"] == 1

    def test_parameter_section(self):
        parameter_list = self.imported_dict["Parameters"]
        assert len(parameter_list) == 9
        for parameter in parameter_list:
            assert "Name" in parameter
            assert "Value" in parameter

    def test_clebsch_gordan(self):
        strength_intensity = self.imported_dict["Intensity"]
        normalized_intensity = strength_intensity["Intensity"]
        incoherent_intensity = normalized_intensity["Intensity"]
        coherent_intensity = incoherent_intensity["Intensities"][0]
        coefficient_amplitude = coherent_intensity["Amplitudes"][0]
        sequential_amplitude = coefficient_amplitude["Amplitude"]
        helicity_decay = sequential_amplitude["Amplitudes"][0]
        canonical_sum = helicity_decay["Canonical"]
        assert list(canonical_sum) == ["LS", "s2s3"]
        s2s3 = canonical_sum["s2s3"]["ClebschGordan"]
        assert list(s2s3) == ["J", "M", "j1", "m1", "j2", "m2"]
        assert s2s3["J"] == 1.0


if __name__ == "__main__":
    create_amplitude_generator()
