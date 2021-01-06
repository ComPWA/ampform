# pylint: disable=redefined-outer-name
import json

import pytest
import yaml

from expertsystem import io
from expertsystem.amplitude.model import AmplitudeModel


@pytest.fixture(scope="session")
def imported_dict(
    output_dir,
    jpsi_to_gamma_pi_pi_canonical_amplitude_model: AmplitudeModel,
):
    output_filename = output_dir + "JPsiToGammaPi0Pi0_cano_recipe.yml"
    io.write(
        instance=jpsi_to_gamma_pi_pi_canonical_amplitude_model,
        filename=output_filename,
    )
    with open(output_filename, "rb") as input_file:
        loaded_dict = yaml.load(input_file, Loader=yaml.FullLoader)
    return loaded_dict


def equalize_dict(input_dict):
    output_dict = json.loads(json.dumps(input_dict, sort_keys=True))
    return output_dict


def test_not_implemented_writer(
    jpsi_to_gamma_pi_pi_canonical_amplitude_model: AmplitudeModel,
):
    with pytest.raises(NotImplementedError):
        io.write(
            instance=jpsi_to_gamma_pi_pi_canonical_amplitude_model,
            filename="JPsiToGammaPi0Pi0.csv",
        )


def test_particle_section(imported_dict):
    particle_list = imported_dict["ParticleList"]
    gamma = particle_list["gamma"]
    assert gamma["PID"] == 22
    assert gamma["Mass"] == 0.0
    gamma_qns = gamma["QuantumNumbers"]
    assert gamma_qns["CParity"] == -1
    f0_980 = particle_list["f(0)(980)"]
    assert f0_980["Width"] == 0.06
    pi0_qns = particle_list["pi0"]["QuantumNumbers"]
    assert pi0_qns["IsoSpin"]["Value"] == 1


def test_parameter_section(imported_dict):
    parameter_list = imported_dict["Parameters"]
    assert len(parameter_list) == 11
    for parameter in parameter_list:
        assert "Name" in parameter
        assert "Value" in parameter


def test_clebsch_gordan(imported_dict):
    incoherent_intensity = imported_dict["Intensity"]
    coherent_intensity = incoherent_intensity["Intensities"][0]
    coefficient_amplitude = coherent_intensity["Amplitudes"][0]
    sequential_amplitude = coefficient_amplitude["Amplitude"]
    helicity_decay = sequential_amplitude["Amplitudes"][0]
    canonical_sum = helicity_decay["Canonical"]
    assert list(canonical_sum) == ["LS", "s2s3"]
    s2s3 = canonical_sum["s2s3"]["ClebschGordan"]
    assert list(s2s3) == ["J", "M", "j1", "m1", "j2", "m2"]
    assert s2s3["J"] == 1.0
