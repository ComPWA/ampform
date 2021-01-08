# pylint: disable=no-self-use,redefined-outer-name
import json
from os.path import dirname, realpath

import pytest
import yaml

from expertsystem import io
from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.particle import ParticleCollection

SCRIPT_PATH = dirname(realpath(__file__))


@pytest.fixture(scope="session")
def particle_selection(particle_database: ParticleCollection):
    selection = ParticleCollection()
    selection += particle_database.filter(lambda p: p.name.startswith("pi"))
    selection += particle_database.filter(lambda p: p.name.startswith("K"))
    selection += particle_database.filter(lambda p: p.name.startswith("D"))
    selection += particle_database.filter(lambda p: p.name.startswith("J/psi"))
    return selection


def test_not_implemented_errors(
    output_dir, particle_selection: ParticleCollection
):
    with pytest.raises(NotImplementedError):
        io.load_particle_collection(output_dir + __file__)
    with pytest.raises(NotImplementedError):
        io.write(particle_selection, output_dir + __file__)
    with pytest.raises(Exception):
        io.write(particle_selection, output_dir + "no_file_extension")
    with pytest.raises(NotImplementedError):
        io.write(666, output_dir + "wont_work_anyway.yml")


def test_serialization(output_dir, particle_selection: ParticleCollection):
    assert len(particle_selection) == 181
    io.write(particle_selection, output_dir + "particle_selection.yml")
    asdict = io.asdict(particle_selection)
    imported_collection = io.fromdict(asdict)
    assert isinstance(imported_collection, ParticleCollection)
    assert len(particle_selection) == len(imported_collection)
    for particle in particle_selection:
        exported = particle_selection[particle.name]
        imported = imported_collection[particle.name]
        assert imported == exported


class TestHelicityFormalism:
    @pytest.fixture(scope="session")
    def model(
        self, jpsi_to_gamma_pi_pi_helicity_amplitude_model: AmplitudeModel
    ):
        return jpsi_to_gamma_pi_pi_helicity_amplitude_model

    @pytest.fixture(scope="session")
    def imported_dict(self, output_dir, model: AmplitudeModel):
        output_filename = output_dir + "JPsiToGammaPi0Pi0_heli_recipe.yml"
        asdict = io.asdict(model)
        io.write(model, output_filename)
        return asdict

    @pytest.fixture(scope="session")
    def expected_dict(self) -> dict:
        expected_recipe_file = f"{SCRIPT_PATH}/expected_recipe.yml"
        with open(expected_recipe_file, "rb") as input_file:
            expected_recipe_dict = yaml.load(
                input_file, Loader=yaml.FullLoader
            )
        return expected_recipe_dict

    def test_recipe_validation(self, expected_dict):
        io.validate(expected_dict)

    def test_not_implemented_writer(
        self,
        jpsi_to_gamma_pi_pi_helicity_amplitude_model: AmplitudeModel,
    ):
        with pytest.raises(NotImplementedError):
            io.write(
                instance=jpsi_to_gamma_pi_pi_helicity_amplitude_model,
                filename="JPsiToGammaPi0Pi0.csv",
            )

    def test_create_recipe_dict(
        self,
        jpsi_to_gamma_pi_pi_helicity_amplitude_model: AmplitudeModel,
    ):
        assert len(jpsi_to_gamma_pi_pi_helicity_amplitude_model.particles) == 5
        assert len(jpsi_to_gamma_pi_pi_helicity_amplitude_model.dynamics) == 3

    def test_particle_section(self, imported_dict):
        particle_list = imported_dict.get("ParticleList", imported_dict)
        gamma = particle_list["gamma"]
        assert gamma["PID"] == 22
        assert gamma["Mass"] == 0.0
        gamma_qns = gamma["QuantumNumbers"]
        assert gamma_qns["Spin"] == 1
        assert gamma_qns["Charge"] == 0
        assert gamma_qns["Parity"] == -1
        assert gamma_qns["CParity"] == -1

        f0_980 = particle_list["f(0)(980)"]
        assert f0_980["Width"] == 0.06

        pi0_qns = particle_list["pi0"]["QuantumNumbers"]
        assert pi0_qns["IsoSpin"]["Value"] == 1
        assert pi0_qns["IsoSpin"]["Projection"] == 0

    def test_kinematics_section(self, imported_dict):
        kinematics = imported_dict["Kinematics"]
        initial_state = kinematics["InitialState"]
        final_state = kinematics["FinalState"]
        assert kinematics["Type"] == "Helicity"
        assert len(initial_state) == 1
        assert initial_state[0]["Particle"] == "J/psi(1S)"
        assert len(final_state) == 3

    def test_parameter_section(self, imported_dict):
        parameter_list = imported_dict["Parameters"]
        assert len(parameter_list) == 11
        for parameter in parameter_list:
            assert "Name" in parameter
            assert "Value" in parameter
            assert parameter.get("Fix", True)

    def test_dynamics_section(self, imported_dict):
        parameter_list: list = imported_dict["Parameters"]

        def get_parameter(parameter_name: str) -> dict:
            for par in parameter_list:
                name = par["Name"]
                if name == parameter_name:
                    return par
            raise LookupError(f'Could not find parameter  "{parameter_name}"')

        dynamics = imported_dict["Dynamics"]
        assert len(dynamics) == 3

        j_psi = dynamics["J/psi(1S)"]
        assert j_psi["Type"] == "NonDynamic"
        assert j_psi["FormFactor"]["Type"] == "BlattWeisskopf"
        assert (
            get_parameter(j_psi["FormFactor"]["MesonRadius"])["Value"] == 1.0
        )

        f0_980 = dynamics.get("f(0)(980)", None)
        if f0_980:
            assert f0_980["Type"] == "RelativisticBreitWigner"
            assert f0_980["FormFactor"]["Type"] == "BlattWeisskopf"
            assert (
                f0_980["FormFactor"]["MesonRadius"] == "MesonRadius_f(0)(980)"
            )
            assert (
                get_parameter(f0_980["FormFactor"]["MesonRadius"])["Value"]
                == 1.0
            )

    def test_intensity_section(self, imported_dict):
        intensity = imported_dict["Intensity"]
        assert intensity["Class"] == "IncoherentIntensity"
        assert len(intensity["Intensities"]) == 4

    @pytest.mark.parametrize(
        "section",
        ["Dynamics", "Kinematics", "Parameters", "ParticleList"],
    )
    def test_expected_recipe_shape(
        self, imported_dict, expected_dict, section
    ):
        expected_section = equalize_dict(expected_dict[section])
        imported_section = equalize_dict(imported_dict[section])
        if isinstance(expected_section, dict):
            assert expected_section.keys() == imported_section.keys()
            imported_items = list(imported_section.values())
            expected_items = list(expected_section.values())
        else:
            expected_items = sorted(expected_section, key=lambda p: p["Name"])
            imported_items = sorted(imported_section, key=lambda p: p["Name"])
        # assert len(imported_items) == len(expected_items)
        for imported, expected in zip(imported_items, expected_items):
            assert imported == expected


class TestCanonicalFormalism:
    @pytest.fixture(scope="session")
    def model(
        self, jpsi_to_gamma_pi_pi_canonical_amplitude_model: AmplitudeModel
    ):
        return jpsi_to_gamma_pi_pi_canonical_amplitude_model

    @pytest.fixture(scope="session")
    def imported_dict(self, output_dir, model: AmplitudeModel):
        output_filename = output_dir + "JPsiToGammaPi0Pi0_cano_recipe.yml"
        asdict = io.asdict(model)
        io.write(model, output_filename)
        return asdict

    def test_not_implemented_writer(
        self,
        jpsi_to_gamma_pi_pi_canonical_amplitude_model: AmplitudeModel,
    ):
        with pytest.raises(NotImplementedError):
            io.write(
                instance=jpsi_to_gamma_pi_pi_canonical_amplitude_model,
                filename="JPsiToGammaPi0Pi0.csv",
            )

    def test_particle_section(self, imported_dict):
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

    def test_parameter_section(self, imported_dict):
        parameter_list = imported_dict["Parameters"]
        assert len(parameter_list) == 11
        for parameter in parameter_list:
            assert "Name" in parameter
            assert "Value" in parameter

    def test_clebsch_gordan(self, imported_dict):
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


def equalize_dict(input_dict):
    output_dict = json.loads(json.dumps(input_dict, sort_keys=True))
    return output_dict
