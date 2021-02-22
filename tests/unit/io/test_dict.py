# pylint: disable=no-self-use,redefined-outer-name
import json
from copy import deepcopy
from os.path import dirname, realpath

import pytest
import yaml

from expertsystem import io
from expertsystem.amplitude.model import AmplitudeModel, Dynamics
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
    output_dir: str, particle_selection: ParticleCollection
):
    with pytest.raises(NotImplementedError):
        io.load(__file__)
    with pytest.raises(NotImplementedError):
        io.write(particle_selection, output_dir + "test.py")
    with pytest.raises(Exception):
        io.write(particle_selection, output_dir + "no_file_extension")
    with pytest.raises(NotImplementedError):
        io.write(666, output_dir + "wont_work_anyway.yml")


def test_serialization(
    output_dir: str, particle_selection: ParticleCollection
):
    io.write(particle_selection, output_dir + "particle_selection.yml")
    assert len(particle_selection) == 181
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
    def imported_dict(self, output_dir: str, model: AmplitudeModel):
        output_filename = output_dir + "JPsiToGammaPi0Pi0_heli_recipe.yml"
        io.write(model, output_filename)
        asdict = io.asdict(model)
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
        output_dir: str,
        jpsi_to_gamma_pi_pi_helicity_amplitude_model: AmplitudeModel,
    ):
        with pytest.raises(NotImplementedError):
            io.write(
                instance=jpsi_to_gamma_pi_pi_helicity_amplitude_model,
                filename=output_dir + "JPsiToGammaPi0Pi0.csv",
            )

    def test_create_recipe_dict(
        self,
        jpsi_to_gamma_pi_pi_helicity_amplitude_model: AmplitudeModel,
    ):
        assert len(jpsi_to_gamma_pi_pi_helicity_amplitude_model.particles) == 5
        assert len(jpsi_to_gamma_pi_pi_helicity_amplitude_model.dynamics) == 3

    def test_particle_section(self, imported_dict):
        particle_list = imported_dict.get("particles", imported_dict)
        gamma = next(p for p in particle_list if p["name"] == "gamma")
        assert gamma["pid"] == 22
        assert gamma["mass"] == 0.0
        assert gamma["spin"] == 1.0
        assert gamma["parity"]["value"] == -1
        assert gamma["c_parity"]["value"] == -1

        f0_980 = next(p for p in particle_list if p["name"] == "f(0)(980)")
        assert f0_980["width"] == 0.06

        pi0 = next(p for p in particle_list if p["name"] == "pi0")
        assert pi0["isospin"]["magnitude"] == 1
        assert pi0["isospin"]["projection"] == 0

    def test_kinematics_section(self, imported_dict):
        kinematics = imported_dict["kinematics"]
        initial_state = kinematics["initial_state"]
        final_state = kinematics["final_state"]
        assert kinematics["type"] == "HELICITY"
        assert initial_state == {0: "J/psi(1S)"}
        assert final_state == {2: "gamma", 3: "pi0", 4: "pi0"}

    def test_parameter_section(self, imported_dict):
        parameter_list = imported_dict["parameters"]
        assert len(parameter_list) == 11
        for parameter in parameter_list:
            assert "name" in parameter
            assert "value" in parameter
            assert isinstance(parameter["fix"], bool)

    def test_dynamics_section(self, imported_dict):
        parameter_list: list = imported_dict["parameters"]

        def get_parameter(parameter_name: str) -> dict:
            for par in parameter_list:
                name = par["name"]
                if name == parameter_name:
                    return par
            raise LookupError(f'Could not find parameter  "{parameter_name}"')

        dynamics = imported_dict["dynamics"]
        assert len(dynamics) == 3

        j_psi = dynamics["J/psi(1S)"]
        assert j_psi["type"] == "NonDynamic"
        assert j_psi["form_factor"]["type"] == "BlattWeisskopf"
        assert (
            get_parameter(j_psi["form_factor"]["meson_radius"])["value"] == 1.0
        )

        f0_980 = dynamics.get("f(0)(980)", None)
        if f0_980:
            assert f0_980["type"] == "RelativisticBreitWigner"
            assert f0_980["form_factor"]["type"] == "BlattWeisskopf"
            assert (
                f0_980["form_factor"]["meson_radius"]
                == "MesonRadius_f(0)(980)"
            )
            assert (
                get_parameter(f0_980["form_factor"]["meson_radius"])["value"]
                == 1.0
            )

    def test_intensity_section(self, imported_dict):
        intensity = imported_dict["intensity"]
        assert intensity["type"] == "IncoherentIntensity"
        assert len(intensity["intensities"]) == 4

    @pytest.mark.parametrize(
        "section",
        ["dynamics", "kinematics", "parameters", "particles"],
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
            expected_items = sorted(expected_section, key=lambda p: p["name"])
            imported_items = sorted(imported_section, key=lambda p: p["name"])
        assert len(imported_items) == len(expected_items)
        for imported, expected in zip(imported_items, expected_items):
            assert imported == expected


class TestCanonicalFormalism:
    @pytest.fixture(scope="session")
    def model(
        self, jpsi_to_gamma_pi_pi_canonical_amplitude_model: AmplitudeModel
    ):
        return jpsi_to_gamma_pi_pi_canonical_amplitude_model

    @pytest.fixture(scope="session")
    def imported_dict(self, output_dir: str, model: AmplitudeModel):
        output_filename = output_dir + "JPsiToGammaPi0Pi0_cano_recipe.yml"
        io.write(model, output_filename)
        asdict = io.asdict(model)
        return asdict

    def test_not_implemented_writer(
        self,
        output_dir: str,
        jpsi_to_gamma_pi_pi_canonical_amplitude_model: AmplitudeModel,
    ):
        with pytest.raises(NotImplementedError):
            io.write(
                instance=jpsi_to_gamma_pi_pi_canonical_amplitude_model,
                filename=output_dir + "JPsiToGammaPi0Pi0.csv",
            )

    def test_particle_section(self, imported_dict):
        particle_list = imported_dict["particles"]
        gamma = next(p for p in particle_list if p["name"] == "gamma")
        assert gamma["pid"] == 22
        assert gamma["mass"] == 0.0
        assert gamma["c_parity"]["value"] == -1
        f0_980 = next(p for p in particle_list if p["name"] == "f(0)(980)")
        assert f0_980["width"] == 0.06
        pi0 = next(p for p in particle_list if p["name"] == "pi0")
        assert pi0["isospin"]["magnitude"] == 1

    def test_parameter_section(self, imported_dict):
        parameter_list = imported_dict["parameters"]
        assert len(parameter_list) == 11
        for parameter in parameter_list:
            assert "name" in parameter
            assert "value" in parameter

    def test_clebsch_gordan(self, imported_dict):
        incoherent_intensity = imported_dict["intensity"]
        coherent_intensity = incoherent_intensity["intensities"][0]
        coefficient_amplitude = coherent_intensity["amplitudes"][0]
        sequential_amplitude = coefficient_amplitude["amplitude"]
        canonical_decay = sequential_amplitude["amplitudes"][0]
        s2s3 = canonical_decay["s2s3"]
        assert list(s2s3) == ["J", "M", "j_1", "m_1", "j_2", "m_2"]
        assert s2s3["J"] == 1.0


def test_form_factor(
    output_dir: str,
    jpsi_to_gamma_pi_pi_helicity_amplitude_model: AmplitudeModel,
):
    model = deepcopy(jpsi_to_gamma_pi_pi_helicity_amplitude_model)
    for dynamics in model.dynamics.values():
        assert isinstance(dynamics, Dynamics)
        dynamics.form_factor = None

    io.write(model, output_dir + "test_form_factor.yml")
    asdict = io.asdict(model)
    imported_model = io.fromdict(asdict)

    assert isinstance(imported_model, AmplitudeModel)
    for dynamics in imported_model.dynamics.values():
        assert isinstance(dynamics, Dynamics)
        assert dynamics.form_factor is None
    assert imported_model == model


def equalize_dict(input_dict):
    output_dict = json.loads(json.dumps(input_dict, sort_keys=True))
    return output_dict
