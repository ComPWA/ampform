import pytest

from expertsystem.amplitude.model import (
    AmplitudeModel,
    FitParameter,
    FitParameters,
    Kinematics,
    KinematicsType,
    NonDynamic,
    ParticleDynamics,
    RelativisticBreitWigner,
    _assert_arg_type,
)
from expertsystem.particle import ParticleCollection
from expertsystem.reaction import Result


class TestFitParameters:
    @staticmethod
    def test_add_exceptions():
        parameters = FitParameters()
        with pytest.raises(TypeError):
            parameters.add("par1")  # type: ignore
        dummy_par = FitParameter(name="par1", value=0.0)
        parameters.add(dummy_par)
        with pytest.raises(KeyError):
            parameters.add(dummy_par)

    @staticmethod
    def test_remove():
        parameters = FitParameters()
        assert len(parameters) == 0
        par1 = FitParameter(name="p1", value=0.0)
        par2 = FitParameter(name="p2", value=0.0)
        parameters.add(par2)
        parameters.add(par1)
        assert list(parameters) == ["p2", "p1"]

    def test_eval(  # pylint: disable=no-self-use
        self,
        jpsi_to_gamma_pi_pi_canonical_amplitude_model: AmplitudeModel,
    ):
        model = jpsi_to_gamma_pi_pi_canonical_amplitude_model
        parameters = model.parameters
        from_repr = eval(repr(parameters))  # pylint: disable=eval-used
        assert from_repr == parameters


class TestKinematics:
    @staticmethod
    def test_post_init(particle_database: ParticleCollection):
        pi0 = particle_database["pi0"]
        with pytest.raises(ValueError):
            Kinematics(
                initial_state={0: pi0, 1: pi0},
                final_state={1: pi0, 2: pi0},
            )

    @staticmethod
    def test_from_graph(jpsi_to_gamma_pi_pi_helicity_solutions: Result):
        result = jpsi_to_gamma_pi_pi_helicity_solutions
        graph = next(iter(result.solutions))
        kinematics = Kinematics.from_graph(graph)
        assert len(kinematics.initial_state) == 1
        assert len(kinematics.final_state) == 3
        assert kinematics.id_to_particle[0].name == "J/psi(1S)"
        assert kinematics.id_to_particle[2].name == "gamma"
        assert kinematics.id_to_particle[3].name == "pi0"
        assert kinematics.id_to_particle[4].name == "pi0"
        assert kinematics.type == KinematicsType.Helicity


class TestParticleDynamics:
    @staticmethod
    def test_init(particle_database: ParticleCollection):
        pdg = particle_database
        dynamics = ParticleDynamics(pdg, FitParameters())
        jpsi = "J/psi(1S)"
        pi0 = "pi0"
        gamma = "gamma"
        assert len(dynamics) == 0
        dynamics.set_non_dynamic(jpsi)
        assert len(dynamics) == 1
        dynamics.set_non_dynamic(pi0)
        assert len(dynamics) == 2
        with pytest.raises(NotImplementedError):
            dynamics.set_breit_wigner(gamma, relativistic=False)
        dynamics.set_breit_wigner(gamma, relativistic=True)
        assert len(dynamics) == 3
        assert dynamics[pi0] is not dynamics[jpsi]
        assert dynamics[pi0] != dynamics[jpsi]
        assert isinstance(dynamics[jpsi], NonDynamic)
        assert isinstance(dynamics[pi0], NonDynamic)
        assert isinstance(dynamics[gamma], RelativisticBreitWigner)
        for particle_name, item in zip([jpsi, pi0, gamma], dynamics):
            assert item == particle_name

        pars = dynamics.parameters
        assert len(pars) == 5
        assert len(pars.filter(lambda p: p.name.startswith("Meson"))) == 3

        gamma_mass = pars[f"Position_{gamma}"]
        gamma_width = pars[f"Width_{gamma}"]
        assert gamma_mass.value == pdg[gamma].mass
        assert gamma_width.value == pdg[gamma].width
        assert list(pars.filter(lambda p: not p.fix).values()) == [
            gamma_mass,
            gamma_width,
        ]


def test_assert_type():
    with pytest.raises(TypeError):
        _assert_arg_type(666, str)
