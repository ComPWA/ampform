import pytest

from expertsystem.amplitude.model import (
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


class TestFitParameters:
    @staticmethod
    def test_add_exceptions():
        parameters = FitParameters()
        with pytest.raises(TypeError):
            parameters.add("par1")  # type: ignore
        parameters.add(FitParameter(name="par1"))
        with pytest.raises(KeyError):
            parameters.add(FitParameter("par1"))

    @staticmethod
    def test_remove():
        parameters = FitParameters()
        assert len(parameters) == 0
        par1 = FitParameter(name="p1")
        par2 = FitParameter(name="p2")
        parameters.add(par2)
        parameters.add(par1)
        assert list(parameters) == ["p2", "p1"]


class TestKinematics:
    @staticmethod
    def test_init(particle_database):
        jpsi = particle_database["J/psi(1S)"]
        gamma = particle_database["gamma"]
        pi0 = particle_database["pi0"]
        kinematics = Kinematics(particle_database)
        assert kinematics.kinematics_type == KinematicsType.Helicity
        kinematics.set_reaction(
            initial_state=["J/psi(1S)"],
            final_state=["gamma", "pi0", "pi0"],
            intermediate_states=1,
        )
        assert kinematics.initial_state == {0: jpsi}
        assert kinematics.final_state == {2: gamma, 3: pi0, 4: pi0}
        assert kinematics.id_to_particle(0) is jpsi
        with pytest.raises(KeyError):
            kinematics.id_to_particle(1)
        assert kinematics.id_to_particle(2) is gamma
        assert kinematics.id_to_particle(3) is pi0
        assert kinematics.id_to_particle(4) is pi0
        with pytest.raises(KeyError):
            kinematics.id_to_particle(5)


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
        assert list(pars.filter(lambda p: not p.is_fixed).values()) == [
            gamma_mass,
            gamma_width,
        ]


def test_assert_type():
    with pytest.raises(TypeError):
        _assert_arg_type(666, str)
