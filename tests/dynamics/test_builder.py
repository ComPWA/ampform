import pytest
import sympy as sp
from qrules.particle import Particle

from ampform.dynamics import EnergyDependentWidth
from ampform.dynamics.builder import (
    RelativisticBreitWignerBuilder,
    TwoBodyKinematicVariableSet,
)
from ampform.dynamics.form_factor import FormFactor


class TestRelativisticBreitWignerBuilder:
    @pytest.fixture(scope="session")
    def particle(self) -> Particle:
        return Particle(
            name="N",
            mass=1.3,
            width=0.2,
            pid=1111111,
            spin=3 / 2,
        )

    @pytest.fixture(scope="session")
    def variable_set(self) -> TwoBodyKinematicVariableSet:
        return TwoBodyKinematicVariableSet(
            incoming_state_mass=sp.Symbol("m"),
            outgoing_state_mass1=sp.Symbol("m1"),
            outgoing_state_mass2=sp.Symbol("m2"),
            helicity_phi=sp.Symbol("phi"),
            helicity_theta=sp.Symbol("theta"),
            angular_momentum=sp.Symbol("L", integer=True, negative=False),  # type: ignore[arg-type]
        )

    def test_simple_breit_wigner(
        self, particle: Particle, variable_set: TwoBodyKinematicVariableSet
    ):
        builder = RelativisticBreitWignerBuilder()
        builder.energy_dependent_width = False
        builder.form_factor = False

        bw, parameters = builder(particle, variable_set)
        s = variable_set.incoming_state_mass**2
        m0 = sp.Symbol("m_{N}", nonnegative=True)
        w0 = sp.Symbol(R"\Gamma_{N}", nonnegative=True)
        assert bw == w0 * m0 / (-sp.I * w0 * m0 - s + m0**2)
        assert set(parameters) == {m0, w0}
        assert parameters[m0] == particle.mass
        assert parameters[w0] == particle.width

        builder.form_factor = True
        bw_with_ff, parameters = builder(particle, variable_set)
        m1 = variable_set.outgoing_state_mass1
        m2 = variable_set.outgoing_state_mass2
        L = variable_set.angular_momentum  # noqa: N806
        d = sp.Symbol(R"d_{N}", positive=True)
        form_factor = FormFactor(s, m1, m2, angular_momentum=L, meson_radius=d)
        assert bw_with_ff / bw == form_factor
        assert set(parameters) == {m0, w0, d}
        assert parameters[m0] == particle.mass
        assert parameters[w0] == particle.width
        assert parameters[d] == 1

    def test_breit_wigner_with_energy_dependent_width(
        self, particle: Particle, variable_set: TwoBodyKinematicVariableSet
    ):
        builder = RelativisticBreitWignerBuilder()
        builder.energy_dependent_width = True
        builder.form_factor = False

        bw, parameters = builder(particle, variable_set)
        s = variable_set.incoming_state_mass**2
        m0 = sp.Symbol("m_{N}", nonnegative=True)
        w0 = sp.Symbol(R"\Gamma_{N}", nonnegative=True)
        m1 = variable_set.outgoing_state_mass1
        m2 = variable_set.outgoing_state_mass2
        ang_mom = variable_set.angular_momentum
        d = sp.Symbol(R"d_{N}", positive=True)
        w = EnergyDependentWidth(
            s, m0, w0, m_a=m1, m_b=m2, angular_momentum=ang_mom, meson_radius=d
        )
        assert bw == w0 * m0 / (-sp.I * w * m0 - s + m0**2)
        assert set(parameters) == {m0, w0, d}
        assert parameters[m0] == particle.mass
        assert parameters[w0] == particle.width
        assert parameters[d] == 1

        builder.form_factor = True
        bw_with_ff, parameters = builder(particle, variable_set)
        ang_mom = variable_set.angular_momentum
        form_factor = FormFactor(s, m1, m2, angular_momentum=ang_mom, meson_radius=d)
        assert bw_with_ff / bw == form_factor
        assert set(parameters) == {m0, w0, d}
        assert parameters[m0] == particle.mass
        assert parameters[w0] == particle.width
        assert parameters[d] == 1
