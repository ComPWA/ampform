import numpy as np
import pytest
import sympy as sp
from qrules.particle import Particle

import ampform.decay
from ampform.decay import IsobarNode, ParticleLike, State, ThreeBodyDecayChain
from ampform.dynamics import EnergyDependentWidth
from ampform.dynamics.builder import (
    RelativisticBreitWignerBuilder,
    TwoBodyKinematicVariableSet,
    create_non_dynamic_with_ff,
    formulate_breit_wigner_with_form_factor,
)
from ampform.dynamics.form_factor import FormFactor


class TestRelativisticBreitWignerBuilder:
    @pytest.fixture(scope="session", params=["ampform", "qrules"])
    def particle(self, request: pytest.FixtureRequest) -> ParticleLike:
        if request.param == "ampform":
            return ampform.decay.Particle(
                name="N",
                latex="N",
                spin=1.5,
                parity=None,
                mass=1.3,
                width=0.2,
            )
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
            angular_momentum=sp.Symbol("L", integer=True, negative=False),
        )

    def test_simple_breit_wigner(
        self, particle: ParticleLike, variable_set: TwoBodyKinematicVariableSet
    ):
        builder = RelativisticBreitWignerBuilder()
        builder.energy_dependent_width = False
        builder.form_factor = False

        bw = builder(particle, variable_set)
        s = variable_set.incoming_state_mass**2
        m0 = sp.Symbol("m_{N}", nonnegative=True)
        w0 = sp.Symbol(R"\Gamma_{N}", nonnegative=True)
        assert bw.expression == w0 * m0 / (-sp.I * w0 * m0 - s + m0**2)
        assert set(bw.parameters) == {m0, w0}
        assert bw.parameters[m0] == particle.mass
        assert bw.parameters[w0] == particle.width

        builder.form_factor = True
        bw_with_ff = builder(particle, variable_set)
        m1 = variable_set.outgoing_state_mass1
        m2 = variable_set.outgoing_state_mass2
        L = variable_set.angular_momentum
        d = sp.Symbol(R"d_{N}", positive=True)
        form_factor = FormFactor(s, m1, m2, angular_momentum=L, meson_radius=d)
        assert bw_with_ff.expression / bw.expression == form_factor
        assert set(bw_with_ff.parameters) == {m0, w0, d}
        assert bw_with_ff.parameters[m0] == particle.mass
        assert bw_with_ff.parameters[w0] == particle.width
        assert bw_with_ff.parameters[d] == 1

    def test_breit_wigner_with_energy_dependent_width(
        self, particle: ParticleLike, variable_set: TwoBodyKinematicVariableSet
    ):
        builder = RelativisticBreitWignerBuilder()
        builder.energy_dependent_width = True
        builder.form_factor = False

        bw = builder(particle, variable_set)
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
        assert bw.expression == w0 * m0 / (-sp.I * w * m0 - s + m0**2)
        assert set(bw.parameters) == {m0, w0, d}
        assert bw.parameters[m0] == particle.mass
        assert bw.parameters[w0] == particle.width
        assert bw.parameters[d] == 1

        builder.form_factor = True
        bw_with_ff = builder(particle, variable_set)
        ang_mom = variable_set.angular_momentum
        form_factor = FormFactor(s, m1, m2, angular_momentum=ang_mom, meson_radius=d)
        assert bw_with_ff.expression / bw.expression == form_factor
        assert set(bw_with_ff.parameters) == {m0, w0, d}
        assert bw_with_ff.parameters[m0] == particle.mass
        assert bw_with_ff.parameters[w0] == particle.width
        assert bw_with_ff.parameters[d] == 1


def test_chain_dynamics_equal_product_of_node_dynamics():
    """Chain-level Breit-Wigner dynamics factorize into node-level dynamics.

    The `.DynamicsBuilder` interface (used by the DPD builder) formulates dynamics for
    an entire three-body decay chain, whereas the `.ResonanceDynamicsBuilder` interface
    (used by the helicity builder) formulates dynamics per two-body decay node. The
    chain expression is numerically the product of a production form factor and a
    Breit-Wigner with form factor for the decay node. The only convention difference
    lies in the invariant mass that enters the production form factor: the chain-level
    builder evaluates it at the nominal resonance mass, whereas the helicity builder
    inserts the running invariant mass of the decaying subsystem there. This test
    therefore feeds the nominal mass symbol into that slot.
    """
    lambda_c = State(
        "Lc+",
        latex=R"\Lambda_c^+",
        spin=0.5,
        parity=1,
        mass=2.28646,
        width=0.0,
        index=0,
    )
    proton = State("p", latex="p", spin=0.5, parity=1, mass=0.93827, width=0.0, index=1)
    kaon = State("K-", latex="K^-", spin=0, parity=-1, mass=0.49368, width=0.0, index=2)
    pion = State(
        "pi+", latex=R"\pi^+", spin=0, parity=-1, mass=0.13957, width=0.0, index=3
    )
    lambda_1520 = ampform.decay.Particle(
        "L(1520)", latex=R"\Lambda(1520)", spin=1.5, parity=-1, mass=1.519, width=0.0156
    )
    l_prod, l_dec = 1, 2
    chain = ThreeBodyDecayChain(
        decay=IsobarNode(
            parent=lambda_c,
            child1=IsobarNode(
                lambda_1520, proton, kaon, interaction=(l_dec, sp.Rational(1, 2))
            ),
            child2=pion,
            interaction=(l_prod, sp.Rational(3, 2)),
        )
    )
    chain_expr = formulate_breit_wigner_with_form_factor(chain).expression

    m_pk = sp.Symbol("m_{pK}", nonnegative=True)
    decay_node_pool = TwoBodyKinematicVariableSet(
        incoming_state_mass=m_pk,
        outgoing_state_mass1=sp.Symbol("m_p", nonnegative=True),
        outgoing_state_mass2=sp.Symbol("m_K", nonnegative=True),
        helicity_theta=sp.Symbol("theta_23"),
        helicity_phi=sp.Symbol("phi_23"),
        angular_momentum=l_dec,
    )
    breit_wigner_builder = RelativisticBreitWignerBuilder(
        energy_dependent_width=True, form_factor=True
    )
    decay_node_expr = breit_wigner_builder(lambda_1520, decay_node_pool).expression
    production_node_pool = TwoBodyKinematicVariableSet(
        incoming_state_mass=sp.Symbol("m_{Lc}", nonnegative=True),
        outgoing_state_mass1=sp.Symbol(R"m_{\Lambda(1520)}", nonnegative=True),
        outgoing_state_mass2=sp.Symbol("m_pi", nonnegative=True),
        helicity_theta=sp.Symbol("theta_1"),
        helicity_phi=sp.Symbol("phi_1"),
        angular_momentum=l_prod,
    )
    production_node_expr = create_non_dynamic_with_ff(
        lambda_c, production_node_pool
    ).expression
    node_product = decay_node_expr * production_node_expr

    sigma = np.linspace((0.93827 + 0.49368) ** 2 + 0.01, (2.28646 - 0.13957) ** 2, 50)
    meson_radius_production = 2.5
    meson_radius_decay = 1.3
    chain_values = {
        "sigma3": sigma,
        "m0": 2.28646,
        "m1": 0.93827,
        "m2": 0.49368,
        "m3": 0.13957,
        R"m_{\Lambda(1520)}": 1.519,
        R"\Gamma_{\Lambda(1520)}": 0.0156,
        R"R_{\Lambda_c^+}": meson_radius_production,
        R"R_\mathrm{res}": meson_radius_decay,
    }
    node_values = {
        "m_{pK}": np.sqrt(sigma),
        "m_{Lc}": 2.28646,
        "m_p": 0.93827,
        "m_K": 0.49368,
        "m_pi": 0.13957,
        R"m_{\Lambda(1520)}": 1.519,
        R"\Gamma_{\Lambda(1520)}": 0.0156,
        R"d_{\Lambda_c^+}": meson_radius_production,
        R"d_{\Lambda(1520)}": meson_radius_decay,
    }

    def evaluate(expression: sp.Expr, values: dict[str, np.ndarray | float]):
        symbols = sorted(expression.free_symbols, key=str)
        assert {str(s) for s in symbols} == set(values)
        func = sp.lambdify(symbols, expression.doit())
        return func(*(values[str(s)] for s in symbols))

    chain_array = evaluate(chain_expr, chain_values)
    node_array = evaluate(node_product, node_values)
    np.testing.assert_allclose(chain_array, node_array)
