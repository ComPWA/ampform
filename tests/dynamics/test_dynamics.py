from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform.dynamics import (
    BreitWigner,
    ChannelArguments,
    EnergyDependentWidth,
    EqualMassPhaseSpaceFactor,
    MultichannelBreitWigner,
    PhaseSpaceFactor,
    PhaseSpaceFactorSWave,
    SimpleBreitWigner,
    relativistic_breit_wigner_with_ff,
)

if TYPE_CHECKING:
    from qrules import ParticleCollection

    from ampform.helicity import HelicityModel


class TestEnergyDependentWidth:
    @staticmethod
    def test_init():
        angular_momentum = sp.Symbol("L", integer=True)
        s, m0, w0, m1, m2, d = sp.symbols("s m0 Gamma0 m1 m2 d", nonnegative=True)
        width = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m1,
            m_b=m1,
            angular_momentum=0,
            meson_radius=1,
        )
        expr_str = str(width.doit())
        assert (
            expr_str == "Gamma0*m0*sqrt(-4*m1**2 + s)/(sqrt(s)*sqrt(m0**2 - 4*m1**2))"
        )
        assert width.phsp_factor is PhaseSpaceFactor
        assert width.name is None

        width = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m1,
            m_b=m2,
            angular_momentum=angular_momentum,
            meson_radius=d,
            phsp_factor=EqualMassPhaseSpaceFactor,
            name="Gamma_1",
        )
        assert width.phsp_factor is EqualMassPhaseSpaceFactor
        assert width.name == "Gamma_1"

    @pytest.mark.parametrize("method", ["subs", "xreplace"])
    def test_doit_and_subs(self, method: str):
        s, m0, w0, m_a, m_b = sp.symbols("s m0 Gamma0 m_a m_b", nonnegative=True)
        parameters = {
            m0: 1.44,
            w0: 0.35,
            m_a: 0.938,
            m_b: 0.548,
        }
        width = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m_a,
            m_b=m_a,
            angular_momentum=0,
            meson_radius=1,
            phsp_factor=PhaseSpaceFactorSWave,
        )
        subs_first = round_nested(_subs(width, parameters, method).doit(), n_decimals=3)
        doit_first = round_nested(_subs(width.doit(), parameters, method), n_decimals=3)
        subs_first = round_nested(subs_first, n_decimals=3)
        doit_first = round_nested(doit_first, n_decimals=3)
        assert str(subs_first) == str(doit_first)


class TestBreitWigner:
    @staticmethod
    def test_simple_limit():
        s, m0, w0 = sp.symbols("s m0 Gamma0", nonnegative=True)
        breit_wigner = BreitWigner(s, m0, w0)
        assert breit_wigner.doit() == SimpleBreitWigner(s, m0, w0).doit()

    @staticmethod
    def test_energy_dependent_width_only_appears_in_denominator():
        s, m0, w0, m1, m2 = sp.symbols("s m0 Gamma0 m1 m2", nonnegative=True)
        breit_wigner = BreitWigner(s, m0, w0, m1, m2)
        running_width = breit_wigner.energy_dependent_width()
        expected = m0 * w0 / (m0**2 - s - m0 * running_width * sp.I)
        assert breit_wigner.doit(deep=False) == expected

    @staticmethod
    def test_multichannel_width_is_sum_of_channel_widths():
        s, m0, w1, w2, m1, m2 = sp.symbols("s m0 Gamma1 Gamma2 m1 m2", nonnegative=True)
        channels = [ChannelArguments(w1, m1, m2), ChannelArguments(w2, m1, m2)]
        breit_wigner = MultichannelBreitWigner(s, m0, channels)
        total_width = sum(channel.formulate_width(s, m0) for channel in channels)
        assert breit_wigner.doit(deep=False) == SimpleBreitWigner(s, m0, total_width)


def _subs(obj: sp.Basic, replacements: dict, method) -> sp.Expr:
    return getattr(obj, method)(replacements)


def test_generate(
    amplitude_model: tuple[str, HelicityModel],
    particle_database: ParticleCollection,
):
    formalism, model = amplitude_model
    if formalism == "canonical-helicity":
        n_amplitudes = 16
        n_parameters = 10
    else:
        n_amplitudes = 8
        n_parameters = 8
    assert len(model.parameter_defaults) == n_parameters
    assert len(model.components) == 4 + n_amplitudes
    assert len(model.expression.free_symbols) == 7 + n_parameters

    total_intensity: sp.Expr = model.expression.doit()
    total_intensity = total_intensity.subs(model.parameter_defaults)
    assert len(total_intensity.free_symbols) == 5

    angle_symbols = {
        s for s in total_intensity.free_symbols if str(s).startswith(("phi", "theta"))
    }
    angle_substitutions = dict.fromkeys(angle_symbols, 0)
    total_intensity = total_intensity.subs(angle_substitutions)  # ty: ignore[no-matching-overload]
    assert len(total_intensity.free_symbols) == 3

    pi0 = particle_database["pi0"]
    total_intensity = total_intensity.subs(
        {
            sp.Symbol("m_1", nonnegative=True): pi0.mass,
            sp.Symbol("m_2", nonnegative=True): pi0.mass,
        },
        simultaneous=True,
    )
    assert len(total_intensity.free_symbols) == 1

    existing_symbol = next(iter(total_intensity.free_symbols))
    m = sp.Symbol("m", nonnegative=True)
    total_intensity = total_intensity.subs({existing_symbol: m})

    assert isinstance(total_intensity, sp.Mul)
    assert total_intensity.args[0] == 2
    intensity = total_intensity / 2

    assert isinstance(intensity, sp.Pow)
    assert intensity.args[1] == 2
    abs_amplitude = intensity.args[0]

    assert isinstance(abs_amplitude, sp.Abs)
    coherent_sum = abs_amplitude.args[0]

    assert isinstance(coherent_sum, sp.Add)
    if formalism == "canonical-helicity":
        assert len(coherent_sum.args) == 4
    else:
        assert len(coherent_sum.args) == 2
    amplitude = coherent_sum.args[0]

    assert isinstance(amplitude, sp.Mul)
    assert len(amplitude.args) == 2

    amplitude = round_nested(amplitude, n_decimals=2)
    a = str(amplitude)
    assert a == "0.06/(m**2 - 0.98 + 0.06*I*sqrt(m**2 - 0.07)/m)"


@pytest.mark.parametrize(
    "func",
    [
        relativistic_breit_wigner_with_ff,
        EnergyDependentWidth,
    ],
)
def test_relativistic_breit_wigner_with_ff_phsp_factor(func):
    # https://github.com/ComPWA/ampform/issues/267
    m, m0, w0, m1, m2 = sp.symbols("m m0 Gamma0 m1 m2")
    expr = func(
        s=m**2,
        mass0=m0,
        gamma0=w0,
        m_a=m1,
        m_b=m2,
        angular_momentum=0,
        meson_radius=1,
        phsp_factor=PhaseSpaceFactor,
    )
    expr_chew_mandelstam = func(
        s=m**2,
        mass0=m0,
        gamma0=w0,
        m_a=m1,
        m_b=m2,
        angular_momentum=0,
        meson_radius=1,
        phsp_factor=PhaseSpaceFactorSWave,
    )
    assert expr.doit() != expr_chew_mandelstam.doit()


def round_nested(expression: sp.Expr, n_decimals: int) -> sp.Expr:
    no_sqrt_expr = expression
    for node in sp.preorder_traversal(expression):
        if node.free_symbols:
            continue
        if isinstance(node, sp.Pow) and node.args[1] == 1 / 2:
            no_sqrt_expr = no_sqrt_expr.xreplace({node: node.n()})
    rounded_expr = no_sqrt_expr
    for node in sp.preorder_traversal(no_sqrt_expr):
        if isinstance(node, (float, sp.Float)):
            rounded_expr = rounded_expr.xreplace({node: round(node, n_decimals)})
    return rounded_expr
