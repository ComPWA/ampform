from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from ampform.dynamics import (
    EnergyDependentWidth,
    EqualMassPhaseSpaceFactor,
    PhaseSpaceFactor,
    PhaseSpaceFactorSWave,
    relativistic_breit_wigner_with_ff,
)

if TYPE_CHECKING:
    from qrules import ParticleCollection

    from ampform.helicity import HelicityModel


class TestEnergyDependentWidth:
    @staticmethod
    def test_init():
        angular_momentum = sp.Symbol("L", integer=True)
        s, m0, w0, m_a, m_b, d = sp.symbols("s m0 Gamma0 m_a m_b d", nonnegative=True)
        width = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m_a,
            m_b=m_a,
            angular_momentum=0,
            meson_radius=1,
        )
        assert width.doit() == w0 * sp.sqrt(-(m_a**2) + s / 4) * sp.sqrt(m0**2) / (
            sp.sqrt(s) * sp.sqrt(m0**2 / 4 - m_a**2)
        )
        assert width.phsp_factor is PhaseSpaceFactor
        assert width.name is None

        width = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m_a,
            m_b=m_b,
            angular_momentum=angular_momentum,
            meson_radius=d,
            phsp_factor=EqualMassPhaseSpaceFactor,  # type:ignore[arg-type]
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
            phsp_factor=PhaseSpaceFactorSWave,  # type:ignore[arg-type]
        )
        subs_first = round_nested(_subs(width, parameters, method).doit(), n_decimals=3)
        doit_first = round_nested(_subs(width.doit(), parameters, method), n_decimals=3)
        subs_first = round_nested(subs_first, n_decimals=3)
        doit_first = round_nested(doit_first, n_decimals=3)
        assert str(subs_first) == str(doit_first)


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

    angle_value = 0
    free_symbols: set[sp.Symbol] = total_intensity.free_symbols  # type: ignore[assignment]
    angle_substitutions = {
        s: angle_value
        for s in free_symbols
        if s.name.startswith("phi") or s.name.startswith("theta")
    }
    total_intensity = total_intensity.subs(angle_substitutions)
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
