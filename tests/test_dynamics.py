# pylint: disable=no-self-use, protected-access, too-many-arguments
from typing import Tuple

import sympy as sp
from qrules import ParticleCollection

from ampform.dynamics import (
    BlattWeisskopfSquared,
    EnergyDependentWidth,
    PhaseSpaceFactor,
    PhaseSpaceFactorAnalytic,
)
from ampform.helicity import HelicityModel


class TestBlattWeisskopfSquared:
    def test_max_angular_momentum(self):
        z = sp.Symbol("z")
        angular_momentum = sp.Symbol("L", integer=True)
        form_factor = BlattWeisskopfSquared(angular_momentum, z=z)
        form_factor_9 = form_factor.subs(angular_momentum, 8).evaluate()
        factor, z_power, _ = form_factor_9.args
        assert factor == 4392846440677
        assert z_power == z ** 8
        assert BlattWeisskopfSquared.max_angular_momentum is None
        BlattWeisskopfSquared.max_angular_momentum = 1
        assert form_factor.evaluate() == sp.Piecewise(
            (1, sp.Eq(angular_momentum, 0)),
            (2 * z / (z + 1), sp.Eq(angular_momentum, 1)),
        )


class TestEnergyDependentWidth:
    @staticmethod
    def test_init():
        angular_momentum = sp.Symbol("L", integer=True)
        s, m0, w0, m_a, m_b, d = sp.symbols("s m0 Gamma0 m_a m_b d", real=True)
        width = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m_a,
            m_b=m_a,
            angular_momentum=0,
            meson_radius=1,
        )
        assert width.doit() == w0 * sp.sqrt(-(m_a ** 2) + s / 4) * sp.sqrt(
            m0 ** 2
        ) / (sp.sqrt(s) * sp.sqrt(m0 ** 2 / 4 - m_a ** 2))
        assert width.phsp_factor is PhaseSpaceFactor
        assert width._name is None

        width = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m_a,
            m_b=m_b,
            angular_momentum=angular_momentum,
            meson_radius=d,
            phsp_factor=PhaseSpaceFactorAnalytic,
            name="Gamma_1",
        )
        assert width.phsp_factor is PhaseSpaceFactorAnalytic
        assert width._name == "Gamma_1"


def test_generate(  # pylint: disable=too-many-locals
    amplitude_model: Tuple[str, HelicityModel],
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
    angle_substitutions = {
        s: angle_value
        for s in total_intensity.free_symbols
        if s.name.startswith("phi") or s.name.startswith("theta")
    }
    total_intensity = total_intensity.subs(angle_substitutions)
    assert len(total_intensity.free_symbols) == 3

    pi0 = particle_database["pi0"]
    total_intensity = total_intensity.subs(
        {
            sp.Symbol("m_1", real=True): pi0.mass,
            sp.Symbol("m_2", real=True): pi0.mass,
        },
        simultaneous=True,
    )
    assert len(total_intensity.free_symbols) == 1

    existing_symbol = next(iter(total_intensity.free_symbols))
    m = sp.Symbol("m", real=True)
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
    assert a == "0.06/(-m**2 - 0.06*I*sqrt(m**2 - 0.07)/Abs(m) + 0.98)"


def round_nested(expression: sp.Expr, n_decimals: int) -> sp.Expr:
    for node in sp.preorder_traversal(expression):
        if node.free_symbols:
            continue
        if isinstance(node, (float, sp.Float)):
            expression = expression.subs(node, round(node, n_decimals))
        if isinstance(node, sp.Pow) and node.args[1] == 1 / 2:
            expression = expression.subs(node, round(node.n(), n_decimals))
    return expression
