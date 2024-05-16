"""Lineshape functions that describe the dynamics of an interaction.

.. seealso:: :doc:`/usage/dynamics` and :doc:`/usage/dynamics/analytic-continuation`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

# pyright: reportUnusedImport=false
from ampform.dynamics.form_factor import BlattWeisskopfSquared
from ampform.dynamics.phasespace import (
    BreakupMomentumSquared,
    EqualMassPhaseSpaceFactor,  # noqa: F401
    PhaseSpaceFactor,
    PhaseSpaceFactorAbs,  # noqa: F401
    PhaseSpaceFactorComplex,  # noqa: F401
    PhaseSpaceFactorProtocol,
    PhaseSpaceFactorSWave,  # noqa: F401
    _indices_to_subscript,
)
from ampform.sympy import argument, determine_indices, unevaluated

if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter


@unevaluated
class EnergyDependentWidth(sp.Expr):
    # cspell:ignore asner
    r"""Mass-dependent width, coupled to the pole position of the resonance.

    See Equation (50.28) in :pdg-review:`2021; Resonances; p.9` and
    :cite:`asnerDalitzPlotAnalysis2006`, equation (6). Default value for
    :code:`phsp_factor` is `.PhaseSpaceFactor`.

    Note that the `.BlattWeisskopfSquared` of AmpForm is normalized in the sense that
    equal powers of :math:`z` appear in the nominator and the denominator, while the
    definition in the PDG (as well as some other sources), always have :math:`1` in the
    nominator of the Blatt-Weisskopf. In that case, one needs an additional factor
    :math:`\left(q/q_0\right)^{2L}` in the definition for :math:`\Gamma(m)`.
    """

    s: Any
    mass0: Any
    gamma0: Any
    m_a: Any
    m_b: Any
    angular_momentum: Any
    meson_radius: Any
    phsp_factor: PhaseSpaceFactorProtocol = argument(
        default=PhaseSpaceFactor, sympify=False
    )
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, mass0, gamma0, m_a, m_b, angular_momentum, meson_radius = self.args
        q_squared = BreakupMomentumSquared(s, m_a, m_b)
        q0_squared = BreakupMomentumSquared(mass0**2, m_a, m_b)  # type: ignore[operator]
        form_factor_sq = BlattWeisskopfSquared(
            q_squared * meson_radius**2,  # type: ignore[operator]
            angular_momentum,
        )
        form_factor0_sq = BlattWeisskopfSquared(
            q0_squared * meson_radius**2,  # type: ignore[operator]
            angular_momentum,
        )
        rho = self.phsp_factor(s, m_a, m_b)
        rho0 = self.phsp_factor(mass0**2, m_a, m_b)  # type: ignore[operator]
        return gamma0 * (form_factor_sq / form_factor0_sq) * (rho / rho0)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.args[0])
        gamma0 = self.args[2]
        subscript = _indices_to_subscript(determine_indices(gamma0))
        name = Rf"\Gamma{subscript}" if self.name is None else self.name
        return Rf"{name}\left({s}\right)"


def relativistic_breit_wigner(s, mass0, gamma0) -> sp.Expr:
    """Relativistic Breit-Wigner lineshape.

    See :ref:`usage/dynamics:_Without_ form factor` and
    :cite:`asnerDalitzPlotAnalysis2006`.
    """
    return gamma0 * mass0 / (mass0**2 - s - gamma0 * mass0 * sp.I)


def relativistic_breit_wigner_with_ff(  # noqa: PLR0917
    s,
    mass0,
    gamma0,
    m_a,
    m_b,
    angular_momentum,
    meson_radius,
    phsp_factor: PhaseSpaceFactorProtocol = PhaseSpaceFactor,
) -> sp.Expr:
    """Relativistic Breit-Wigner with `.BlattWeisskopfSquared` factor.

    See :ref:`usage/dynamics:_With_ form factor` and :pdg-review:`2021; Resonances;
    p.9`.
    """
    form_factor = formulate_form_factor(s, m_a, m_b, angular_momentum, meson_radius)
    energy_dependent_width = EnergyDependentWidth(
        s, mass0, gamma0, m_a, m_b, angular_momentum, meson_radius, phsp_factor
    )
    return (mass0 * gamma0 * form_factor) / (
        mass0**2 - s - energy_dependent_width * mass0 * sp.I
    )


def formulate_form_factor(s, m_a, m_b, angular_momentum, meson_radius) -> sp.Expr:
    """Formulate a Blatt-Weisskopf form factor.

    Returns the production process factor :math:`n_a` from Equation (50.26) in
    :pdg-review:`2021; Resonances; p.9`, which features the
    `~sympy.functions.elementary.miscellaneous.sqrt` of a `.BlattWeisskopfSquared`.
    """
    q_squared = BreakupMomentumSquared(s, m_a, m_b)
    ff_squared = BlattWeisskopfSquared(q_squared * meson_radius**2, angular_momentum)
    return sp.sqrt(ff_squared)
