"""Lineshape functions that describe the dynamics of an interaction.

.. seealso:: :doc:`/usage/dynamics` and :doc:`/usage/dynamics/analytic-continuation`
"""

# cspell:ignore asner mhash
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import sympy as sp

# pyright: reportUnusedImport=false
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
class BlattWeisskopfSquared(sp.Expr):
    # cspell:ignore pychyGekoppeltePartialwellenanalyseAnnihilationen
    r"""Blatt-Weisskopf function :math:`B_L^2(z)`, up to :math:`L \leq 8`.

    Args:
        z: Argument of the Blatt-Weisskopf function :math:`B_L^2(z)`. A usual
            choice is :math:`z = (d q)^2` with :math:`d` the impact parameter and
            :math:`q` the breakup-momentum (see `.BreakupMomentumSquared`).

        angular_momentum: Angular momentum :math:`L` of the decaying particle.

    Note that equal powers of :math:`z` appear in the nominator and the denominator,
    while some sources have nominator :math:`1`, instead of :math:`z^L`. Compare for
    instance Equation (50.27) in :pdg-review:`2021; Resonances; p.9`.

    Each of these cases for :math:`L` has been taken from
    :cite:`pychyGekoppeltePartialwellenanalyseAnnihilationen2016`, p.59,
    :cite:`chungPartialWaveAnalysis1995`, p.415, and
    :cite:`chungFormulasAngularMomentumBarrier2015`. For a good overview of where to use
    these Blatt-Weisskopf functions, see :cite:`asnerDalitzPlotAnalysis2006`.

    See also :ref:`usage/dynamics:Form factor`.
    """

    z: Any
    angular_momentum: Any
    _latex_repr_ = R"B_{{{angular_momentum}}}^2\left({z}\right)"

    max_angular_momentum: ClassVar[int | None] = None
    """Limit the maximum allowed angular momentum :math:`L`.

    This improves performance when :math:`L` is a `~sympy.core.symbol.Symbol` and you
    are note interested in higher angular momenta.
    """

    def evaluate(self) -> sp.Expr:
        z: sp.Expr = self.args[0]  # type: ignore[assignment]
        angular_momentum: sp.Expr = self.args[1]  # type: ignore[assignment]
        cases: dict[int, sp.Expr] = {
            0: sp.S.One,
            1: 2 * z / (z + 1),
            2: 13 * z**2 / ((z - 3) * (z - 3) + 9 * z),
            3: 277 * z**3 / (z * (z - 15) * (z - 15) + 9 * (2 * z - 5) * (2 * z - 5)),
            4: (
                12746
                * z**4
                / (
                    (z**2 - 45 * z + 105) * (z**2 - 45 * z + 105)
                    + 25 * z * (2 * z - 21) * (2 * z - 21)
                )
            ),
            5: (
                998881
                * z**5
                / (z**5 + 15 * z**4 + 315 * z**3 + 6300 * z**2 + 99225 * z + 893025)
            ),
            6: (
                118394977
                * z**6
                / (
                    z**6
                    + 21 * z**5
                    + 630 * z**4
                    + 18900 * z**3
                    + 496125 * z**2
                    + 9823275 * z
                    + 108056025
                )
            ),
            7: (
                19727003738
                * z**7
                / (
                    z**7
                    + 28 * z**6
                    + 1134 * z**5
                    + 47250 * z**4
                    + 1819125 * z**3
                    + 58939650 * z**2
                    + 1404728325 * z
                    + 18261468225
                )
            ),
            8: (
                4392846440677
                * z**8
                / (
                    z**8
                    + 36 * z**7
                    + 1890 * z**6
                    + 103950 * z**5
                    + 5457375 * z**4
                    + 255405150 * z**3
                    + 9833098275 * z**2
                    + 273922023375 * z
                    + 4108830350625
                )
            ),
        }
        return sp.Piecewise(*[
            (expression, sp.Eq(angular_momentum, value))
            for value, expression in cases.items()
            if self.max_angular_momentum is None or value <= self.max_angular_momentum
        ])


@unevaluated
class EnergyDependentWidth(sp.Expr):
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
