"""Implementations of the form factor, or barrier factor."""

from __future__ import annotations

from typing import Any, ClassVar

import sympy as sp

from ampform.sympy import unevaluated


@unevaluated
class BlattWeisskopfSquared(sp.Expr):
    # cspell:ignore asner pychyGekoppeltePartialwellenanalyseAnnihilationen
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
