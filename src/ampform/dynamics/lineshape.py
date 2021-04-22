# cspell:ignore Asner Nakamura
# pylint: disable=protected-access, unbalanced-tuple-unpacking, unused-argument
"""Lineshape functions that describe the dynamics.

.. seealso:: :doc:`/usage/dynamics/lineshapes`
"""

from typing import Any

import sympy as sp
from sympy.printing.latex import LatexPrinter

from .decorator import UnevaluatedExpression, implement_doit_method


@implement_doit_method()
class BlattWeisskopf(UnevaluatedExpression):
    r"""Blatt-Weisskopf function :math:`B_L(z)`, up to :math:`L \leq 8`.

    Args:
        angular_momentum: Angular momentum :math:`L` of the decaying particle.

        z: Argument of the Blatt-Weisskopf function :math:`B_L(z)`. A usual
            choice is :math:`z = (d q)^2` with :math:`d` the impact parameter
            and :math:`q` the `breakup_momentum`.

    Function :math:`B_L(z)` is defined as:

    .. glue:math:: BlattWeisskopf
        :label: BlattWeisskopf

    Each of these cases has been taken from
    :cite:`chungPartialWaveAnalysis1995`, p. 415, and
    :cite:`chungFormulasAngularMomentumBarrier2015`. For a good overview of
    where to use these Blatt-Weisskopf functions, see
    :cite:`asnerDalitzPlotAnalysis2006`.

    See also :ref:`usage/dynamics/lineshapes:Form factor`.
    """

    def __new__(  # pylint: disable=arguments-differ
        cls,
        angular_momentum: sp.Symbol,
        z: sp.Symbol,
        evaluate: bool = False,
        **hints: Any,
    ) -> "BlattWeisskopf":
        args = sp.sympify((angular_momentum, z))
        if evaluate:
            # pylint: disable=no-member
            return sp.Expr.__new__(cls, *args, **hints).evaluate()
        return sp.Expr.__new__(cls, *args, **hints)

    def evaluate(self) -> sp.Expr:
        angular_momentum, z = self.args
        return sp.sqrt(
            sp.Piecewise(
                (
                    1,
                    sp.Eq(angular_momentum, 0),
                ),
                (
                    2 * z / (z + 1),
                    sp.Eq(angular_momentum, 1),
                ),
                (
                    13 * z ** 2 / ((z - 3) * (z - 3) + 9 * z),
                    sp.Eq(angular_momentum, 2),
                ),
                (
                    (
                        277
                        * z ** 3
                        / (
                            z * (z - 15) * (z - 15)
                            + 9 * (2 * z - 5) * (2 * z - 5)
                        )
                    ),
                    sp.Eq(angular_momentum, 3),
                ),
                (
                    (
                        12746
                        * z ** 4
                        / (
                            (z ** 2 - 45 * z + 105) * (z ** 2 - 45 * z + 105)
                            + 25 * z * (2 * z - 21) * (2 * z - 21)
                        )
                    ),
                    sp.Eq(angular_momentum, 4),
                ),
                (
                    998881
                    * z ** 5
                    / (
                        z ** 5
                        + 15 * z ** 4
                        + 315 * z ** 3
                        + 6300 * z ** 2
                        + 99225 * z
                        + 893025
                    ),
                    sp.Eq(angular_momentum, 5),
                ),
                (
                    118394977
                    * z ** 6
                    / (
                        z ** 6
                        + 21 * z ** 5
                        + 630 * z ** 4
                        + 18900 * z ** 3
                        + 496125 * z ** 2
                        + 9823275 * z
                        + 108056025
                    ),
                    sp.Eq(angular_momentum, 6),
                ),
                (
                    19727003738
                    * z ** 7
                    / (
                        z ** 7
                        + 28 * z ** 6
                        + 1134 * z ** 5
                        + 47250 * z ** 4
                        + 1819125 * z ** 3
                        + 58939650 * z ** 2
                        + 1404728325 * z
                        + 18261468225
                    ),
                    sp.Eq(angular_momentum, 7),
                ),
                (
                    4392846440677
                    * z ** 8
                    / (
                        z ** 8
                        + 36 * z ** 7
                        + 1890 * z ** 6
                        + 103950 * z ** 5
                        + 5457375 * z ** 4
                        + 255405150 * z ** 3
                        + 9833098275 * z ** 2
                        + 273922023375 * z
                        + 4108830350625
                    ),
                    sp.Eq(angular_momentum, 8),
                ),
            )
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        angular_momentum, z = tuple(map(printer._print, self.args))
        return fR"B_{{{angular_momentum}}}\left({z}\right)"


def relativistic_breit_wigner(
    mass: sp.Symbol, mass0: sp.Symbol, gamma0: sp.Symbol
) -> sp.Expr:
    """Relativistic Breit-Wigner lineshape.

    .. glue:math:: relativistic_breit_wigner
        :label: relativistic_breit_wigner

    See :ref:`usage/dynamics/lineshapes:_Without_ form factor` and
    :cite:`asnerDalitzPlotAnalysis2006`.
    """
    return gamma0 * mass0 / (mass0 ** 2 - mass ** 2 - gamma0 * mass0 * sp.I)


def relativistic_breit_wigner_with_ff(  # pylint: disable=too-many-arguments
    mass: sp.Symbol,
    mass0: sp.Symbol,
    gamma0: sp.Symbol,
    m_a: sp.Symbol,
    m_b: sp.Symbol,
    angular_momentum: sp.Symbol,
    meson_radius: sp.Symbol,
) -> sp.Expr:
    """Relativistic Breit-Wigner with `.BlattWeisskopf` factor.

    For :math:`L=0`, this lineshape has the following form:

    .. glue:math:: relativistic_breit_wigner_with_ff
        :label: relativistic_breit_wigner_with_ff

    See :ref:`usage/dynamics/lineshapes:_With_ form factor` and
    :cite:`asnerDalitzPlotAnalysis2006`.
    """
    q = breakup_momentum(mass, m_a, m_b)
    q0 = breakup_momentum(mass0, m_a, m_b)
    form_factor = BlattWeisskopf(angular_momentum, z=(q * meson_radius) ** 2)
    form_factor0 = BlattWeisskopf(angular_momentum, z=(q0 * meson_radius) ** 2)
    mass_dependent_width = (
        gamma0
        * (mass0 / mass)
        * (form_factor ** 2 / form_factor0 ** 2)
        * (q / q0)
    )
    return (mass0 * gamma0 * form_factor) / (
        mass0 ** 2 - mass ** 2 - mass_dependent_width * mass0 * sp.I
    )


def breakup_momentum(
    m_r: sp.Symbol, m_a: sp.Symbol, m_b: sp.Symbol
) -> sp.Expr:
    return sp.sqrt(
        (m_r ** 2 - (m_a + m_b) ** 2)
        * (m_r ** 2 - (m_a - m_b) ** 2)
        / (4 * m_r ** 2)
    )
