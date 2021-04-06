# cspell:ignore Asner Nakamura
# pylint: disable=invalid-name,protected-access,unused-argument
"""Lineshape functions that describe the dynamics.

.. seealso:: :doc:`/usage/dynamics/lineshapes`
"""

from abc import abstractmethod
from typing import Any, Callable, Type

import sympy as sp
from sympy.printing.latex import LatexPrinter


class UnevaluatedExpression(sp.Expr):
    @abstractmethod
    def evaluate(self) -> sp.Expr:
        pass

    @abstractmethod
    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        """Provide a mathematical Latex representation for notebooks."""
        args = tuple(map(printer._print, self.args))
        return f"{self.__class__.__name__}{args}"


def implement_expr(
    n_args: int,
) -> Callable[[Type[UnevaluatedExpression]], sp.Expr]:
    """Decorator for classes that derive from `UnevaluatedExpression`.

    Implement a `~object.__new__` and `~sympy.core.basic.Basic.doit` method for
    a class that derives from `~sympy.core.expr.Expr` (via
    `UnevaluatedExpression`). It is important to derive from
    `~UnevaluatedExpression.evaluate` method has to be implemented
    """

    def decorator(decorated_class: Type[UnevaluatedExpression]) -> sp.Expr:
        def new_method(
            cls: Type,
            *args: sp.Symbol,
            **hints: Any,
        ) -> bool:
            if len(args) != n_args:
                raise ValueError(
                    f"{n_args} parameters expected, got {len(args)}"
                )
            args = sp.sympify(args)
            evaluate = hints.get("evaluate", False)
            if evaluate:
                return sp.Expr.__new__(cls, *args).evaluate()  # type: ignore  # pylint: disable=no-member
            return sp.Expr.__new__(cls, *args)

        def doit_method(self: Any, **hints: Any) -> sp.Expr:
            return type(self)(*self.args, **hints, evaluate=True)

        decorated_class.__new__ = new_method  # type: ignore
        decorated_class.doit = doit_method
        return decorated_class

    return decorator


@implement_expr(n_args=3)
class BlattWeisskopf(UnevaluatedExpression):
    r"""Blatt-Weisskopf function :math:`B_L`, up to :math:`L \leq 4`.

    Args:
        q: Break-up momentum. Can be computed with `breakup_momentum`.
        d: impact parameter :math:`d`, also called meson radius. Usually of the
            order 1 fm.
        angular_momentum: Angular momentum :math:`L` of the decaying particle.

    .. glue:math:: BlattWeisskopf
        :label: BlattWeisskopf

    Each of these cases has been taken from
    :cite:`chungPartialWaveAnalysis1995`, p. 415. For a good overview of where
    to use these Blatt-Weisskopf functions, see
    :cite:`asnerDalitzPlotAnalysis2006`.

    See also :ref:`usage/dynamics/lineshapes:Form factor`.
    """

    @property
    def q(self) -> sp.Symbol:
        """Break-up momentum."""
        return self.args[0]

    @property
    def d(self) -> sp.Symbol:
        """Impact parameter, also called meson radius."""
        return self.args[1]

    @property
    def angular_momentum(self) -> sp.Symbol:
        return self.args[2]

    def evaluate(self) -> sp.Expr:
        angular_momentum = self.angular_momentum
        z = (self.q * self.d) ** 2
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
            )
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        l, q = tuple(map(printer._print, (self.angular_momentum, self.q)))
        return fR"B_{l}\left({q}\right)"


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
    ff = BlattWeisskopf(q, meson_radius, angular_momentum)
    ff0 = BlattWeisskopf(q0, meson_radius, angular_momentum)
    mass_dependent_width = gamma0 * (mass0 / mass) * (ff ** 2 / ff0 ** 2)
    mass_dependent_width = mass_dependent_width * (q / q0)
    return (mass0 * gamma0 * ff) / (
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
