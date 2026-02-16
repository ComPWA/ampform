"""Functions for determining phase space boundaries.

.. seealso:: :doc:`/usage/kinematics`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from ampform.sympy import argument, determine_indices, unevaluated
from ampform.sympy.math import ComplexSqrt

if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter


@unevaluated
class BreakupMomentum(sp.Expr):
    r"""Break-up momentum of a two-body decay.

    For a two-body decay :math:`R \to 12`, the *break-up momentum* is the absolute value
    of the momentum of both :math:`1` and :math:`2` in the rest frame of :math:`R`. See
    Equation (50.7) on :pdg-review:`2024; Resonances; p.7`.

    In AmpForm's standard implementation, the numerator is represented as a single
    square root. This results in :ref:`better computational performance
    <usage/dynamics/analytic-continuation:Numerical precision and performance>`, as the
    expression tree has fewer computational nodes, but comes at the cost of a :ref:`more
    complicated cut structure <usage/dynamics/analytic-continuation:Cut structure>` when
    the function is continued to the complex plane. The square root itself is defined as
    the standard :func:`sympy.sqrt <sympy.functions.elementary.miscellaneous.sqrt>`.

    Alternative implementations:

    * `BreakupMomentumComplex`
    * `BreakupMomentumKallen`
    * `BreakupMomentumSplitSqrt`
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, kw_only=True, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return sp.sqrt((s - (m1 - m2) ** 2) * (s - (m1 + m2) ** 2)) / (2 * sp.sqrt(s))

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        name = self.name or f"q{_get_subscript(self.s)}"
        return Rf"{name}\left({s}\right)"


@unevaluated
class BreakupMomentumKallen(sp.Expr):
    """Two-body break-up momentum with a Källén function.

    This version of the `BreakupMomentum` represents the numerator using the `.Kallen`
    function. This is common practice in literature (e.g. :pdg-review:`2024; Resonances;
    p.7`), but results in a :ref:`more complicated cut
    <usage/dynamics/analytic-continuation:Cut structure>` and :ref:`worse numerical
    performance <usage/dynamics/analytic-continuation:Numerical precision and
    performance>` than `BreakupMomentum`.
    """

    s: Any
    m1: Any
    m2: Any

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return sp.sqrt(Kallen(s, m1**2, m2**2) / (4 * s))

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        return Rf"q\left({s}\right)"


@unevaluated
class BreakupMomentumSplitSqrt(sp.Expr):
    """Two-body break-up momentum with cut structure.

    This version of the `BreakupMomentum` represents the numerator as two separate
    square roots. This results in a :ref:`cleaner cut structure
    <usage/dynamics/analytic-continuation:Cut structure>` at the cost of :ref:`slightly
    worse numerical performance <usage/dynamics/analytic-continuation:Numerical
    precision and performance>` than `BreakupMomentum`.
    """

    s: Any
    m1: Any
    m2: Any

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return (
            sp.sqrt(s - (m1 + m2) ** 2) * sp.sqrt(s - (m1 - m2) ** 2) / (2 * sp.sqrt(s))
        )

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        return Rf"q\left({s}\right)"


@unevaluated
class BreakupMomentumComplex(sp.Expr):
    """Two-body break-up momentum with a square root that is defined on the real axis.

    In this version of the `BreakupMomentumSplitSqrt`, the square roots are replaced by
    `.ComplexSqrt`, which has a defined behavior for negative input values, so that it
    can be evaluated on the entire real axis.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, kw_only=True, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return (
            ComplexSqrt(s - (m1 + m2) ** 2)
            * ComplexSqrt(s - (m1 - m2) ** 2)
            / (2 * sp.sqrt(s))
        )

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        name = self.name or Rf"q^\mathrm{{c}}{_get_subscript(self.s)}"
        return Rf"{name}\left({s}\right)"


@unevaluated
class BreakupMomentumSquared(sp.Expr):
    """Squared value of the two-body `BreakupMomentum`.

    It's up to the caller in which way to take the square root of this break-up
    momentum, because :math:`q^2` can have negative values for non-zero :math:`m_1,m_2`.
    In this case, one may want to use `.ComplexSqrt` instead of the standard
    :func:`~sympy.functions.elementary.miscellaneous.sqrt`.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, kw_only=True, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (4 * s)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        name = self.name or f"q^2{_get_subscript(self.s)}"
        return Rf"{name}\left({s}\right)"


@unevaluated
class Kibble(sp.Expr):
    """Kibble function for determining the phase space region."""

    sigma1: Any
    sigma2: Any
    sigma3: Any
    m0: Any
    m1: Any
    m2: Any
    m3: Any
    _latex_repr_ = R"\phi\left({sigma1}, {sigma2}\right)"

    def evaluate(self) -> Kallen:
        sigma1, sigma2, sigma3, m0, m1, m2, m3 = self.args
        return Kallen(
            Kallen(sigma2, m2**2, m0**2),
            Kallen(sigma3, m3**2, m0**2),
            Kallen(sigma1, m1**2, m0**2),
        )


@unevaluated
class Kallen(sp.Expr):
    """Källén function, used for computing break-up momenta."""

    x: Any
    y: Any
    z: Any
    _latex_repr_ = R"\lambda\left({x}, {y}, {z}\right)"

    def evaluate(self) -> sp.Expr:
        x, y, z = self.args
        return x**2 + y**2 + z**2 - 2 * x * y - 2 * y * z - 2 * z * x


def is_within_phasespace(
    sigma1, sigma2, m0, m1, m2, m3, outside_value=sp.nan
) -> sp.Piecewise:
    """Determine whether a set of masses lie within phase space."""
    sigma3 = compute_third_mandelstam(sigma1, sigma2, m0, m1, m2, m3)
    kibble = Kibble(sigma1, sigma2, sigma3, m0, m1, m2, m3)
    return sp.Piecewise(
        (1, sp.LessThan(kibble, 0)),
        (outside_value, True),
    )


def compute_third_mandelstam(sigma1, sigma2, m0, m1, m2, m3) -> sp.Add:
    """Compute the third Mandelstam variable in a three-body decay."""
    return m0**2 + m1**2 + m2**2 + m3**2 - sigma1 - sigma2


def _get_subscript(symbol: sp.Basic, /, *, superscript: bool = False) -> str:
    r"""Get the subscript name for a symbol.

    >>> _get_subscript(sp.symbols("s"))
    ''
    >>> _get_subscript(sp.symbols("sigma1"))
    '_{1}'
    >>> _get_subscript(sp.symbols("x12"))
    '_{12}'
    >>> _get_subscript(sp.symbols(R"x^k_{1,2}"))
    '_{1,2}'
    >>> _get_subscript(sp.symbols(R"\alpha_{i2, 5}"))
    '_{2,5}'
    """
    indices = determine_indices(symbol)
    if len(indices) == 0:
        return ""
    subscript = ",".join(map(str, indices))
    return "_{" + subscript + "}"
