"""Functions for determining phase space boundaries.

.. seealso:: :doc:`/usage/kinematics`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from ampform.sympy import argument, determine_indices, unevaluated
from ampform.sympy.math import ComplexSqrt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sympy.printing.latex import LatexPrinter


@unevaluated
class BreakupMomentum(sp.Expr):
    r"""Two-body break-up momentum.

    For a two-body decay :math:`R \to ab`, the *break-up momentum* is the absolute value
    of the momentum of both :math:`a` and :math:`b` in the rest frame of :math:`R`. See
    Equation (49.17) on :pdg-review:`2021; Kinematics; p.3`, as well as Equation (50.5)
    on :pdg-review:`2021; Resonances; p.5`.

    The numerator is represented as two square roots, as it gives a cleaner cut
    structure when the function is continued to the complex plane. The square root is
    defined as the standard :func:`sympy.sqrt
    <sympy.functions.elementary.miscellaneous.sqrt>`.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return (
            sp.sqrt(s - (m1 + m2) ** 2) * sp.sqrt(s - (m1 - m2) ** 2) / (2 * sp.sqrt(s))
        )

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = self.args[0]
        s_latex = printer._print(self.args[0])
        subscript = _indices_to_subscript(determine_indices(s))
        name = "q" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class BreakupMomentumComplex(sp.Expr):
    r"""Two-body break-up momentum.

    For a two-body decay :math:`R \to ab`, the *break-up momentum* is the absolute value
    of the momentum of both :math:`a` and :math:`b` in the rest frame of :math:`R`. See
    Equation (49.17) on :pdg-review:`2021; Kinematics; p.3`, as well as Equation (50.5)
    on :pdg-review:`2021; Resonances; p.5`.

    The numerator is represented as two square roots, as it gives a cleaner cut
    structure when the function is continued to the complex plane. The square root is
    the same as :func:`BreakupMomentum`, but using a `.ComplexSqrt` that does have
    defined behavior for defined for negative input values.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return (
            ComplexSqrt(s - (m1 + m2) ** 2)
            * ComplexSqrt(s - (m1 - m2) ** 2)
            / (2 * sp.sqrt(s))
        )

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = self.args[0]
        s_latex = printer._print(self.args[0])
        subscript = _indices_to_subscript(determine_indices(s))
        name = R"q^\mathrm{c}" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class BreakupMomentumSquared(sp.Expr):
    r"""Squared value of the two-body break-up momentum.

    For a two-body decay :math:`R \to ab`, the *break-up momentum* is the absolute value
    of the momentum of both :math:`a` and :math:`b` in the rest frame of :math:`R`. See
    Equation (49.17) on :pdg-review:`2021; Kinematics; p.3`, as well as Equation (50.5)
    on :pdg-review:`2021; Resonances; p.5`.

    It's up to the caller in which way to take the square root of this break-up
    momentum, because :math:`q^2` can have negative values for non-zero :math:`m1,m2`.
    In this case, one may want to use `.ComplexSqrt` instead of the standard
    :func:`~sympy.functions.elementary.miscellaneous.sqrt`.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (4 * s)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = self.args[0]
        s_latex = printer._print(self.args[0])
        subscript = _indices_to_subscript(determine_indices(s))
        name = "q^2" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


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
            Kallen(sigma2, m2**2, m0**2),  # type: ignore[operator]
            Kallen(sigma3, m3**2, m0**2),  # type: ignore[operator]
            Kallen(sigma1, m1**2, m0**2),  # type: ignore[operator]
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
        return x**2 + y**2 + z**2 - 2 * x * y - 2 * y * z - 2 * z * x  # type: ignore[operator]


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


def _indices_to_subscript(indices: Sequence[int]) -> str:
    """Create a LaTeX subscript from a list of indices.

    >>> _indices_to_subscript([])
    ''
    >>> _indices_to_subscript([1])
    '_{1}'
    >>> _indices_to_subscript([1, 2])
    '_{1,2}'
    """
    if len(indices) == 0:
        return ""
    subscript = ",".join(map(str, indices))
    return "_{" + subscript + "}"
