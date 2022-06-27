# pylint: disable=arguments-differ, abstract-method, protected-access,
# pylint: disable=too-many-arguments, unbalanced-tuple-unpacking
"""Functions for determining phase space boundaries.

.. seealso:: :doc:`/usage/kinematics`
"""
from __future__ import annotations

import sympy as sp

from ampform.sympy import (
    UnevaluatedExpression,
    create_expression,
    implement_doit_method,
    make_commutative,
)


@make_commutative
@implement_doit_method
class Kibble(UnevaluatedExpression):
    """Kibble function for determining the phase space region."""

    def __new__(cls, sigma1, sigma2, sigma3, m0, m1, m2, m3, **hints) -> Kibble:
        return create_expression(cls, sigma1, sigma2, sigma3, m0, m1, m2, m3, **hints)

    def evaluate(self) -> Kallen:
        sigma1, sigma2, sigma3, m0, m1, m2, m3 = self.args
        return Kallen(
            Kallen(sigma2, m2**2, m0**2),  # type: ignore[operator]
            Kallen(sigma3, m3**2, m0**2),  # type: ignore[operator]
            Kallen(sigma1, m1**2, m0**2),  # type: ignore[operator]
        )

    def _latex(self, printer, *args):
        sigma1, sigma2, *_ = map(printer._print, self.args)
        return Rf"\phi\left({sigma1}, {sigma2}\right)"


@make_commutative
@implement_doit_method
class Kallen(UnevaluatedExpression):
    """Källén function, used for computing break-up momenta."""

    def __new__(cls, x, y, z, **hints) -> Kallen:
        return create_expression(cls, x, y, z, **hints)

    def evaluate(self) -> sp.Expr:
        x, y, z = self.args
        return x**2 + y**2 + z**2 - 2 * x * y - 2 * y * z - 2 * z * x  # type: ignore[operator]

    def _latex(self, printer, *args):
        x, y, z = map(printer._print, self.args)
        return Rf"\lambda\left({x}, {y}, {z}\right)"


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
