"""Functions for determining phase space boundaries.

.. seealso:: :doc:`/usage/kinematics`
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from ampform.sympy import unevaluated


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
