"""Implementations of the form factor, or barrier factor."""

from __future__ import annotations

from functools import cache, lru_cache
from typing import Any, Callable

import sympy as sp

from ampform.dynamics.phasespace import BreakupMomentumSquared
from ampform.sympy import unevaluated


@unevaluated
class FormFactor(sp.Expr):
    """Formulate a Blatt-Weisskopf form factor.

    Returns the production process factor :math:`n_a` from Equation (50.26) in
    :pdg-review:`2021; Resonances; p.9`, which features the
    `~sympy.functions.elementary.miscellaneous.sqrt` of a `.BlattWeisskopfSquared`.
    """

    s: Any
    m1: Any
    m2: Any
    angular_momentum: Any
    meson_radius: Any = 1

    _latex_repr_ = R"\mathcal{{F}}_{{{angular_momentum}}}\left({s}, {m1}, {m2}\right)"

    def evaluate(self):
        s, m1, m2, angular_momentum, meson_radius = self.args
        q2 = BreakupMomentumSquared(s, m1, m2)
        ff_squared = BlattWeisskopfSquared(q2 * meson_radius**2, angular_momentum)
        return sp.sqrt(ff_squared)


@unevaluated
class BlattWeisskopfSquared(sp.Expr):
    r"""Normalized Blatt-Weisskopf function :math:`B_L^2(z)`, with :math:`B_L^2(1)=1`.

    Args:
        z: Argument of the Blatt-Weisskopf function :math:`B_L^2(z)`. A usual
            choice is :math:`z = (d q)^2` with :math:`d` the impact parameter and
            :math:`q` the breakup-momentum (see `.BreakupMomentumSquared`).

        angular_momentum: Angular momentum :math:`L` of the decaying particle.

    Note that equal powers of :math:`z` appear in the nominator and the denominator,
    while some sources define an *non-normalized* form factor :math:`F_L` with :math:`1`
    in the nominator, instead of :math:`z^L`. See for instance Equation (50.27) in
    :pdg-review:`2021; Resonances; p.9`. We normalize the form factor such that
    :math:`B_L^2(1)=1` and that :math:`B_L^2` is unitless no matter what :math:`z` is.

    .. seealso:: :ref:`usage/dynamics:Form factor`, :doc:`TR-029<compwa-report:029/index>`,
      and :cite:`chungFormulasAngularMomentumBarrier2015`.

    With this, the implementation becomes
    """

    z: Any
    angular_momentum: Any
    _latex_repr_ = R"B_{{{angular_momentum}}}^2\left({z}\right)"

    def evaluate(self) -> sp.Expr:
        z, ell = self.args
        if ell.free_symbols:
            return _formulate_blatt_weisskopf(ell, z)
        expr = _get_polynomial_blatt_weisskopf(ell)(z)
        return sp.sympify(expr)


@lru_cache(maxsize=20)
def _get_polynomial_blatt_weisskopf(ell: int | sp.Integer) -> Callable[[Any], Any]:
    """Get the Blatt-Weisskopf factor as a fraction of polynomials.

    See https://github.com/ComPWA/ampform/issues/426.
    """
    z = sp.Symbol("z", nonnegative=True, real=True)
    expr = _formulate_blatt_weisskopf(ell, z)
    expr = expr.doit().simplify()
    return sp.lambdify(z, expr, "math")


def _formulate_blatt_weisskopf(ell, z) -> sp.Expr:
    return (
        sp.Abs(SphericalHankel1(ell, 1)) ** 2
        / sp.Abs(SphericalHankel1(ell, sp.sqrt(z))) ** 2
        / z
    )


@unevaluated
class SphericalHankel1(sp.Expr):
    r"""Spherical Hankel function of the first kind for real-valued :math:`z`.

    See :cite:`VonHippel:1972fg`, Equation (A12), and :doc:`TR-029<compwa-report:029/index>`
    for more info. `This page
    <https://mathworld.wolfram.com/SphericalHankelFunctionoftheFirstKind.html>`_
    explains the difference with the *general* Hankel function of the first kind,
    :math:`H_\ell^{(1)}`.

    This expression class assumes that :math:`z` is real and evaluates to the following
    series:
    """

    l: Any  # noqa: E741
    z: Any
    _latex_repr_ = R"h_{{{l}}}^{{(1)}}\left({z}\right)"

    def evaluate(self) -> sp.Expr:
        l, z = self.args  # noqa: E741
        k = sp.Dummy("k", integer=True, nonnegative=True)
        return (
            (-sp.I) ** (1 + l)  # type:ignore[operator]
            * (sp.exp(z * sp.I) / z)
            * _SymbolicSum(
                sp.factorial(l + k)
                / (sp.factorial(l - k) * sp.factorial(k))
                * (sp.I / (2 * z)) ** k,  # type:ignore[operator]
                (k, 0, l),
            )
        )


class _SymbolicSum(sp.Sum):
    """See [TR-029](https://compwa.github.io/report/029.html) for why this class is needed."""

    def doit(self, deep: bool = True, **kwargs) -> sp.Expr:
        if _get_indices(self):
            expression = self.args[0]
            indices = self.args[1:]
            return _SymbolicSum(expression.doit(deep=deep, **kwargs), *indices)
        return super().doit(deep=deep, **kwargs)


@cache
def _get_indices(expr: sp.Sum) -> set[sp.Basic]:
    free_symbols = set()
    for index in expr.args[1:]:
        free_symbols.update(index.free_symbols)
    return {s for s in free_symbols if not isinstance(s, sp.Dummy)}
