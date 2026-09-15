"""Implementations of the form factor, or barrier factor."""

from __future__ import annotations

from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any

import sympy as sp

from ampform.kinematics.phasespace import BreakupMomentumSquared
from ampform.sympy import argument, unevaluated

if TYPE_CHECKING:
    from collections.abc import Callable

    from sympy.printing.latex import LatexPrinter


@unevaluated
class FormFactor(sp.Expr):
    r"""Formulate a Blatt–Weisskopf form factor.

    Returns the `~sympy.functions.elementary.miscellaneous.sqrt` of a
    `.BlattWeisskopfSquared` with :math:`z = q^2 d^2`, where :math:`q^2` is the
    `.BreakupMomentumSquared` and :math:`d` is the meson radius. With
    ``normalize=False``, this is the production process factor :math:`n_a` from Equation
    (50.33) in :pdg-review:`2026; Resonances; p.12`, with :math:`d = 1/q_0`. The default
    normalized form factor :math:`\hat{\mathcal{F}}_L` differs from :math:`n_a` by the
    constant :math:`\left|h_L^{(1)}(1)\right|`. This constant cancels in ratios, such as
    in `.EnergyDependentWidth`, but not when the form factor is used as a vertex factor.
    """

    s: Any
    m1: Any
    m2: Any
    angular_momentum: Any
    meson_radius: Any = 1
    normalize: bool = argument(default=True, kw_only=True, sympify=False)

    def evaluate(self):
        s, m1, m2, angular_momentum, meson_radius = self.args
        q2 = BreakupMomentumSquared(s, m1, m2)
        ff_squared = BlattWeisskopfSquared(
            q2 * meson_radius**2, angular_momentum, normalize=self.normalize
        )
        return sp.sqrt(ff_squared)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s, m1, m2, angular_momentum = map(printer._print, self.args[:4])
        symbol = R"\hat{\mathcal{F}}" if self.normalize else R"\mathcal{F}"
        return Rf"{symbol}_{{{angular_momentum}}}\left({s}, {m1}, {m2}\right)"


@unevaluated
class BlattWeisskopfSquared(sp.Expr):
    r"""Normalized Blatt–Weisskopf function :math:`\hat{B}_L^2(z)`.

    Args:
        z: Argument of the Blatt–Weisskopf function. A usual choice is :math:`z = (d
            q)^2` with :math:`d` the impact parameter and :math:`q` the breakup-momentum
            (see `.BreakupMomentumSquared`).

        angular_momentum: Angular momentum :math:`L` of the decaying particle.

        normalize: Set to `False` to omit the normalization constant
            :math:`\left|h_L^{(1)}(1)\right|^2`. The resulting :math:`B_L^2(z)` equals
            :math:`z^L F_L^2(\sqrt{z})`, where :math:`F_L` is the non-normalized
            Blatt–Weisskopf function of Equation (50.34) in :pdg-review:`2026;
            Resonances; p.13`. This is the square of the factor :math:`n_L` of Equation
            (50.33).

    The hat indicates the normalization :math:`\hat{B}_L^2(1)=1`. Both variants have
    equal powers of :math:`z` in the numerator and the denominator, so they are unitless
    no matter what :math:`z` is. The PDG function :math:`F_L` instead has :math:`1` in
    the numerator and carries the threshold factor :math:`z^L` separately.

    >>> z = sp.Symbol("z", nonnegative=True)
    >>> BlattWeisskopfSquared(z, angular_momentum=2).doit()
    13*z**2/(z**2 + 3*z + 9)
    >>> BlattWeisskopfSquared(z, angular_momentum=2, normalize=False).doit()
    z**2/(z**2 + 3*z + 9)

    .. seealso:: :ref:`dynamics:Form factor`, :doc:`TR-029<compwa-report:029/index>`,
      and :cite:`Chung:2015-FormulasAngularMomentumBarrier`.

    With this, the implementation becomes
    """

    z: Any
    angular_momentum: Any
    normalize: bool = argument(default=True, kw_only=True, sympify=False)

    def evaluate(self) -> sp.Expr:
        z, ell = self.args
        if ell.free_symbols:
            return _formulate_blatt_weisskopf(ell, z, self.normalize)
        expr = _get_polynomial_blatt_weisskopf(ell, self.normalize)(z)
        return sp.sympify(expr)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        z, angular_momentum = map(printer._print, self.args)
        symbol = R"\hat{B}" if self.normalize else "B"
        return Rf"{symbol}_{{{angular_momentum}}}^2\left({z}\right)"


@lru_cache(maxsize=40)
def _get_polynomial_blatt_weisskopf(
    ell: int | sp.Integer, normalize: bool = True
) -> Callable[[Any], Any]:
    """Get the Blatt–Weisskopf factor as a fraction of polynomials.

    See https://github.com/ComPWA/ampform/issues/426.
    """
    z = sp.Symbol("z", nonnegative=True, real=True)
    expr = _formulate_blatt_weisskopf(ell, z, normalize)
    expr = expr.doit().simplify()
    return sp.lambdify(z, expr, "math")


def _formulate_blatt_weisskopf(ell, z, normalize: bool = True) -> sp.Expr:
    expr = 1 / sp.Abs(SphericalHankel1(ell, sp.sqrt(z))) ** 2 / z
    if normalize:
        return sp.Abs(SphericalHankel1(ell, z=1)) ** 2 * expr
    return expr


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

    l: Any  # ruff: ignore[ambiguous-variable-name]
    z: Any
    _latex_repr_ = R"h_{{{l}}}^{{(1)}}\left({z}\right)"

    def evaluate(self) -> sp.Expr:
        l, z = self.args  # ruff: ignore[ambiguous-variable-name]
        k = sp.Dummy("k", integer=True, nonnegative=True)
        return (
            (-sp.I) ** (1 + l)
            * (sp.exp(z * sp.I) / z)
            * _SymbolicSum(
                sp.factorial(l + k)
                / (sp.factorial(l - k) * sp.factorial(k))
                * (sp.I / (2 * z)) ** k,
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
