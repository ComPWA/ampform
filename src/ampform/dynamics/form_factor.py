"""Implementations of the form factor, or barrier factor."""

from __future__ import annotations

from typing import Any

import sympy as sp

from ampform.sympy import unevaluated


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

    .. seealso:: :ref:`usage/dynamics:Form factor`, :doc:`TR-029<compwa:report/029>`,
      and :cite:`chungFormulasAngularMomentumBarrier2015`.

    With this, the implementation becomes
    """

    z: Any
    angular_momentum: Any
    _latex_repr_ = R"B_{{{angular_momentum}}}^2\left({z}\right)"

    def evaluate(self) -> sp.Expr:
        z, angular_momentum = self.args
        return (
            sp.Abs(SphericalHankel1(angular_momentum, 1)) ** 2
            / sp.Abs(SphericalHankel1(angular_momentum, sp.sqrt(z))) ** 2
            / z
        )


@unevaluated(implement_doit=False)
class SphericalHankel1(sp.Expr):
    r"""Spherical Hankel function of the first kind for real-valued :math:`z`.

    See :cite:`VonHippel:1972fg`, Equation (A12), and :doc:`TR-029<compwa:report/029>`
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

    def doit(self, deep: bool = True, **kwargs):
        expr = self.evaluate()
        if deep and isinstance(self.l, sp.Integer):
            return expr.doit()
        return expr

    def evaluate(self) -> sp.Expr:
        l, z = self.args  # noqa: E741
        k = sp.Dummy("k", integer=True, nonnegative=True)
        return (
            (-sp.I) ** (1 + l)  # type:ignore[operator]
            * (sp.exp(z * sp.I) / z)
            * sp.Sum(
                sp.factorial(l + k)
                / (sp.factorial(l - k) * sp.factorial(k))
                * (sp.I / (2 * z)) ** k,  # type:ignore[operator]
                (k, 0, l),
            )
        )
