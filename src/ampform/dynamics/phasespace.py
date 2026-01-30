"""Different parametrizations of phase space factors.

Phase space factors are computed by integrating over the phase space element given by
Equation (49.12) in :pdg-review:`2021; Kinematics; p.2`. See also Equation (50.9) on
:pdg-review:`2021; Resonances; p.6`. This integral is not always easy to solve, which
leads to different parametrizations.

This module provides several parametrizations. They all comply with the
`PhaseSpaceFactorProtocol`, so that they can be used in parametrizations like
`.EnergyDependentWidth`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol  # pragma: no cover

import sympy as sp

from ampform.dynamics.form_factor import FormFactor
from ampform.kinematics.phasespace import (
    BreakupMomentum,  # noqa: F401  # pyright:ignore[reportUnusedImport]
    BreakupMomentumComplex,
    BreakupMomentumKallen,  # noqa: F401  # pyright:ignore[reportUnusedImport]
    BreakupMomentumSplitSqrt,  # noqa: F401  # pyright:ignore[reportUnusedImport]
    BreakupMomentumSquared,
    Kallen,
    _indices_to_subscript,
)
from ampform.sympy import (
    UnevaluatableIntegral,
    argument,
    determine_indices,
    unevaluated,
)
from ampform.sympy.math import ComplexSqrt

if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter


class PhaseSpaceFactorProtocol(Protocol):
    """Protocol that is used by `.EnergyDependentWidth`.

    Follow this `~typing.Protocol` when defining other implementations of a phase space.
    Even functions like `.BreakupMomentum` comply with this protocol, but are
    technically speaking not phase space factors.
    """

    def __call__(self, s, m1, m2) -> sp.Expr:
        """Expected `~inspect.signature`.

        Args:
            s: `Mandelstam variable <https://pwa.rtfd.io/physics/#mandelstam-variables>`_
                :math:`s`. Commonly, this is just :math:`s = m_R^2`, with :math:`m_R`
                the invariant mass of decaying particle :math:`R`.

            m1: Mass of decay product :math:`a`.
            m2: Mass of decay product :math:`b`.
        """


@unevaluated
class PhaseSpaceFactor(sp.Expr):
    r"""Standard phase-space factor, using a definition consistent with `.BreakupMomentum`.

    See :pdg-review:`2025; Resonances; p.6`, Equation (50.11). We ignore the factor
    :math:`\frac{1}{16\pi}` as done in :cite:`Chung:1995-PrimerKmatrixFormalism`, p.5.

    Similarly to `.BreakupMomentum`, this class represents the numerator as a single
    square root for better numerical performance. This comes at the cost of a :ref:`more
    complicated cut structure <usage/dynamics/analytic-continuation:Cut structure>` when
    the function is continued to the complex plane.

    Alternative implementations:

    * `PhaseSpaceFactorAbs`
    * `PhaseSpaceFactorComplex`
    * `PhaseSpaceFactorKallen`
    * `PhaseSpaceFactorSplitSqrt`
    * `PhaseSpaceFactorSWave`
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return sp.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)) / s

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\rho" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class PhaseSpaceFactorAbs(sp.Expr):
    r"""Phase space factor with square root over the absolute value.

    As opposed to `.PhaseSpaceFactor`, this takes the square root of the
    `~sympy.functions.elementary.complexes.Abs` value of `.BreakupMomentumSquared`.

    This version of the phase space factor is often denoted as :math:`\hat{\rho}` and is
    used in `.EqualMassPhaseSpaceFactor`.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        q_squared = BreakupMomentumSquared(s, m1, m2)
        return 2 * sp.sqrt(sp.Abs(q_squared)) / sp.sqrt(s)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\hat{\rho}" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class PhaseSpaceFactorComplex(sp.Expr):
    """Phase-space factor with `.ComplexSqrt`.

    Same as `PhaseSpaceFactorSplitSqrt`, but using a `.ComplexSqrt` that does have
    defined behavior for defined for negative input values along the real axis.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return ComplexSqrt(s - (m1 + m2) ** 2) * ComplexSqrt(s - (m1 - m2) ** 2) / s

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\rho^\mathrm{c}" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class PhaseSpaceFactorKallen(sp.Expr):
    """Phase-space factor that is the equivalent of `.BreakupMomentumKallen`."""

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return sp.sqrt(Kallen(s, m1**2, m2**2)) / s

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\rho" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class PhaseSpaceFactorSplitSqrt(sp.Expr):
    """Phase-space factor that is the equivalent of `.BreakupMomentumSplitSqrt`.

    This version of the `PhaseSpaceFactor` represents the numerator as two separate
    square roots. This results in a :ref:`cleaner cut structure
    <usage/dynamics/analytic-continuation:Cut structure>` at the cost of :ref:`slightly
    worse numerical performance <usage/dynamics/analytic-continuation:Numerical
    precision and performance>` than `PhaseSpaceFactor`.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return sp.sqrt(s - (m1 + m2) ** 2) * sp.sqrt(s - (m1 - m2) ** 2) / s

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\rho" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class PhaseSpaceFactorSWave(sp.Expr):
    """Phase space factor using :func:`chew_mandelstam_s_wave`.

    This `PhaseSpaceFactor` provides an analytic continuation for decay products with
    both equal and unequal masses (compare `EqualMassPhaseSpaceFactor`).
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        chew_mandelstam = chew_mandelstam_s_wave(s, m1, m2)
        return -sp.I * chew_mandelstam

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\rho^\mathrm{CM}" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class PhaseSpaceFactorPWave(sp.Expr):
    r"""Phase space factor using `ChewMandelstamIntegral` for :math:`L=1`.

    This `PhaseSpaceFactor` uses the numerical dispersion integral implemented in
    `ChewMandelstamIntegral`. As such, you have to be careful when lambdifying this
    function and evaluating this over an array. In many cases, you want to wrap the
    resulting lambdified numerical function with :obj:`numpy.vectorize`.

    >>> import numpy as np
    >>> from ampform.dynamics.phasespace import PhaseSpaceFactorPWave
    >>> s, m1, m2 = sp.symbols("s m_1 m_2")
    >>> rho_expr = PhaseSpaceFactorPWave(s, m1, m2)
    >>> rho_func = sp.lambdify((s, m1, m2), rho_expr.doit())
    >>> rho_func = np.vectorize(rho_func)
    >>> s_values = np.linspace(0.1, 4.0, num=4)
    >>> rho_func(s_values, 0.14, 0.98).real
    array([-4.08315014e-07,  8.05561163e-03,  2.65015019e-01,  5.43083429e-01])
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        chew_mandelstam = ChewMandelstamIntegral(s, m1, m2, L=1, epsilon=1e-5)
        return -sp.I * chew_mandelstam

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\rho^\mathrm{CM}_1" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


def chew_mandelstam_s_wave(s, m1, m2):
    """Chew-Mandelstam function for :math:`S`-waves (no angular momentum)."""
    q = BreakupMomentumComplex(s, m1, m2)
    left_term = sp.Mul(
        2 * q / sp.sqrt(s),
        sp.log((m1**2 + m2**2 - s + 2 * sp.sqrt(s) * q) / (2 * m1 * m2)),
        evaluate=False,
    )
    right_term = (m1**2 - m2**2) * (1 / s - 1 / (m1 + m2) ** 2) * sp.log(m1 / m2)
    # evaluate=False in order to keep same style as PDG
    return sp.Mul(
        1 / sp.pi,
        left_term - right_term,
        evaluate=False,
    )


@unevaluated
class ChewMandelstamIntegral(sp.Expr):
    """Dispersion integral for obtaining the analytic phase space factor for angular momenta L>0.

    Parameters:
        s: Mandelstam variable s.
        m1: Mass of particle 1.
        m2: Mass of particle 2.
        L: Angular momentum.
        meson_radius: Meson radius, default is 1 (optional).
        s_prime: Integration variable defaults to 'x'.
        epsilon: Small imaginary part default is positive epsilon.
    """

    s: Any
    m1: Any
    m2: Any
    L: Any
    s_prime: Any = sp.Symbol("x", real=True)
    epsilon: Any = sp.Symbol("epsilon", positive=True)
    meson_radius: Any = 1
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2, L, s_prime, epsilon, meson_radius, *_ = self.args  # noqa: N806
        ff_squared = FormFactor(s_prime, m1, m2, L, meson_radius) ** 2
        phsp_factor = PhaseSpaceFactor(s_prime, m1, m2)
        s_thr = (m1 + m2) ** 2
        return sp.Mul(
            (s - s_thr) / sp.pi,
            UnevaluatableIntegral(
                (phsp_factor * ff_squared)
                / (s_prime - s_thr)
                / (s_prime - s - sp.I * epsilon),
                (s_prime, s_thr, sp.oo),
            ),
            evaluate=False,
        )

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_latex = printer._print(self.s)
        l_latex = printer._print(self.L)
        subscript = _indices_to_subscript(determine_indices(self.s))
        name = Rf"\Sigma_{l_latex}" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class EqualMassPhaseSpaceFactor(sp.Expr):
    """Analytic continuation for the `PhaseSpaceFactor`.

    See :pdg-review:`2018; Resonances; p.9` and
    :doc:`/usage/dynamics/analytic-continuation`.

    **Warning**: The PDG specifically derives this formula for a two-body decay *with
    equal masses*.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        rho_hat = PhaseSpaceFactorAbs(s, m1, m2)
        s_threshold = (m1 + m2) ** 2
        return _analytic_continuation(rho_hat, s, s_threshold)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\rho^\mathrm{eq}" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


def _analytic_continuation(rho, s, s_threshold) -> sp.Piecewise:
    return sp.Piecewise(
        (
            sp.I * rho / sp.pi * sp.log(sp.Abs((1 + rho) / (1 - rho))),
            s < 0,
        ),
        (
            rho + sp.I * rho / sp.pi * sp.log(sp.Abs((1 + rho) / (1 - rho))),
            s > s_threshold,
        ),
        (
            2 * sp.I * rho / sp.pi * sp.atan(1 / rho),
            True,
        ),
    )
