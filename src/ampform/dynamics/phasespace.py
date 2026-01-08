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

from typing import TYPE_CHECKING, Any

import sympy as sp

from ampform.kinematics.phasespace import Kallen
from ampform.sympy import argument, determine_indices, unevaluated
from ampform.sympy.math import ComplexSqrt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sympy.printing.latex import LatexPrinter

from typing import Protocol  # pragma: no cover


class PhaseSpaceFactorProtocol(Protocol):
    """Protocol that is used by `.EnergyDependentWidth`.

    Follow this `~typing.Protocol` when defining other implementations of a phase space.
    Even functions like `BreakupMomentum` comply with this protocol, but are technically
    speaking not phase space factors.
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
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        return sp.sqrt((s - (m1 - m2) ** 2) * (s - (m1 + m2) ** 2)) / (2 * sp.sqrt(s))

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = self.args[0]
        s_latex = printer._print(self.args[0])
        subscript = _indices_to_subscript(determine_indices(s))
        name = "q" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


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
        s = self.args[0]
        s_latex = printer._print(s)
        return Rf"q\left({s_latex}\right)"


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
        s = self.args[0]
        s_latex = printer._print(s)
        return Rf"q\left({s_latex}\right)"


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
    """Squared value of the two-body `BreakupMomentum`.

    It's up to the caller in which way to take the square root of this break-up
    momentum, because :math:`q^2` can have negative values for non-zero :math:`m_1,m_2`.
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
class PhaseSpaceFactor(sp.Expr):
    r"""Standard phase-space factor, using a definition consistent with `BreakupMomentum`.

    See :pdg-review:`2024; Resonances; p.6`, Equation (50.11). We ignore the factor
    :math:`\frac{1}{16\pi}` as done in :cite:`Chung:1995-PrimerKmatrixFormalism`, p.5.

    Similarly to `BreakupMomentum`, this class represents the numerator as a single
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
class PhaseSpaceFactorKallen(sp.Expr):
    """Phase-space factor that is the equivalent of `BreakupMomentumKallen`."""

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
    """Phase-space factor that is the equivalent of `BreakupMomentumSplitSqrt`.

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
