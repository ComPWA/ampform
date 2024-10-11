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

from ampform.sympy import argument, determine_indices, unevaluated
from ampform.sympy.math import ComplexSqrt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sympy.printing.latex import LatexPrinter

from typing import Protocol  # pragma: no cover


class PhaseSpaceFactorProtocol(Protocol):
    """Protocol that is used by `.EnergyDependentWidth`.

    Use this `~typing.Protocol` when defining other implementations of a phase space
    factor. Some implementations:

    - `PhaseSpaceFactor`
    - `PhaseSpaceFactorAbs`
    - `PhaseSpaceFactorComplex`
    - `PhaseSpaceFactorSWave`
    - `EqualMassPhaseSpaceFactor`

    Even `BreakupMomentumSquared` and :func:`chew_mandelstam_s_wave` comply with this
    protocol, but are technically speaking not phase space factors.
    """

    def __call__(self, s, m1, m2) -> sp.Expr:
        """Expected `~inspect.signature`.

        Args:
            s: :ref:`Mandelstam variable <pwa:mandelstam-variables>` :math:`s`.
                Commonly, this is just :math:`s = m_R^2`, with :math:`m_R` the invariant
                mass of decaying particle :math:`R`.

            m1: Mass of decay product :math:`a`.
            m2: Mass of decay product :math:`b`.
        """


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
        return (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (4 * s)  # type: ignore[operator]

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = self.args[0]
        s_latex = printer._print(self.args[0])
        subscript = _indices_to_subscript(determine_indices(s))
        name = "q^2" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class PhaseSpaceFactor(sp.Expr):
    r"""Standard phase-space factor, using :func:`BreakupMomentumSquared`.

    See :pdg-review:`2021; Resonances; p.6`, Equation (50.9). We ignore the factor
    :math:`\frac{1}{16\pi}` as done in :cite:`chungPrimerKmatrixFormalism1995`, p.5.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        q_squared = BreakupMomentumSquared(s, m1, m2)
        return 2 * sp.sqrt(q_squared) / sp.sqrt(s)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\rho" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class PhaseSpaceFactorAbs(sp.Expr):
    r"""Phase space factor square root over the absolute value.

    As opposed to `.PhaseSpaceFactor`, this takes the
    `~sympy.functions.elementary.complexes.Abs` value of `.BreakupMomentumSquared`, then
    the :func:`~sympy.functions.elementary.miscellaneous.sqrt`.

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

    Same as :func:`PhaseSpaceFactor`, but using a `.ComplexSqrt` that does have defined
    behavior for defined for negative input values.
    """

    s: Any
    m1: Any
    m2: Any
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m1, m2 = self.args
        q_squared = BreakupMomentumSquared(s, m1, m2)
        return 2 * ComplexSqrt(q_squared) / sp.sqrt(s)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s_symbol = self.args[0]
        s_latex = printer._print(s_symbol)
        subscript = _indices_to_subscript(determine_indices(s_symbol))
        name = R"\rho^\mathrm{c}" + subscript if self.name is None else self.name
        return Rf"{name}\left({s_latex}\right)"


@unevaluated
class PhaseSpaceFactorSWave(sp.Expr):
    r"""Phase space factor using :func:`chew_mandelstam_s_wave`.

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
    q_squared = BreakupMomentumSquared(s, m1, m2)
    q = ComplexSqrt(q_squared)
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
class EqualMassPhaseSpaceFactor(sp.Expr):
    """Analytic continuation for the :func:`PhaseSpaceFactor`.

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
        s_threshold = (m1 + m2) ** 2  # type: ignore[operator]
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
