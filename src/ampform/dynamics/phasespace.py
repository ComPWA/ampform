# pylint: disable=abstract-method, arguments-differ, protected-access
# pylint: disable=unbalanced-tuple-unpacking
"""Different parametrizations of phase space factors.

Phase space factors are computed by integrating over the phase space element
given by Equation (49.12) in :pdg-review:`2021; Kinematics; p.2`. See also
Equation (50.9) on :pdg-review:`2021; Resonances; p.6`. This integral is not
always easy to solve, which leads to different parametrizations.

This module provides several parametrizations. They all comply with the
`PhaseSpaceFactorProtocol`, so that they can be used in parametrizations like
`.EnergyDependentWidth`.
"""
from __future__ import annotations

import re
import sys
from typing import Sequence

import sympy as sp
from sympy.printing.conventions import split_super_sub
from sympy.printing.latex import LatexPrinter

from ampform.sympy import (
    UnevaluatedExpression,
    create_expression,
    implement_doit_method,
)
from ampform.sympy.math import ComplexSqrt

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol  # pragma: no cover


class PhaseSpaceFactorProtocol(Protocol):
    """Protocol that is used by `.EnergyDependentWidth`.

    Use this `~typing.Protocol` when defining other implementations of a phase
    space factor. Some implementations:

    - `PhaseSpaceFactor`
    - `PhaseSpaceFactorAbs`
    - `PhaseSpaceFactorComplex`
    - `PhaseSpaceFactorSWave`
    - `EqualMassPhaseSpaceFactor`

    Even `BreakupMomentumSquared` and :func:`chew_mandelstam_s_wave` comply
    with this protocol, but are technically speaking not phase space factors.
    """

    def __call__(self, s, m_a, m_b) -> sp.Expr:
        """Expected `~inspect.signature`.

        Args:
            s: :ref:`Mandelstam variable <pwa:mandelstam-variables>` :math:`s`.
                Commonly, this is just :math:`s = m_R^2`, with :math:`m_R` the
                invariant mass of decaying particle :math:`R`.

            m_a: Mass of decay product :math:`a`.
            m_b: Mass of decay product :math:`b`.
        """


@implement_doit_method
class BreakupMomentumSquared(UnevaluatedExpression):
    r"""Squared value of the two-body break-up momentum.

    For a two-body decay :math:`R \to ab`, the *break-up momentum* is the
    absolute value of the momentum of both :math:`a` and :math:`b` in the rest
    frame of :math:`R`. See Equation (49.17) on :pdg-review:`2021; Kinematics;
    p.3`, as well as Equation (50.5) on :pdg-review:`2021; Resonances; p.5`.

    It's up to the caller in which way to take the square root of this break-up
    momentum, because :math:`q^2` can have negative values for non-zero
    :math:`m_a,m_b`. In this case, one may want to use `.ComplexSqrt` instead
    of the standard :func:`~sympy.functions.elementary.miscellaneous.sqrt`.
    """

    is_commutative = True

    def __new__(cls, s, m_a, m_b, **hints) -> BreakupMomentumSquared:
        return create_expression(cls, s, m_a, m_b, **hints)

    def evaluate(self) -> sp.Expr:
        s, m_a, m_b = self.args
        return (s - (m_a + m_b) ** 2) * (s - (m_a - m_b) ** 2) / (4 * s)  # type: ignore[operator]

    def _latex(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.args[0])
        subscript = _indices_to_subscript(_determine_indices(s))
        name = "q^2" + subscript if self._name is None else self._name
        return Rf"{name}\left({s}\right)"


@implement_doit_method
class PhaseSpaceFactor(UnevaluatedExpression):
    """Standard phase-space factor, using :func:`BreakupMomentumSquared`.

    See :pdg-review:`2021; Resonances; p.6`, Equation (50.9).
    """

    is_commutative = True

    def __new__(cls, s, m_a, m_b, **hints) -> PhaseSpaceFactor:
        return create_expression(cls, s, m_a, m_b, **hints)

    def evaluate(self) -> sp.Expr:
        s, m_a, m_b = self.args
        q_squared = BreakupMomentumSquared(s, m_a, m_b)
        denominator = _phase_space_factor_denominator(s)
        return sp.sqrt(q_squared) / denominator

    def _latex(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.args[0])
        subscript = _indices_to_subscript(_determine_indices(s))
        name = R"\rho" + subscript if self._name is None else self._name
        return Rf"{name}\left({s}\right)"


@implement_doit_method
class PhaseSpaceFactorAbs(UnevaluatedExpression):
    r"""Phase space factor square root over the absolute value.

    As opposed to `.PhaseSpaceFactor`, this takes the
    `~sympy.functions.elementary.complexes.Abs` value of
    `.BreakupMomentumSquared`, then the
    :func:`~sympy.functions.elementary.miscellaneous.sqrt`.

    This version of the phase space factor is often denoted as
    :math:`\hat{\rho}` and is used in `.EqualMassPhaseSpaceFactor`.
    """

    is_commutative = True

    def __new__(cls, s, m_a, m_b, **hints) -> PhaseSpaceFactorAbs:
        return create_expression(cls, s, m_a, m_b, **hints)

    def evaluate(self) -> sp.Expr:
        s, m_a, m_b = self.args
        q_squared = BreakupMomentumSquared(s, m_a, m_b)
        denominator = _phase_space_factor_denominator(s)
        return sp.sqrt(sp.Abs(q_squared)) / denominator

    def _latex(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.args[0])
        subscript = _indices_to_subscript(_determine_indices(s))
        name = R"\hat{\rho}" + subscript if self._name is None else self._name
        return Rf"{name}\left({s}\right)"


@implement_doit_method
class PhaseSpaceFactorComplex(UnevaluatedExpression):
    """Phase-space factor with `.ComplexSqrt`.

    Same as :func:`PhaseSpaceFactor`, but using a `.ComplexSqrt` that does have
    defined behavior for defined for negative input values.
    """

    is_commutative = True

    def __new__(cls, s, m_a, m_b, **hints) -> PhaseSpaceFactorComplex:
        return create_expression(cls, s, m_a, m_b, **hints)

    def evaluate(self) -> sp.Expr:
        s, m_a, m_b = self.args
        q_squared = BreakupMomentumSquared(s, m_a, m_b)
        denominator = _phase_space_factor_denominator(s)
        return ComplexSqrt(q_squared) / denominator

    def _latex(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.args[0])
        subscript = _indices_to_subscript(_determine_indices(s))
        name = (
            R"\rho^\mathrm{c}" + subscript
            if self._name is None
            else self._name
        )
        return Rf"{name}\left({s}\right)"


@implement_doit_method
class PhaseSpaceFactorSWave(UnevaluatedExpression):
    r"""Phase space factor using :func:`chew_mandelstam_s_wave`.

    This `PhaseSpaceFactor` provides an analytic continuation for decay
    products with both equal and unequal masses (compare
    `EqualMassPhaseSpaceFactor`).
    """

    is_commutative = True

    def __new__(cls, s, m_a, m_b, **hints) -> PhaseSpaceFactorSWave:
        return create_expression(cls, s, m_a, m_b, **hints)

    def evaluate(self) -> sp.Expr:
        s, m_a, m_b = self.args
        chew_mandelstam = chew_mandelstam_s_wave(s, m_a, m_b)
        return -sp.I * chew_mandelstam

    def _latex(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.args[0])
        subscript = _indices_to_subscript(_determine_indices(s))
        name = (
            R"\rho^\mathrm{CM}" + subscript
            if self._name is None
            else self._name
        )
        return Rf"{name}\left({s}\right)"


def chew_mandelstam_s_wave(s, m_a, m_b):
    """Chew-Mandelstam function for :math:`S`-waves (no angular momentum)."""
    q_squared = BreakupMomentumSquared(s, m_a, m_b)
    q = ComplexSqrt(q_squared)
    left_term = sp.Mul(
        2 * q / sp.sqrt(s),
        sp.log(
            (m_a**2 + m_b**2 - s + 2 * sp.sqrt(s) * q) / (2 * m_a * m_b)
        ),
        evaluate=False,
    )
    right_term = (
        (m_a**2 - m_b**2)
        * (1 / s - 1 / (m_a + m_b) ** 2)
        * sp.log(m_a / m_b)
    )
    # evaluate=False in order to keep same style as PDG
    return sp.Mul(
        1 / (16 * sp.pi**2),
        left_term - right_term,
        evaluate=False,
    )


@implement_doit_method
class EqualMassPhaseSpaceFactor(UnevaluatedExpression):
    """Analytic continuation for the :func:`PhaseSpaceFactor`.

    See :pdg-review:`2018; Resonances; p.9` and
    :doc:`/usage/dynamics/analytic-continuation`.

    **Warning**: The PDG specifically derives this formula for a two-body decay
    *with equal masses*.
    """

    is_commutative = True

    def __new__(cls, s, m_a, m_b, **hints) -> EqualMassPhaseSpaceFactor:
        return create_expression(cls, s, m_a, m_b, **hints)

    def evaluate(self) -> sp.Expr:
        s, m_a, m_b = self.args
        rho_hat = PhaseSpaceFactorAbs(s, m_a, m_b)
        s_threshold = (m_a + m_b) ** 2  # type: ignore[operator]
        return _analytic_continuation(rho_hat, s, s_threshold)

    def _latex(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.args[0])
        subscript = _indices_to_subscript(_determine_indices(s))
        name = (
            R"\rho^\mathrm{eq}" + subscript
            if self._name is None
            else self._name
        )
        return Rf"{name}\left({s}\right)"


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


def _phase_space_factor_denominator(s) -> sp.Mul:
    return 8 * sp.pi * sp.sqrt(s)


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


def _determine_indices(symbol) -> list[int]:
    r"""Extract any indices if available from a `~sympy.core.symbol.Symbol`.

    >>> _determine_indices(sp.Symbol("m1"))
    [1]
    >>> _determine_indices(sp.Symbol("m_a2"))
    [2]
    >>> _determine_indices(sp.Symbol(R"\alpha_{i2, 5}"))
    [2, 5]
    >>> _determine_indices(sp.Symbol("m"))
    []

    `~sympy.tensor.indexed.Indexed` instances can also be handled:
    >>> m_a = sp.IndexedBase("m_a")
    >>> _determine_indices(m_a[0])
    [0]
    """
    _, _, subscripts = split_super_sub(sp.latex(symbol))
    if not subscripts:
        return []
    subscript: str = subscripts[-1]
    subscript = re.sub(r"[^0-9^\,]", "", subscript)
    subscript = f"[{subscript}]"
    try:
        indices = eval(subscript)  # pylint: disable=eval-used
    except SyntaxError:
        return []
    return list(indices)
