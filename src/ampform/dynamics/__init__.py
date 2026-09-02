"""Lineshape functions that describe the dynamics of an interaction.

.. seealso:: :doc:`/dynamics` and :doc:`/analyticity/phasespace-factors`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from warnings import warn

import sympy as sp
from attrs import asdict, frozen

from ampform.dynamics import phasespace as phasespace
from ampform.dynamics.form_factor import (
    BlattWeisskopfSquared,  # ruff: ignore[unused-import]
    FormFactor,
)
from ampform.dynamics.phasespace import (
    EqualMassPhaseSpaceFactor,  # ruff: ignore[unused-import]
    PhaseSpaceFactor,
    PhaseSpaceFactorAbs,  # ruff: ignore[unused-import]
    PhaseSpaceFactorComplex,  # ruff: ignore[unused-import]
    PhaseSpaceFactorProtocol,
    PhaseSpaceFactorSWave,  # ruff: ignore[unused-import]
)
from ampform.kinematics.phasespace import (
    BreakupMomentumSquared,  # ruff: ignore[unused-import]
    _get_subscript,
)
from ampform.sympy import argument, unevaluated

if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter


@unevaluated
class SimpleBreitWigner(sp.Expr):
    r"""Simple Breit–Wigner with :math:`m_0 \Gamma_0` in the numerator."""

    s: Any
    mass: Any
    width: Any
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{BW}}\left({s}; {mass}, {width}\right)"

    def evaluate(self):
        s, m0, w0 = self.args
        return m0 * w0 / (m0**2 - s - m0 * w0 * sp.I)


@unevaluated
class BreitWigner(sp.Expr):
    r"""Relativistic Breit–Wigner with :math:`m_0 \Gamma_0` in the numerator.

    Uses an `EnergyDependentWidth` in the denominator (see Equations
    :eq:`BreitWigner` and :eq:`EnergyDependentWidth`).
    """

    s: Any
    mass: Any
    width: Any
    m1: Any = 0
    m2: Any = 0
    angular_momentum: Any = 0
    meson_radius: Any = 1
    phsp_factor: PhaseSpaceFactorProtocol = argument(
        default=PhaseSpaceFactor, sympify=False
    )  # ty: ignore[invalid-assignment]

    def evaluate(self):
        width = self.energy_dependent_width()
        return (
            self.mass * self.width / (self.mass**2 - self.s - self.mass * width * sp.I)
        )

    def energy_dependent_width(self) -> EnergyDependentWidth | sp.Basic:
        s, m0, w0, m1, m2, ang_mom, d = self.args
        if ang_mom == 0 and m1 == 0 and m2 == 0:
            return w0
        return EnergyDependentWidth(s, m0, w0, m1, m2, ang_mom, d, self.phsp_factor)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        function_symbol = R"\mathcal{R}^\mathrm{BW}"
        mass = printer._print(self.mass)
        width = printer._print(self.width)
        arg = Rf"\left({s}; {mass}, {width}\right)"
        angular_momentum = printer._print(self.angular_momentum)
        if isinstance(self.angular_momentum, sp.Integer):
            return Rf"{function_symbol}_{{L={angular_momentum}}}{arg}"
        return Rf"{function_symbol}_{{{angular_momentum}}}{arg}"


@unevaluated
class EnergyDependentWidth(sp.Expr):
    r"""Mass-dependent width, coupled to the pole position of the resonance.

    See Equation (50.28) in :pdg-review:`2021; Resonances; p.9` and
    :cite:`ParticleDataGroup:2012pjm`, equation (6). Default value for
    :code:`phsp_factor` is `.PhaseSpaceFactor`.

    Note that the `.FormFactor` of AmpForm is normalized in the sense that equal powers
    of :math:`z` appear in the nominator and the denominator, while the definition in
    the PDG (as well as some other sources), always have :math:`1` in the nominator of
    the Blatt–Weisskopf. In that case, one needs an additional factor
    :math:`\left(q/q_0\right)^{2L}` in the definition for :math:`\Gamma(m)`.
    """

    s: Any
    mass0: Any
    gamma0: Any
    m_a: Any
    m_b: Any
    angular_momentum: Any
    meson_radius: Any
    phsp_factor: PhaseSpaceFactorProtocol = argument(
        default=PhaseSpaceFactor, sympify=False
    )  # ty: ignore[invalid-assignment]
    name: str | None = argument(default=None, kw_only=True, sympify=False)

    def evaluate(self) -> sp.Expr:
        m0: sp.Expr
        s, m0, width0, m1, m2, angular_momentum, meson_radius = self.args  # ty: ignore[invalid-assignment]
        ff = FormFactor(s, m1, m2, angular_momentum, meson_radius)
        ff0 = FormFactor(m0**2, m1, m2, angular_momentum, meson_radius)
        rho = self.phsp_factor(s, m1, m2)
        rho0 = self.phsp_factor(m0**2, m1, m2)
        return width0 * (ff / ff0) ** 2 * (rho / rho0)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        name = self.name or Rf"\Gamma{_get_subscript(self.gamma0)}"
        return Rf"{name}\left({s}\right)"


@unevaluated
class MultichannelBreitWigner(sp.Expr):
    """`BreitWigner` for multiple channels."""

    s: Any
    mass: Any
    channels: list[ChannelArguments] = argument(sympify=False)

    def evaluate(self):
        s = self.s
        m0 = self.mass
        width = sum(channel.formulate_width(s, m0) for channel in self.channels)
        return SimpleBreitWigner(s, m0, width)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        latex = R"\mathcal{R}^\mathrm{BW}_\mathrm{multi}\left("
        latex += printer._print(self.s) + "; "
        latex += ", ".join(printer._print(channel.width) for channel in self.channels)
        latex += R"\right)"
        return latex


@frozen
class ChannelArguments:
    """Arguments for a channel in a `MultichannelBreitWigner`."""

    width: Any
    m1: Any = 0
    m2: Any = 0
    angular_momentum: Any = 0
    meson_radius: Any = 1
    phsp_factor: PhaseSpaceFactorProtocol = PhaseSpaceFactor  # ty: ignore[invalid-assignment]

    def __attrs_post_init__(self) -> None:
        for name, value in asdict(self).items():
            object.__setattr__(self, name, sp.sympify(value))

    def formulate_width(self, s: Any, m0: Any) -> EnergyDependentWidth:
        return EnergyDependentWidth(
            s,
            m0,
            self.width,
            self.m1,
            self.m2,
            self.angular_momentum,
            self.meson_radius,
            self.phsp_factor,
        )


def relativistic_breit_wigner(s, mass0, gamma0) -> sp.Expr:
    """Relativistic Breit–Wigner lineshape.

    See :ref:`dynamics:_Without_ form factor` and :cite:`ParticleDataGroup:2012pjm`.

    .. deprecated:: 0.17.0
        Use `.SimpleBreitWigner` instead.
    """
    warn("Use SimpleBreitWigner instead", category=DeprecationWarning, stacklevel=2)
    return SimpleBreitWigner(s, mass0, gamma0)


def relativistic_breit_wigner_with_ff(  # ruff: ignore[too-many-positional-arguments]
    s,
    mass0,
    gamma0,
    m_a,
    m_b,
    angular_momentum,
    meson_radius,
    phsp_factor: PhaseSpaceFactorProtocol = PhaseSpaceFactor,  # ty: ignore[invalid-parameter-default]
) -> sp.Expr:
    """Relativistic Breit–Wigner with `.FormFactor`.

    See :ref:`dynamics:_With_ form factor` and :pdg-review:`2021; Resonances; p.9`.
    """
    ff = FormFactor(s, m_a, m_b, angular_momentum, meson_radius)
    bw = BreitWigner(
        s, mass0, gamma0, m_a, m_b, angular_momentum, meson_radius, phsp_factor
    )
    return ff * bw


def formulate_form_factor(s, m_a, m_b, angular_momentum, meson_radius) -> sp.Expr:
    """Formulate a Blatt–Weisskopf form factor.

    .. deprecated:: 0.16.0
        Use `.FormFactor` instead.
    """
    warn(
        message="Use the FormFactor expression class instead.",
        category=DeprecationWarning,
        stacklevel=1,
    )
    return FormFactor(s, m_a, m_b, angular_momentum, meson_radius)
