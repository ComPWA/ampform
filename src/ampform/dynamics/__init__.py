"""Lineshape functions that describe the dynamics of an interaction.

.. seealso:: :doc:`/dynamics` and :doc:`/analyticity/phasespace-factors`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

import sympy as sp

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
    r"""Simple Breit–Wigner with a configurable numerator.

    With the default ``numerator="mass-width"``, the propagator is multiplied by
    :math:`m_0 \Gamma_0`, so that
    :math:`\left|\hat{\mathcal{R}}^\mathrm{BW}(m_0^2)\right| = 1`. Set
    ``numerator="unity"`` for the dressed propagator of `PDG2026, Eq. (50.31)
    <https://pdg.lbl.gov/2026/reviews/rpp2026-rev-resonances.pdf#page=12>`__, which is
    also the convention of `MultichannelBreitWigner`.
    """

    s: Any
    mass: Any
    width: Any
    numerator: Literal["mass-width", "unity"] = argument(
        default="mass-width", kw_only=True, sympify=False
    )

    def evaluate(self):
        s, m0, w0 = self.args
        numerator = _formulate_numerator(self.numerator, m0, w0)
        return numerator * _formulate_breit_wigner(s, m0, w0)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s, mass, width = map(printer._print, self.args)
        function_symbol = _get_breit_wigner_symbol(self.numerator)
        return Rf"{function_symbol}\left({s}; {mass}, {width}\right)"


@unevaluated
class BreitWigner(sp.Expr):
    r"""Relativistic Breit–Wigner with a configurable numerator.

    Uses an `EnergyDependentWidth` in the denominator (see Equations :eq:`BreitWigner`
    and :eq:`EnergyDependentWidth`). With the default ``numerator="mass-width"``, the
    propagator is multiplied by :math:`m_0 \Gamma_0`, so that
    :math:`\left|\hat{\mathcal{R}}^\mathrm{BW}(m_0^2)\right| = 1`, because
    :math:`\Gamma(m_0^2) = \Gamma_0`. Set ``numerator="unity"`` for the dressed
    propagator of `PDG2026, Eq. (50.31)
    <https://pdg.lbl.gov/2026/reviews/rpp2026-rev-resonances.pdf#page=12>`__, which is
    also the convention of `MultichannelBreitWigner`. The numerator does not affect the
    `.FormFactor` inside the `EnergyDependentWidth`, where its normalization cancels.
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
    numerator: Literal["mass-width", "unity"] = argument(
        default="mass-width", kw_only=True, sympify=False
    )

    def evaluate(self):
        width = self.energy_dependent_width()
        numerator = _formulate_numerator(self.numerator, self.mass, self.width)
        return numerator * _formulate_breit_wigner(self.s, self.mass, width)

    def energy_dependent_width(self) -> EnergyDependentWidth | sp.Basic:
        s, m0, w0, m1, m2, ang_mom, d = self.args
        if ang_mom == 0 and m1 == 0 and m2 == 0:
            return w0
        return EnergyDependentWidth(s, m0, w0, m1, m2, ang_mom, d, self.phsp_factor)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        function_symbol = _get_breit_wigner_symbol(self.numerator)
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

    See `PDG2021, Eq. (50.28)
    <https://pdg.lbl.gov/2021/reviews/rpp2021-rev-resonances.pdf#page=9>`__ and
    :cite:`ParticleDataGroup:2012pjm`, equation (6). Default value for
    :code:`phsp_factor` is `.PhaseSpaceFactor`.

    .. warning:: Equation (50.28) no longer appears in
        `PDG2026, §Resonances, p.12 <https://pdg.lbl.gov/2026/reviews/rpp2026-rev-resonances.pdf#page=12>`__.
        The width is now defined in terms of bare couplings, :math:`\Gamma_b(s) =
        g_b^2 \rho_b(s) n_b^2(s) / m_\mathrm{BW}` (Equation (50.32)), and Equation (50.35)
        trades :math:`g_b` for the partial width :math:`\Gamma_{\mathrm{BW},b}`.
        Combining the two gives the old Equation (50.28), but the PDG stresses that this
        substitution is only valid for narrow resonances with all channel thresholds
        below :math:`m_\mathrm{BW}`.

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
    r"""Breit–Wigner with a running width summed over several decay channels.

    Each channel is a `ChannelArguments` term :math:`\Gamma_i^\text{ch}(s)`, giving

    .. math::

        \frac{1}{m_0^2 - s - i \sum_i g_i^2 \rho_i(s) F_{L_i}^2(s)},

    where :math:`g_i^2` is the coupling squared, :math:`\rho_i` is a
    `.PhaseSpaceFactor`, and :math:`F_{L_i}` is a `.FormFactor`. Unlike an
    `EnergyDependentWidth`, a channel term is not normalized at the pole position. See
    `PDG2026, Eqs. (50.31) and (50.32)
    <https://pdg.lbl.gov/2026/reviews/rpp2026-rev-resonances.pdf#page=12>`__.
    """

    s: Any
    mass: Any
    channels: tuple[ChannelArguments, ...]

    def evaluate(self):
        s = self.s
        m0 = self.mass
        width = sp.Add(*self.channels)
        return _formulate_breit_wigner(s, m0, width)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        latex = R"\mathcal{R}^\mathrm{BW}_\mathrm{multi}\left("
        latex += printer._print(self.s) + "; "
        latex += ", ".join(
            printer._print(channel.coupling_squared) for channel in self.channels
        )
        latex += R"\right)"
        return latex


@unevaluated
class ChannelArguments(sp.Expr):
    r"""One channel term :math:`\Gamma_i^\text{ch}(s)`.

    .. math::

        \Gamma_i^\text{ch}(s) = \frac{g_i^2}{m_0} \rho_i(s) F_{L_i}^2(s)

    See `PDG2026, Eq. (50.32)
    <https://pdg.lbl.gov/2026/reviews/rpp2026-rev-resonances.pdf#page=12>`__.
    """

    s: Any
    mass: Any
    coupling_squared: Any
    m1: Any = 0
    m2: Any = 0
    angular_momentum: Any = 0
    meson_radius: Any = 1
    _latex_repr_ = R"\Gamma^\text{{ch}}\left({s}; {mass}, {coupling_squared}\right)"

    def evaluate(self) -> sp.Expr:
        s, m0, coupling_squared, m1, m2, angular_momentum, meson_radius = self.args
        rho = PhaseSpaceFactor(s, m1, m2)
        ff = FormFactor(s, m1, m2, angular_momentum, meson_radius)
        return coupling_squared * rho * ff**2 / m0


def _formulate_breit_wigner(s: Any, mass: Any, width: Any) -> sp.Expr:
    return 1 / (mass**2 - s - sp.I * mass * width)


def _formulate_numerator(numerator: str, mass: Any, width: Any) -> Any:
    _check_numerator(numerator)
    if numerator == "mass-width":
        return mass * width
    return sp.S.One


def _get_breit_wigner_symbol(numerator: str) -> str:
    _check_numerator(numerator)
    if numerator == "mass-width":
        return R"\hat{\mathcal{R}}^\mathrm{BW}"
    return R"\mathcal{R}^\mathrm{BW}"


def _check_numerator(numerator: str) -> None:
    allowed = ("mass-width", "unity")
    if numerator not in allowed:
        msg = f"Unknown numerator {numerator!r}, expected one of {', '.join(map(repr, allowed))}"
        raise ValueError(msg)


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

    See :ref:`dynamics:_With_ form factor` and `PDG2026, §Resonances, p.12
    <https://pdg.lbl.gov/2026/reviews/rpp2026-rev-resonances.pdf#page=12>`__.
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
