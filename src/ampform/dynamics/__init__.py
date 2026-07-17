"""Lineshape functions that describe the dynamics of an interaction.

.. seealso:: :doc:`/dynamics` and :doc:`/analyticity/phasespace-factors`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from warnings import warn

import sympy as sp

from ampform.dynamics import phasespace as phasespace
from ampform.dynamics.form_factor import (
    BlattWeisskopfSquared,  # ruff: ignore[unused-import]
    FormFactor,
)
from ampform.dynamics.phasespace import (
    BreakupMomentum,
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
from ampform.sympy import determine_indices as determine_indices

if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter


@unevaluated
class EnergyDependentWidth(sp.Expr):
    # cspell:ignore asner
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


def relativistic_breit_wigner(s, mass0, gamma0) -> sp.Expr:
    """Relativistic Breit–Wigner lineshape.

    The lineshape is normalized such that its numerator is :code:`mass0 * gamma0`, see
    `SimpleBreitWigner` for the unnormalized expression class. See also
    :ref:`dynamics:_Without_ form factor` and :cite:`ParticleDataGroup:2012pjm`.
    """
    return mass0 * gamma0 * SimpleBreitWigner(s, mass0, gamma0).doit(deep=False)


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

    The Breit–Wigner (with energy-dependent width) is defined by
    `RelativisticBreitWigner`. See also :ref:`dynamics:_With_ form factor` and
    :pdg-review:`2021; Resonances; p.9`.
    """
    form_factor = FormFactor(s, m_a, m_b, angular_momentum, meson_radius)
    breit_wigner = RelativisticBreitWigner(
        s, mass0, gamma0, m_a, m_b, angular_momentum, meson_radius, phsp_factor
    )
    return form_factor * breit_wigner.doit(deep=False)


@unevaluated
class RelativisticBreitWigner(sp.Expr):
    s: Any
    mass0: Any
    gamma0: Any
    m1: Any
    m2: Any
    angular_momentum: Any
    meson_radius: Any
    phsp_factor: PhaseSpaceFactorProtocol = argument(
        default=PhaseSpaceFactor, sympify=False
    )  # ty:ignore[invalid-assignment]
    _latex_repr_ = (
        R"\mathcal{{R}}_{{{angular_momentum}}}\left({s}, {mass0}, {gamma0}\right)"
    )

    def evaluate(self):
        s, m0, w0, m1, m2, angular_momentum, meson_radius = self.args
        width = EnergyDependentWidth(
            s=s,
            mass0=m0,
            gamma0=w0,
            m_a=m1,
            m_b=m2,
            angular_momentum=angular_momentum,
            meson_radius=meson_radius,
            phsp_factor=self.phsp_factor,
        )
        return (m0 * w0) / (m0**2 - s - width * m0 * sp.I)


@unevaluated
class BreitWignerMinL(sp.Expr):
    s: Any
    decaying_mass: Any
    spectator_mass: Any
    resonance_mass: Any
    resonance_width: Any
    child2_mass: Any
    child1_mass: Any
    l_dec: Any
    l_prod: Any
    R_dec: Any
    R_prod: Any
    phsp_factor: PhaseSpaceFactorProtocol = argument(
        default=PhaseSpaceFactor, sympify=False
    )  # ty:ignore[invalid-assignment]
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{BW}}_{{{l_dec},{l_prod}}}\left({s}\right)"

    def evaluate(self):  # ruff: ignore[too-many-locals]
        s, m_top, m_spec, m0, Γ0, m1, m2, l_dec, l_prod, R_dec, R_prod = self.args
        ff_prod = FormFactor(m_top**2, sp.sqrt(s), m_spec, l_prod, R_prod)
        ff0_prod = FormFactor(m_top**2, m0, m_spec, l_prod, R_prod)
        ff_dec = FormFactor(s, m1, m2, l_dec, R_dec)
        ff0_dec = FormFactor(m0**2, m1, m2, l_dec, R_dec)
        width = EnergyDependentWidth(s, m0, Γ0, m1, m2, l_dec, R_dec, self.phsp_factor)
        return sp.Mul(
            ff_prod / ff0_prod,
            1 / (m0**2 - s - sp.I * m0 * width),
            ff_dec / ff0_dec,
            evaluate=False,
        )


@unevaluated
class BuggBreitWigner(sp.Expr):
    s: Any
    m0: Any
    Γ0: Any
    m1: Any
    m2: Any
    γ: Any
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{Bugg}}\left({s}\right)"

    def evaluate(self):
        s, m0, Γ0, m1, m2, γ = self.args
        # Adler zero
        s_A = m1**2 - m2**2 / 2  # ruff: ignore[non-lowercase-variable-in-function]
        g_squared = sp.Mul(
            (s - s_A) / (m0**2 - s_A),
            m0 * Γ0 * sp.exp(-γ * s),
            evaluate=False,
        )
        return 1 / (m0**2 - s - sp.I * g_squared)


@unevaluated
class FlattéSWave(sp.Expr):
    # https://github.com/ComPWA/polarimetry/blob/34f5330/julia/notebooks/model0.jl#L151-L161
    s: Any
    m0: Any
    widths: tuple[Any, Any]
    masses1: tuple[Any, Any]
    masses2: tuple[Any, Any]
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{Flatté}}\left({s}\right)"

    def evaluate(self):
        m0: sp.Expr
        s, m0, (Γ1, Γ2), (ma1, mb1), (ma2, mb2) = self.args  # ty:ignore[not-iterable, invalid-assignment]
        p = BreakupMomentum(s, ma1, mb1)
        p0 = BreakupMomentum(m0**2, ma2, mb2)
        q = BreakupMomentum(s, ma2, mb2)
        q0 = BreakupMomentum(m0**2, ma2, mb2)
        Γ1 *= (p / p0) * m0 / sp.sqrt(s)
        Γ2 *= (q / q0) * m0 / sp.sqrt(s)
        Γ = Γ1 + Γ2
        return 1 / (m0**2 - s - sp.I * m0 * Γ)


@unevaluated
class MultichannelBreitWigner(sp.Expr):
    s: Any
    mass: Any
    channels: tuple[ChannelArguments, ...]

    def evaluate(self):
        s = self.s
        m0 = self.mass
        width = sum(channel.evaluate() for channel in self.channels)
        return BreitWigner(s, m0, width)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        latex = R"\mathcal{R}^\mathrm{BW}_\mathrm{multi}\left("
        latex += printer._print(self.s) + "; "
        latex += ", ".join(printer._print(channel.width) for channel in self.channels)
        latex += R"\right)"
        return latex


@unevaluated
class ChannelArguments(sp.Expr):
    s: Any
    m0: Any
    width: Any
    m1: Any = 0
    m2: Any = 0
    angular_momentum: Any = 0
    meson_radius: Any = 1
    _latex_repr_ = R"\Gamma^\text{channel}\left({{s}}, {{m0}}, {{width}}\right)"

    def evaluate(self) -> sp.Expr:
        s, m0, Γ0, m1, m2, L, R = self.args
        ff = FormFactor(s, m1, m2, L, R) ** 2
        return Γ0 * m0 / sp.sqrt(s) * ff


@unevaluated
class BreitWigner(sp.Expr):
    s: Any
    mass: Any
    width: Any
    m1: Any = 0
    m2: Any = 0
    angular_momentum: Any = 0
    meson_radius: Any = 1
    phsp_factor: PhaseSpaceFactorProtocol = argument(
        default=PhaseSpaceFactor, sympify=False
    )  # ty:ignore[invalid-assignment]

    def evaluate(self):
        width = self.energy_dependent_width()
        expr = SimpleBreitWigner(self.s, self.mass, width)
        if self.angular_momentum == 0 and self.m1 == 0 and self.m2 == 0:
            return expr.evaluate()
        return expr

    def energy_dependent_width(self) -> sp.Expr:
        s, m0, Γ0, m1, m2, L, d = self.args
        if L == 0 and m1 == 0 and m2 == 0:
            return Γ0  # ty:ignore[invalid-return-type]
        return EnergyDependentWidth(s, m0, Γ0, m1, m2, L, d, self.phsp_factor)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.s)
        function_symbol = R"\mathcal{R}^\mathrm{BW}"
        mass = printer._print(self.mass)
        width = printer._print(self.width)
        arg = Rf"\left({s}; {mass}, {width}\right)"
        L = printer._print(self.angular_momentum)
        if isinstance(self.angular_momentum, sp.Integer):
            return Rf"{function_symbol}_{{L={L}}}{arg}"
        return Rf"{function_symbol}_{{{L}}}{arg}"


@unevaluated
class SimpleBreitWigner(sp.Expr):
    s: Any
    mass: Any
    width: Any
    _latex_repr_ = R"\mathcal{{R}}^\mathrm{{BW}}\left({s}; {mass}, {width}\right)"

    def evaluate(self):
        s, m0, Γ0 = self.args
        return 1 / (m0**2 - s - m0 * Γ0 * sp.I)


def formulate_form_factor(s, m_a, m_b, angular_momentum, meson_radius) -> sp.Expr:
    """Formulate a Blatt–Weisskopf form factor.

    .. deprecated:: 0.16
    """
    warn(
        message="Use the FormFactor expression class instead.",
        category=DeprecationWarning,
        stacklevel=1,
    )
    return FormFactor(s, m_a, m_b, angular_momentum, meson_radius)
