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
from ampform.dynamics.phasespace import BreakupMomentum as BreakupMomentum
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
