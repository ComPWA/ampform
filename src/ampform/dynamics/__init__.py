"""Lineshape functions that describe the dynamics of an interaction.

.. seealso:: :doc:`/usage/dynamics` and :doc:`/usage/dynamics/analytic-continuation`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from warnings import warn

import sympy as sp

# pyright: reportUnusedImport=false
from ampform.dynamics.form_factor import (
    BlattWeisskopfSquared,  # noqa: F401
    FormFactor,
)
from ampform.dynamics.phasespace import (
    BreakupMomentumSquared,  # noqa: F401
    EqualMassPhaseSpaceFactor,  # noqa: F401
    PhaseSpaceFactor,
    PhaseSpaceFactorAbs,  # noqa: F401
    PhaseSpaceFactorComplex,  # noqa: F401
    PhaseSpaceFactorProtocol,
    PhaseSpaceFactorSWave,  # noqa: F401
    _indices_to_subscript,
)
from ampform.sympy import argument, determine_indices, unevaluated

if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter


@unevaluated
class EnergyDependentWidth(sp.Expr):
    # cspell:ignore asner
    r"""Mass-dependent width, coupled to the pole position of the resonance.

    See Equation (50.28) in :pdg-review:`2021; Resonances; p.9` and
    :cite:`ParticleDataGroup:2020ssz`, equation (6). Default value for
    :code:`phsp_factor` is `.PhaseSpaceFactor`.

    Note that the `.FormFactor` of AmpForm is normalized in the sense that equal powers
    of :math:`z` appear in the nominator and the denominator, while the definition in
    the PDG (as well as some other sources), always have :math:`1` in the nominator of
    the Blatt-Weisskopf. In that case, one needs an additional factor
    :math:`\left(q/q_0\right)^{2L}` in the definition for :math:`\Gamma(m)`.
    """

    s: Any
    mass0: Any
    gamma0: Any
    m_a: Any
    m_b: Any
    angular_momentum: Any
    meson_radius: Any
    phsp_factor: PhaseSpaceFactorProtocol = argument(  # type:ignore[assignment]
        default=PhaseSpaceFactor, sympify=False
    )
    name: str | None = argument(default=None, sympify=False)

    def evaluate(self) -> sp.Expr:
        s, m0, width0, m1, m2, angular_momentum, meson_radius = self.args
        ff = FormFactor(s, m1, m2, angular_momentum, meson_radius)
        ff0 = FormFactor(m0**2, m1, m2, angular_momentum, meson_radius)  # type: ignore[operator]
        rho = self.phsp_factor(s, m1, m2)
        rho0 = self.phsp_factor(m0**2, m1, m2)  # type: ignore[operator]
        return width0 * (ff / ff0) ** 2 * (rho / rho0)

    def _latex_repr_(self, printer: LatexPrinter, *args) -> str:
        s = printer._print(self.args[0])
        width0 = self.args[2]
        subscript = _indices_to_subscript(determine_indices(width0))
        name = Rf"\Gamma{subscript}" if self.name is None else self.name
        return Rf"{name}\left({s}\right)"


def relativistic_breit_wigner(s, mass0, gamma0) -> sp.Expr:
    """Relativistic Breit-Wigner lineshape.

    See :ref:`usage/dynamics:_Without_ form factor` and
    :cite:`ParticleDataGroup:2020ssz`.
    """
    return gamma0 * mass0 / (mass0**2 - s - gamma0 * mass0 * sp.I)


def relativistic_breit_wigner_with_ff(  # noqa: PLR0917
    s,
    mass0,
    gamma0,
    m_a,
    m_b,
    angular_momentum,
    meson_radius,
    phsp_factor: PhaseSpaceFactorProtocol = PhaseSpaceFactor,  # type:ignore[assignment]
) -> sp.Expr:
    """Relativistic Breit-Wigner with `.FormFactor`.

    See :ref:`usage/dynamics:_With_ form factor` and :pdg-review:`2021; Resonances;
    p.9`.
    """
    form_factor = FormFactor(s, m_a, m_b, angular_momentum, meson_radius)
    energy_dependent_width = EnergyDependentWidth(
        s, mass0, gamma0, m_a, m_b, angular_momentum, meson_radius, phsp_factor
    )
    return (mass0 * gamma0 * form_factor) / (
        mass0**2 - s - energy_dependent_width * mass0 * sp.I
    )


def formulate_form_factor(s, m_a, m_b, angular_momentum, meson_radius) -> sp.Expr:
    """Formulate a Blatt-Weisskopf form factor.

    .. deprecated:: 0.16
    """
    warn(
        message="Use the FormFactor expression class instead.",
        category=DeprecationWarning,
        stacklevel=1,
    )
    return FormFactor(s, m_a, m_b, angular_momentum, meson_radius)
