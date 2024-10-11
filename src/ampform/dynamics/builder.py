"""Build `~ampform.dynamics` with correct variable names and values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import sympy as sp
from attrs import field, frozen
from attrs.validators import instance_of

from ampform.dynamics import EnergyDependentWidth, FormFactor, relativistic_breit_wigner
from ampform.dynamics.phasespace import (
    EqualMassPhaseSpaceFactor,
    PhaseSpaceFactor,
    PhaseSpaceFactorProtocol,
)

if TYPE_CHECKING:
    from qrules.particle import Particle


@frozen
class TwoBodyKinematicVariableSet:
    """Data container for the essential variables of a two-body decay.

    This data container is inserted into a `.ResonanceDynamicsBuilder`, so that it can
    build some lineshape expression from the :mod:`.dynamics` module. It also allows to
    insert :doc:`custom dynamics </usage/dynamics/custom>` into the amplitude model.
    """

    incoming_state_mass: sp.Symbol = field(validator=instance_of(sp.Symbol))
    outgoing_state_mass1: sp.Symbol = field(validator=instance_of(sp.Symbol))
    outgoing_state_mass2: sp.Symbol = field(validator=instance_of(sp.Symbol))
    helicity_theta: sp.Symbol = field(validator=instance_of(sp.Symbol))
    helicity_phi: sp.Symbol = field(validator=instance_of(sp.Symbol))
    angular_momentum: int | None = field(default=None)


BuilderReturnType = tuple[sp.Expr, dict[sp.Symbol, float]]
"""Type that a `.ResonanceDynamicsBuilder` should return.

The first element in this `tuple` is the `sympy.Expr <sympy.core.expr.Expr>` that
describes the dynamics for the resonance. The second element are suggested parameter
values (see :attr:`.parameter_defaults`) for the `~sympy.core.symbol.Symbol` instances
that appear in the `sympy.Expr <sympy.core.expr.Expr>`.
"""


class ResonanceDynamicsBuilder(Protocol):
    """Protocol that is used by `.DynamicsSelector.assign`.

    Follow this `~typing.Protocol` when defining a builder function that is to be used
    by `.DynamicsSelector.assign`. For an example, see the source code
    `.create_relativistic_breit_wigner`, which creates a `.relativistic_breit_wigner`.

    .. seealso:: :doc:`/usage/dynamics/custom`
    """

    def __call__(
        self, resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> BuilderReturnType:
        """Formulate a dynamics `~sympy.core.expr.Expr` for this resonance."""


def create_non_dynamic(
    resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
) -> BuilderReturnType:
    return (sp.S.One, {})


def create_non_dynamic_with_ff(
    resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
) -> BuilderReturnType:
    """Generate (only) a Blatt-Weisskopf form factor for a two-body decay.

    See also :class:`.FormFactor`.
    """
    if variable_pool.angular_momentum is None:
        msg = "Angular momentum is not defined but is required in the form factor!"
        raise ValueError(msg)
    res_identifier = resonance.latex or resonance.name
    meson_radius = sp.Symbol(f"d_{{{res_identifier}}}", positive=True)
    form_factor = FormFactor(
        s=variable_pool.incoming_state_mass**2,
        m1=variable_pool.outgoing_state_mass1,
        m2=variable_pool.outgoing_state_mass2,
        angular_momentum=variable_pool.angular_momentum,
        meson_radius=meson_radius,
    )
    return (
        form_factor,
        {meson_radius: 1},
    )


class RelativisticBreitWignerBuilder:
    """Factory for building relativistic Breit-Wigner expressions.

    The :meth:`__call__` of this builder complies with the `.ResonanceDynamicsBuilder`,
    so instances of this class can be used in :meth:`.DynamicsSelector.assign`.

    Args:
        form_factor: Formulate a relativistic Breit-Wigner function multiplied
            by a Blatt-Weisskopf form factor (`.FormFactor`), like in Equation (50.26)
            on :pdg-review:`2021; Resonances; p.9`.
        energy_dependent_width: Use an `.EnergyDependentWidth` in the
            denominator of the Breit-Wigner.
        phsp_factor: A class that complies with the
            `.PhaseSpaceFactorProtocol` that is used in the energy-dependent width.
            Defaults to `.PhaseSpaceFactor`.
    """

    def __init__(
        self,
        form_factor: bool = False,
        energy_dependent_width: bool = False,
        phsp_factor: PhaseSpaceFactorProtocol | None = None,
    ) -> None:
        if phsp_factor is None:
            phsp_factor = PhaseSpaceFactor  # type:ignore[arg-type,assignment]
        self.phsp_factor = phsp_factor
        self.energy_dependent_width = energy_dependent_width
        self.form_factor = form_factor

    def __call__(
        self, resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> BuilderReturnType:
        """Formulate a relativistic Breit-Wigner for this resonance."""
        if self.energy_dependent_width:
            expr, parameter_defaults = self.__energy_dependent_breit_wigner(
                resonance, variable_pool
            )
        else:
            expr, parameter_defaults = self.__simple_breit_wigner(
                resonance, variable_pool
            )
        if self.form_factor:
            form_factor, parameters = self.__create_form_factor(
                resonance, variable_pool
            )
            parameter_defaults.update(parameters)
            return form_factor * expr, parameter_defaults
        return expr, parameter_defaults

    @staticmethod
    def __simple_breit_wigner(
        resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> BuilderReturnType:
        inv_mass = variable_pool.incoming_state_mass
        identifier = resonance.latex or resonance.name
        res_mass = sp.Symbol(f"m_{{{identifier}}}", nonnegative=True)
        res_width = sp.Symbol(Rf"\Gamma_{{{identifier}}}", nonnegative=True)
        expression = relativistic_breit_wigner(
            s=inv_mass**2,
            mass0=res_mass,
            gamma0=res_width,
        )
        parameter_defaults = {
            res_mass: resonance.mass,
            res_width: resonance.width,
        }
        return expression, parameter_defaults

    def __energy_dependent_breit_wigner(
        self, resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> BuilderReturnType:
        if variable_pool.angular_momentum is None:
            msg = "Angular momentum is not defined but is required in the form factor!"
            raise ValueError(msg)

        inv_mass = variable_pool.incoming_state_mass
        m_a = variable_pool.outgoing_state_mass1
        m_b = variable_pool.outgoing_state_mass2
        angular_momentum = variable_pool.angular_momentum
        res_mass, res_width, meson_radius = self.__create_symbols(resonance)

        s = inv_mass**2
        mass_dependent_width = EnergyDependentWidth(
            s=s,
            mass0=res_mass,
            gamma0=res_width,
            m_a=m_a,
            m_b=m_b,
            angular_momentum=angular_momentum,
            meson_radius=meson_radius,
            phsp_factor=self.phsp_factor,  # type:ignore[arg-type]
        )
        breit_wigner_expr = (res_mass * res_width) / (
            res_mass**2 - s - mass_dependent_width * res_mass * sp.I
        )
        parameter_defaults = {
            res_mass: resonance.mass,
            res_width: resonance.width,
            meson_radius: 1,
        }
        return breit_wigner_expr, parameter_defaults

    def __create_form_factor(
        self, resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> BuilderReturnType:
        if variable_pool.angular_momentum is None:
            msg = "Angular momentum is not defined but is required in the form factor!"
            raise ValueError(msg)

        inv_mass = variable_pool.incoming_state_mass
        _, __, meson_radius = self.__create_symbols(resonance)
        form_factor = FormFactor(
            s=inv_mass**2,
            m1=variable_pool.outgoing_state_mass1,
            m2=variable_pool.outgoing_state_mass2,
            angular_momentum=variable_pool.angular_momentum,
            meson_radius=meson_radius,
        )
        parameter_defaults = {
            meson_radius: 1,
        }
        return form_factor, parameter_defaults  # type: ignore[return-value]

    @staticmethod
    def __create_symbols(
        resonance: Particle,
    ) -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
        identifier = resonance.latex or resonance.name
        res_mass = sp.Symbol(f"m_{{{identifier}}}", nonnegative=True)
        res_width = sp.Symbol(Rf"\Gamma_{{{identifier}}}", nonnegative=True)
        meson_radius = sp.Symbol(f"d_{{{identifier}}}", positive=True)
        return res_mass, res_width, meson_radius


create_relativistic_breit_wigner = RelativisticBreitWignerBuilder(
    form_factor=False
).__call__
"""Create a `.relativistic_breit_wigner` for a two-body decay.

This is a convenience function for a `RelativisticBreitWignerBuilder` _without_ form
factor.
"""

create_relativistic_breit_wigner_with_ff = RelativisticBreitWignerBuilder(
    energy_dependent_width=True,
    form_factor=True,
    phsp_factor=PhaseSpaceFactor,  # type:ignore[arg-type]
).__call__
"""Create a `.relativistic_breit_wigner_with_ff` for a two-body decay.

This is a convenience function for a `RelativisticBreitWignerBuilder` _with_ form factor
and a 'normal' `.PhaseSpaceFactor`.
"""

create_analytic_breit_wigner = RelativisticBreitWignerBuilder(
    energy_dependent_width=True,
    form_factor=True,
    phsp_factor=EqualMassPhaseSpaceFactor,  # type:ignore[arg-type]
).__call__
"""Create a `.relativistic_breit_wigner_with_ff` with analytic continuation.

This is a convenience function for a `RelativisticBreitWignerBuilder` _with_ form factor
and a 'analytic' phase space factor (see `.EqualMassPhaseSpaceFactor`).

.. seealso:: :doc:`/usage/dynamics/analytic-continuation`.
"""
