"""Build `~ampform.dynamics` with correct variable names and values."""

from __future__ import annotations

import operator
from functools import wraps
from typing import TYPE_CHECKING, Any, Protocol

import sympy as sp
from attrs import define, field, frozen
from attrs.validators import instance_of

from ampform.dynamics import EnergyDependentWidth, FormFactor, relativistic_breit_wigner
from ampform.dynamics.phasespace import (
    EqualMassPhaseSpaceFactor,
    PhaseSpaceFactor,
    PhaseSpaceFactorProtocol,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ampform.decay import ParticleLike


@frozen
class TwoBodyKinematicVariableSet:
    """Data container for the essential variables of a two-body decay.

    This data container is inserted into a `.ResonanceDynamicsBuilder`, so that it can
    build some lineshape expression from the :mod:`ampform.dynamics` module. It also allows to
    insert :doc:`custom dynamics </dynamics/custom>` into the amplitude model.
    """

    incoming_state_mass: sp.Symbol = field(validator=instance_of(sp.Symbol))
    outgoing_state_mass1: sp.Symbol = field(validator=instance_of(sp.Symbol))
    outgoing_state_mass2: sp.Symbol = field(validator=instance_of(sp.Symbol))
    helicity_theta: sp.Symbol = field(validator=instance_of(sp.Symbol))
    helicity_phi: sp.Symbol = field(validator=instance_of(sp.Symbol))
    angular_momentum: int | None = field(default=None)


def _binary_operation(op: Callable[[Any, Any], Any]):
    def decorator(func):
        @wraps(func)
        def wrapper(self: DefinedExpression, other):
            if isinstance(other, DefinedExpression):
                return DefinedExpression(
                    expression=op(self.expression, other.expression),
                    parameters=self.parameters | other.parameters,
                    subexpressions=self.subexpressions | other.subexpressions,
                )
            return DefinedExpression(
                expression=op(self.expression, other),
                parameters=self.parameters,
                subexpressions=self.subexpressions,
            )

        return wrapper

    return decorator


@define
class DefinedExpression:
    """Expression with suggested parameter values and subexpression definitions.

    Every `.ResonanceDynamicsBuilder` returns its lineshape in this form. The
    `parameters` provide suggested starting values (see
    :attr:`.HelicityModel.parameter_defaults`) for the `~sympy.core.symbol.Symbol`
    instances that appear in the `expression`.
    """

    expression: sp.Expr = field(converter=sp.sympify, default=sp.S.One)  # ty:ignore[invalid-assignment]
    parameters: dict[sp.Basic, complex | float] = field(factory=dict)
    subexpressions: dict[sp.Basic, sp.Expr] = field(factory=dict)

    @_binary_operation(operator.mul)
    def __mul__(self, other) -> DefinedExpression: ...  # ty:ignore[empty-body]
    @_binary_operation(operator.add)
    def __add__(self, other) -> DefinedExpression: ...  # ty:ignore[empty-body]
    @_binary_operation(operator.sub)
    def __sub__(self, other) -> DefinedExpression: ...  # ty:ignore[empty-body]
    @_binary_operation(operator.truediv)
    def __truediv__(self, other) -> DefinedExpression: ...  # ty:ignore[empty-body]
    @_binary_operation(operator.pow)
    def __pow__(self, other) -> DefinedExpression: ...  # ty:ignore[empty-body]


class ResonanceDynamicsBuilder(Protocol):
    """Protocol that is used by `.DynamicsSelector.assign`.

    Follow this `~typing.Protocol` when defining a builder function that is to be used
    by `.DynamicsSelector.assign`. For an example, see the source code
    `.create_relativistic_breit_wigner`, which creates a `.relativistic_breit_wigner`.

    .. seealso:: :doc:`/dynamics/custom`
    """

    def __call__(
        self, resonance: ParticleLike, variable_pool: TwoBodyKinematicVariableSet
    ) -> DefinedExpression:
        """Formulate a dynamics `~sympy.core.expr.Expr` for this resonance."""


def create_non_dynamic(
    resonance: ParticleLike, variable_pool: TwoBodyKinematicVariableSet
) -> DefinedExpression:
    return DefinedExpression()


def create_non_dynamic_with_ff(
    resonance: ParticleLike, variable_pool: TwoBodyKinematicVariableSet
) -> DefinedExpression:
    """Generate (only) a Blatt–Weisskopf form factor for a two-body decay.

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
    return DefinedExpression(form_factor, parameters={meson_radius: 1})


class RelativisticBreitWignerBuilder:
    """Factory for building relativistic Breit–Wigner expressions.

    The :meth:`__call__` of this builder complies with the `.ResonanceDynamicsBuilder`,
    so instances of this class can be used in :meth:`.DynamicsSelector.assign`.

    Args:
        form_factor: Formulate a relativistic Breit–Wigner function multiplied
            by a Blatt–Weisskopf form factor (`.FormFactor`), like in Equation (50.26)
            on :pdg-review:`2021; Resonances; p.9`.
        energy_dependent_width: Use an `.EnergyDependentWidth` in the
            denominator of the Breit–Wigner.
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
            phsp_factor = PhaseSpaceFactor  # ty: ignore[invalid-assignment]
        self.phsp_factor = phsp_factor
        self.energy_dependent_width = energy_dependent_width
        self.form_factor = form_factor

    def __call__(
        self, resonance: ParticleLike, variable_pool: TwoBodyKinematicVariableSet
    ) -> DefinedExpression:
        """Formulate a relativistic Breit–Wigner for this resonance."""
        if self.energy_dependent_width:
            expression = self.__energy_dependent_breit_wigner(resonance, variable_pool)
        else:
            expression = self.__simple_breit_wigner(resonance, variable_pool)
        if self.form_factor:
            expression *= self.__create_form_factor(resonance, variable_pool)
        return expression

    @staticmethod
    def __simple_breit_wigner(
        resonance: ParticleLike, variable_pool: TwoBodyKinematicVariableSet
    ) -> DefinedExpression:
        inv_mass = variable_pool.incoming_state_mass
        identifier = resonance.latex or resonance.name
        res_mass = sp.Symbol(f"m_{{{identifier}}}", nonnegative=True)
        res_width = sp.Symbol(Rf"\Gamma_{{{identifier}}}", nonnegative=True)
        return DefinedExpression(
            expression=relativistic_breit_wigner(
                s=inv_mass**2,
                mass0=res_mass,
                gamma0=res_width,
            ),
            parameters={
                res_mass: resonance.mass,
                res_width: resonance.width,
            },
        )

    def __energy_dependent_breit_wigner(
        self, resonance: ParticleLike, variable_pool: TwoBodyKinematicVariableSet
    ) -> DefinedExpression:
        if variable_pool.angular_momentum is None:
            msg = "Angular momentum is not defined but is required in the form factor!"
            raise ValueError(msg)

        inv_mass = variable_pool.incoming_state_mass
        res_mass, res_width, meson_radius = self.__create_symbols(resonance)
        s = inv_mass**2
        mass_dependent_width = EnergyDependentWidth(
            s=s,
            mass0=res_mass,
            gamma0=res_width,
            m_a=variable_pool.outgoing_state_mass1,
            m_b=variable_pool.outgoing_state_mass2,
            angular_momentum=variable_pool.angular_momentum,
            meson_radius=meson_radius,
            phsp_factor=self.phsp_factor,
        )
        breit_wigner_expr = (res_mass * res_width) / (
            res_mass**2 - s - mass_dependent_width * res_mass * sp.I
        )
        return DefinedExpression(
            expression=breit_wigner_expr,
            parameters={
                res_mass: resonance.mass,
                res_width: resonance.width,
                meson_radius: 1,
            },
        )

    def __create_form_factor(
        self, resonance: ParticleLike, variable_pool: TwoBodyKinematicVariableSet
    ) -> DefinedExpression:
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
        return DefinedExpression(form_factor, parameters={meson_radius: 1})

    @staticmethod
    def __create_symbols(
        resonance: ParticleLike,
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
    phsp_factor=PhaseSpaceFactor,
).__call__
"""Create a `.relativistic_breit_wigner_with_ff` for a two-body decay.

This is a convenience function for a `RelativisticBreitWignerBuilder` _with_ form factor
and a 'normal' `.PhaseSpaceFactor`.
"""

create_analytic_breit_wigner = RelativisticBreitWignerBuilder(
    energy_dependent_width=True,
    form_factor=True,
    phsp_factor=EqualMassPhaseSpaceFactor,
).__call__
"""Create a `.relativistic_breit_wigner_with_ff` with analytic continuation.

This is a convenience function for a `RelativisticBreitWignerBuilder` _with_ form factor
and a 'analytic' phase space factor (see `.EqualMassPhaseSpaceFactor`).

.. seealso:: :doc:`/analyticity/phasespace-factors`.
"""
