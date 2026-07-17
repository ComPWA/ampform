"""Build `~ampform.dynamics` with correct variable names and values.

This module provides two builder interfaces. A `.ResonanceDynamicsBuilder` formulates
dynamics for a single resonance in a two-body decay node, given a
`.TwoBodyKinematicVariableSet`. A `.DynamicsBuilder` formulates dynamics over an
**entire decay chain** and is used by the `.DalitzPlotDecompositionBuilder`.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from functools import wraps
from typing import Any, Protocol

import sympy as sp
from attrs import define, field, frozen
from attrs.validators import instance_of

from ampform.decay import (
    DecayNode,
    IsobarNode,
    Particle,
    ParticleLike,
    State,
    ThreeBodyDecayChain,
    to_particle,
)
from ampform.dynamics import (
    FormFactor,
    RelativisticBreitWigner,
    SimpleBreitWigner,
    relativistic_breit_wigner,
)
from ampform.dynamics.phasespace import (
    EqualMassPhaseSpaceFactor,
    PhaseSpaceFactor,
    PhaseSpaceFactorProtocol,
)


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

    Every dynamics builder returns its lineshape in this form: a
    `.ResonanceDynamicsBuilder` for a single two-body decay node and a
    `.DynamicsBuilder` for an entire decay chain. The `parameters` provide suggested
    starting values (see :attr:`.HelicityModel.parameter_defaults`) for the
    `~sympy.core.symbol.Symbol` instances that appear in the `expression`.
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
        breit_wigner_expr = RelativisticBreitWigner(
            s=inv_mass**2,
            mass0=res_mass,
            gamma0=res_width,
            m1=variable_pool.outgoing_state_mass1,
            m2=variable_pool.outgoing_state_mass2,
            angular_momentum=variable_pool.angular_momentum,
            meson_radius=meson_radius,
            phsp_factor=self.phsp_factor,
        ).doit(deep=False)
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


DynamicsBuilder = Callable[[ThreeBodyDecayChain], DefinedExpression]
"""Protocol for functions that formulate dynamics expressions for decay chains.

As opposed to a `.ResonanceDynamicsBuilder`, which formulates dynamics for a single
two-body decay node, a `DynamicsBuilder` defines dynamics over an **entire decay
chain**. It is used by :meth:`.DalitzPlotDecompositionBuilder.formulate`.
"""


@define
class BreitWignerBuilder:
    """Chain-level dynamics builder for (relativistic) Breit–Wigner functions.

    The :meth:`__call__` of this builder complies with the `.DynamicsBuilder` protocol,
    so instances of this class can be used in
    :meth:`.DynamicsConfigurator.register_builder`.
    """

    energy_dependent_width: bool = True
    decay_form_factor: bool = True
    production_form_factor: bool = True
    phsp_factor: PhaseSpaceFactorProtocol = PhaseSpaceFactor  # ty:ignore[invalid-assignment]

    def __call__(self, decay_chain: ThreeBodyDecayChain) -> DefinedExpression:
        """Formulate a (relativistic) Breit-Wigner for this resonance."""
        decay_node = decay_chain.decay_node
        s = get_mandelstam_s(decay_node)
        if self.energy_dependent_width:
            expression = _create_breit_wigner(s, decay_node, self.phsp_factor)
        else:
            expression = _create_simple_breit_wigner(s, decay_node)
        if self.decay_form_factor:
            expression *= _create_form_factor(decay_node)
        if self.production_form_factor:
            expression *= _create_form_factor(decay_chain.production_node)
        return expression


formulate_breit_wigner_with_form_factor = BreitWignerBuilder()


def _create_form_factor(isobar: IsobarNode) -> DefinedExpression:
    parameter_defaults: dict[sp.Basic, complex | float] = {}
    if isinstance(isobar.parent, State):
        inv_mass = sp.Symbol("m0", nonnegative=True)
        parameter_defaults[inv_mass] = to_particle(isobar).mass
        s = inv_mass**2
    else:
        s = get_mandelstam_s(isobar)
    outgoing_state_mass1 = create_mass_symbol(isobar.child1)
    outgoing_state_mass2 = create_mass_symbol(isobar.child2)
    meson_radius = _create_meson_radius_symbol(isobar)
    form_factor = FormFactor(
        s=s,
        m1=outgoing_state_mass1,
        m2=outgoing_state_mass2,
        angular_momentum=_get_angular_momentum(isobar),
        meson_radius=meson_radius,
    )
    parameter_defaults.update({
        meson_radius: 1,
        outgoing_state_mass1: to_particle(isobar.child1).mass,
        outgoing_state_mass2: to_particle(isobar.child2).mass,
    })
    return DefinedExpression(form_factor, parameter_defaults)


def _create_breit_wigner(
    s: sp.Symbol, isobar: DecayNode, phsp_factor: PhaseSpaceFactorProtocol
) -> DefinedExpression:
    outgoing_state_mass1 = create_mass_symbol(isobar.child1)
    outgoing_state_mass2 = create_mass_symbol(isobar.child2)
    angular_momentum = _get_angular_momentum(isobar)
    res_mass = create_mass_symbol(isobar.parent)
    res_width = sp.Symbol(Rf"\Gamma_{{{isobar.parent.latex}}}", nonnegative=True)
    meson_radius = _create_meson_radius_symbol(isobar)
    breit_wigner_expr = RelativisticBreitWigner(
        s=s,
        mass0=res_mass,
        gamma0=res_width,
        m1=outgoing_state_mass1,
        m2=outgoing_state_mass2,
        angular_momentum=angular_momentum,
        meson_radius=meson_radius,
        phsp_factor=phsp_factor,
    )
    parameter_defaults: dict[sp.Basic, complex | float] = {
        res_mass: isobar.parent.mass,
        res_width: isobar.parent.width,
        meson_radius: 1,
    }
    return DefinedExpression(breit_wigner_expr, parameter_defaults)


def _create_simple_breit_wigner(s: sp.Symbol, isobar: DecayNode) -> DefinedExpression:
    mass = create_mass_symbol(isobar.parent)
    width = sp.Symbol(Rf"\Gamma_{{{isobar.parent.latex}}}", nonnegative=True)
    meson_radius = _create_meson_radius_symbol(isobar)
    return DefinedExpression(
        expression=SimpleBreitWigner(s, mass, width),
        parameters={
            mass: isobar.parent.mass,
            width: isobar.parent.width,
            meson_radius: 1,
        },
    )


def _get_angular_momentum(isobar: IsobarNode) -> int:
    if isobar.interaction is None:
        msg = "Need LS couplings to formulate a form factor"
        raise ValueError(msg)
    return isobar.interaction.L


def _create_meson_radius_symbol(isobar: IsobarNode) -> sp.Symbol:
    if isinstance(isobar.parent, State):
        return sp.Symbol(Rf"R_{{{isobar.parent.latex}}}", nonnegative=True)
    return sp.Symbol(R"R_\mathrm{res}", nonnegative=True)


def create_mass_symbol(particle: IsobarNode | Particle | State) -> sp.Symbol:
    particle = to_particle(particle)
    if isinstance(particle, State):
        return sp.Symbol(f"m{particle.index}", nonnegative=True)
    return sp.Symbol(f"m_{{{particle.latex}}}", nonnegative=True)


def get_mandelstam_s(decay: DecayNode) -> sp.Symbol:
    subsystem_id, *_ = {1, 2, 3} - {
        s.index for s in decay.children if isinstance(s, State)
    }
    return sp.Symbol(f"sigma{subsystem_id}", nonnegative=True)
