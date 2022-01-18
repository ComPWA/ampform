"""Build `~ampform.dynamics` with correct variable names and values."""

import sys
from typing import Dict, Optional, Tuple

import attr
import sympy as sp
from attr.validators import instance_of
from qrules.particle import Particle

from . import (
    BlattWeisskopfSquared,
    BreakupMomentumSquared,
    PhaseSpaceFactor,
    PhaseSpaceFactorAnalytic,
    PhaseSpaceFactorProtocol,
    relativistic_breit_wigner,
    relativistic_breit_wigner_with_ff,
)

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


@attr.frozen
class TwoBodyKinematicVariableSet:
    """Data container for the essential variables of a two-body decay.

    This data container is inserted into a `.ResonanceDynamicsBuilder`, so that
    it can build some lineshape expression from the `.dynamics` module. It also
    allows to insert :doc:`custom dynamics </usage/dynamics/custom>` into the
    amplitude model.
    """

    incoming_state_mass: sp.Symbol = attr.ib(validator=instance_of(sp.Symbol))
    outgoing_state_mass1: sp.Symbol = attr.ib(validator=instance_of(sp.Symbol))
    outgoing_state_mass2: sp.Symbol = attr.ib(validator=instance_of(sp.Symbol))
    helicity_theta: sp.Symbol = attr.ib(validator=instance_of(sp.Symbol))
    helicity_phi: sp.Symbol = attr.ib(validator=instance_of(sp.Symbol))
    angular_momentum: Optional[int] = attr.ib(default=None)


BuilderReturnType = Tuple[sp.Expr, Dict[sp.Symbol, float]]
"""Type that a `.ResonanceDynamicsBuilder` should return.

The first element in this `tuple` is the `sympy.Expr <sympy.core.expr.Expr>`
that describes the dynamics for the resonance. The second element are suggested
parameter values (see :attr:`.parameter_defaults`) for the
`~sympy.core.symbol.Symbol` instances that appear in the `sympy.Expr
<sympy.core.expr.Expr>`.
"""


class ResonanceDynamicsBuilder(Protocol):
    """Protocol that is used by `.set_dynamics`.

    Follow this `~typing.Protocol` when defining a builder function that is to
    be used by `.set_dynamics`. For an example, see the source code
    `.create_relativistic_breit_wigner`, which creates a
    `.relativistic_breit_wigner`.

    .. seealso:: :doc:`/usage/dynamics/custom`
    """

    def __call__(
        self, resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> "BuilderReturnType":
        """Formulate a dynamics `~sympy.core.expr.Expr` for this resonance."""
        ...


def create_non_dynamic(
    resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
) -> "BuilderReturnType":
    # pylint: disable=unused-argument
    return (1, {})


def create_non_dynamic_with_ff(
    resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
) -> "BuilderReturnType":
    """Generate (only) a Blatt-Weisskopf form factor for a two-body decay.

    Returns the `~sympy.functions.elementary.miscellaneous.sqrt` of a
    `.BlattWeisskopfSquared`.
    """
    angular_momentum = variable_pool.angular_momentum
    if angular_momentum is None:
        raise ValueError(
            "Angular momentum is not defined but is required in the form"
            " factor!"
        )
    q_squared = BreakupMomentumSquared(
        s=variable_pool.incoming_state_mass ** 2,
        m_a=variable_pool.outgoing_state_mass1,
        m_b=variable_pool.outgoing_state_mass2,
    )
    meson_radius = sp.Symbol(f"d_{resonance.name}")
    form_factor_squared = BlattWeisskopfSquared(
        angular_momentum,
        z=q_squared * meson_radius ** 2,
    )
    return (
        sp.sqrt(form_factor_squared),
        {meson_radius: 1},
    )


class RelativisticBreitWignerBuilder:
    """Factory for building relativistic Breit-Wigner expressions.

    The :meth:`__call__` of this builder complies with the
    `.ResonanceDynamicsBuilder`, so instances of this class can be used in
    :meth:`.set_dynamics`.

    Args:
        form_factor: Formulate a relativistic Breit-Wigner function with form
            factor, using :func:`.relativistic_breit_wigner_with_ff`. If set to
            `False`, :meth:`__call__` builds a
            :func:`.relativistic_breit_wigner` (_without_ form factor).
        phsp_factor: A class that complies with the
            `.PhaseSpaceFactorProtocol`. Defaults to `.PhaseSpaceFactor`.
    """

    def __init__(
        self,
        form_factor: bool = False,
        phsp_factor: Optional[PhaseSpaceFactorProtocol] = None,
    ) -> None:
        if phsp_factor is None:
            phsp_factor = PhaseSpaceFactor
        self.__phsp_factor = phsp_factor
        self.__with_form_factor = form_factor

    def __call__(
        self, resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> "BuilderReturnType":
        """Formulate a relativistic Breit-Wigner for this resonance."""
        if self.__with_form_factor:
            return self.__formulate_with_form_factor(resonance, variable_pool)
        return self.__formulate(resonance, variable_pool)

    @staticmethod
    def __formulate(
        resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> "BuilderReturnType":
        inv_mass = variable_pool.incoming_state_mass
        res_mass = sp.Symbol(f"m_{resonance.name}")
        res_width = sp.Symbol(f"Gamma_{resonance.name}")
        expression = relativistic_breit_wigner(
            s=inv_mass ** 2,
            mass0=res_mass,
            gamma0=res_width,
        )
        parameter_defaults = {
            res_mass: resonance.mass,
            res_width: resonance.width,
        }
        return expression, parameter_defaults

    def __formulate_with_form_factor(
        self, resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> "BuilderReturnType":
        if variable_pool.angular_momentum is None:
            raise ValueError(
                "Angular momentum is not defined but is required in the"
                " form factor!"
            )

        inv_mass = variable_pool.incoming_state_mass
        res_mass = sp.Symbol(f"m_{resonance.name}")
        res_width = sp.Symbol(f"Gamma_{resonance.name}")
        product1_inv_mass = variable_pool.outgoing_state_mass1
        product2_inv_mass = variable_pool.outgoing_state_mass2
        angular_momentum = variable_pool.angular_momentum
        meson_radius = sp.Symbol(f"d_{resonance.name}")

        expression = relativistic_breit_wigner_with_ff(
            s=inv_mass ** 2,
            mass0=res_mass,
            gamma0=res_width,
            m_a=product1_inv_mass,
            m_b=product2_inv_mass,
            angular_momentum=angular_momentum,
            meson_radius=meson_radius,
            phsp_factor=self.__phsp_factor,
        )
        parameter_defaults = {
            res_mass: resonance.mass,
            res_width: resonance.width,
            meson_radius: 1,
        }
        return expression, parameter_defaults


create_relativistic_breit_wigner = RelativisticBreitWignerBuilder(
    form_factor=False
).__call__
"""
Create a `.relativistic_breit_wigner` for a two-body decay.

This is a convenience function for a `RelativisticBreitWignerBuilder` _without_
form factor.
"""

create_relativistic_breit_wigner_with_ff = RelativisticBreitWignerBuilder(
    form_factor=True,
    phsp_factor=PhaseSpaceFactor,
).__call__
"""
Create a `.relativistic_breit_wigner_with_ff` for a two-body decay.

This is a convenience function for a `RelativisticBreitWignerBuilder` _with_
form factor and a 'normal' `.PhaseSpaceFactor`.
"""

create_analytic_breit_wigner = RelativisticBreitWignerBuilder(
    form_factor=True,
    phsp_factor=PhaseSpaceFactorAnalytic,
).__call__
"""
Create a `.relativistic_breit_wigner_with_ff` with analytic continuation.

This is a convenience function for a `RelativisticBreitWignerBuilder` _with_
form factor and a 'analytic' phase space factor (see
`.PhaseSpaceFactorAnalytic`).

.. seealso:: :doc:`/usage/dynamics/analytic-continuation`.
"""
