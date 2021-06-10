"""Build `~ampform.dynamics` with correct variable names and values."""

from typing import Dict, Optional, Tuple

import attr
import sympy as sp
from qrules.particle import Particle

from . import (
    BlattWeisskopfSquared,
    PhaseSpaceFactor,
    breakup_momentum_squared,
    phase_space_factor,
    phase_space_factor_ac,
    relativistic_breit_wigner,
    relativistic_breit_wigner_with_ff,
)

# pyright: reportUnusedImport=false
from .decorator import verify_signature  # noqa: F401

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore


@attr.s(frozen=True)
class TwoBodyKinematicVariableSet:
    """Data container for the essential variables of a two-body decay.

    This data container is inserted into a `.ResonanceDynamicsBuilder`, so that
    it can build some lineshape expression from the `.dynamics` module. It also
    allows to insert :doc:`custom dynamics </usage/dynamics/custom>` into the
    amplitude model.
    """

    in_edge_inv_mass: sp.Symbol = attr.ib()
    out_edge_inv_mass1: sp.Symbol = attr.ib()
    out_edge_inv_mass2: sp.Symbol = attr.ib()
    helicity_theta: sp.Symbol = attr.ib()
    helicity_phi: sp.Symbol = attr.ib()
    angular_momentum: Optional[int] = attr.ib(default=None)


BuilderReturnType = Tuple[sp.Expr, Dict[sp.Symbol, float]]
"""Type that a `.ResonanceDynamicsBuilder` should return."""


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
    ) -> BuilderReturnType:
        ...


def create_non_dynamic(
    resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
) -> BuilderReturnType:
    # pylint: disable=unused-argument
    return (1, {})


def create_non_dynamic_with_ff(
    resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
) -> BuilderReturnType:
    """Generate (only) a Blatt-Weisskopf form factor for a two-body decay.

    Returns the `~sympy.functions.elementary.miscellaneous.sqrt` of a
    `.BlattWeisskopfSquared`.
    """
    angular_momentum = variable_pool.angular_momentum
    if angular_momentum is None:
        raise ValueError(
            "Angular momentum is not defined but is required in the form factor!"
        )
    q_squared = breakup_momentum_squared(
        s=variable_pool.in_edge_inv_mass ** 2,
        m_a=variable_pool.out_edge_inv_mass1,
        m_b=variable_pool.out_edge_inv_mass2,
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


def create_relativistic_breit_wigner(
    resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
) -> BuilderReturnType:
    """Create a `.relativistic_breit_wigner` for a two-body decay."""
    inv_mass = variable_pool.in_edge_inv_mass
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


def _make_relativistic_breit_wigner_with_ff(
    phsp_factor: PhaseSpaceFactor,
    docstring: str,
) -> ResonanceDynamicsBuilder:
    """Factory for a `.ResonanceDynamicsBuilder` that uses `.relativistic_breit_wigner_with_ff`."""

    def dynamics_builder(
        resonance: Particle, variable_pool: TwoBodyKinematicVariableSet
    ) -> BuilderReturnType:
        if variable_pool.angular_momentum is None:
            raise ValueError(
                "Angular momentum is not defined but is required in the form factor!"
            )

        inv_mass = variable_pool.in_edge_inv_mass
        res_mass = sp.Symbol(f"m_{resonance.name}")
        res_width = sp.Symbol(f"Gamma_{resonance.name}")
        product1_inv_mass = variable_pool.out_edge_inv_mass1
        product2_inv_mass = variable_pool.out_edge_inv_mass2
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
            phsp_factor=phsp_factor,
        )
        parameter_defaults = {
            res_mass: resonance.mass,
            res_width: resonance.width,
            meson_radius: 1,
        }
        return expression, parameter_defaults

    dynamics_builder.__doc__ = docstring
    return dynamics_builder


create_relativistic_breit_wigner_with_ff = _make_relativistic_breit_wigner_with_ff(
    phsp_factor=phase_space_factor,
    docstring="Create a `.relativistic_breit_wigner_with_ff` for a two-body decay.",
)
create_analytic_breit_wigner = _make_relativistic_breit_wigner_with_ff(
    phsp_factor=phase_space_factor_ac,
    docstring="""
Create a `.relativistic_breit_wigner_with_ff` with analytic continuation.

.. seealso:: :doc:`/usage/dynamics/analytic-continuation`.
""",
)
