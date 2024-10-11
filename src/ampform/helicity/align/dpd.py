"""Spin alignment with Dalitz-Plot Decomposition.

See :cite:`Marangotto:2019ucc`.
"""

from __future__ import annotations

from functools import cache, singledispatch
from typing import TYPE_CHECKING, Literal, TypeVar

import attrs
import sympy as sp
from attrs import define, field
from attrs.validators import in_
from qrules.topology import Topology
from qrules.transition import ReactionInfo, StateTransition
from sympy.physics.quantum.spin import Rotation as Wigner

from ampform._qrules import get_qrules_version
from ampform.helicity.align import SpinAlignment
from ampform.helicity.decay import (
    get_outer_state_ids,
    get_spectator_id,
    group_by_topology,
)
from ampform.helicity.naming import create_amplitude_base, create_spin_projection_symbol
from ampform.kinematics.angles import formulate_zeta_angle
from ampform.sympy import PoolSum

if TYPE_CHECKING:
    from sympy.physics.quantum.spin import WignerD

if get_qrules_version() < (0, 10):
    from qrules.transition import (  # type: ignore[attr-defined]
        StateTransitionCollection,
    )


@define
class DalitzPlotDecomposition(SpinAlignment):
    """Alignment amplitudes with the "axis-angle" method.

    See :cite:`Marangotto:2019ucc` and `Wigner rotations
    <https://en.wikipedia.org/wiki/Wigner_rotation>`_.
    """

    reference_subsystem: Literal[1, 2, 3] = field(validator=in_({1, 2, 3}))

    def formulate_amplitude(self, reaction: ReactionInfo) -> sp.Expr:
        return _formulate_aligned_amplitude(reaction, self.reference_subsystem)[0]

    def define_symbols(self, reaction: ReactionInfo) -> dict[sp.Symbol, sp.Expr]:
        return _formulate_aligned_amplitude(reaction, self.reference_subsystem)[1]


@cache
def _formulate_aligned_amplitude(  # noqa: PLR0914
    reaction: ReactionInfo, reference_subsystem: Literal[1, 2, 3]
) -> tuple[sp.Expr, dict[sp.Symbol, sp.Expr]]:
    wigner_generator = _DPDAlignmentWignerGenerator(reference_subsystem)
    outer_state_ids = get_outer_state_ids(reaction)
    λ0, λ1, λ2, λ3 = (  # noqa: PLC2401
        create_spin_projection_symbol(i) for i in outer_state_ids
    )
    _λ0, _λ1, _λ2, _λ3 = sp.symbols(R"\lambda_(:4)^", rational=True)  # noqa: PLC2401
    some_transition = reaction.transitions[0]
    j0, j1, j2, j3 = (
        sp.Rational(some_transition.states[i].particle.spin) for i in outer_state_ids
    )
    topology_groups = group_by_topology(reaction.transitions)
    aligned_amplitudes: list[sp.Mul] = []
    for topology in topology_groups:
        spectator_id = get_spectator_id(topology)
        base = create_amplitude_base(topology)
        aligned_amplitudes += [
            base[_λ0, _λ1, _λ2, _λ3]
            * wigner_generator(j0, λ0, _λ0, 0, spectator_id)
            * wigner_generator(j1, _λ1, λ1, 1, spectator_id)
            * wigner_generator(j2, _λ2, λ2, 2, spectator_id)
            * wigner_generator(j3, _λ3, λ3, 3, spectator_id)
        ]
    outer_helicities = _collect_outer_state_helicities(reaction)
    amp_expr = PoolSum(
        sp.Add(*aligned_amplitudes),
        (_λ0, outer_helicities[0]),
        (_λ1, outer_helicities[1]),
        (_λ2, outer_helicities[2]),
        (_λ3, outer_helicities[3]),
    )
    return amp_expr, wigner_generator.angle_definitions


class _DPDAlignmentWignerGenerator:
    def __init__(self, reference_subsystem: Literal[1, 2, 3] = 1) -> None:
        self.angle_definitions: dict[sp.Symbol, sp.Expr] = {}
        self.reference_subsystem = reference_subsystem

    def __call__(
        self,
        j: sp.Rational | sp.Symbol,
        m: sp.Rational | sp.Symbol,
        m_prime: sp.Rational | sp.Symbol,
        rotated_state: Literal[0, 1, 2, 3],
        aligned_subsystem: Literal[1, 2, 3],
    ) -> sp.Rational | WignerD:
        if j == 0:
            return sp.Rational(1)
        zeta, zeta_expr = formulate_zeta_angle(
            rotated_state, aligned_subsystem, self.reference_subsystem
        )
        self.angle_definitions[zeta] = zeta_expr
        return Wigner.d(j, m, m_prime, zeta)


if get_qrules_version() < (0, 10):
    T = TypeVar("T", ReactionInfo, StateTransition, StateTransitionCollection, Topology)
    """Allowed types for :func:`relabel_edge_ids`."""
else:
    T = TypeVar(  # type: ignore[misc]  # pyright: ignore[reportConstantRedefinition]
        "T", ReactionInfo, StateTransition, Topology
    )
    """Allowed types for :func:`relabel_edge_ids`."""


@singledispatch
def relabel_edge_ids(obj: T) -> T:  # type: ignore[reportInvalidTypeForm]
    msg = f"Cannot relabel edge IDs of a {type(obj).__name__}"
    raise NotImplementedError(msg)


@relabel_edge_ids.register(ReactionInfo)
def _(obj: ReactionInfo) -> ReactionInfo:  # type: ignore[misc]
    if get_qrules_version() < (0, 10):
        return ReactionInfo(  # type: ignore[call-arg]
            transition_groups=[relabel_edge_ids(g) for g in obj.transition_groups],  # type: ignore[attr-defined]
            formalism=obj.formalism,
        )
    return ReactionInfo(
        # no attrs.evolve() in order to call __attrs_post_init__()
        transitions=[relabel_edge_ids(g) for g in obj.transitions],
        formalism=obj.formalism,
    )


if get_qrules_version() < (0, 10):

    def __relabel_stc(obj: StateTransitionCollection) -> StateTransitionCollection:  # type: ignore[misc]
        return StateTransitionCollection([
            relabel_edge_ids(transition) for transition in obj.transitions
        ])

    relabel_edge_ids.register(StateTransitionCollection)(__relabel_stc)


def __relabel_st(obj: StateTransition) -> StateTransition:  # type: ignore[misc]
    mapping = __get_default_relabel_mapping()
    return attrs.evolve(
        obj,
        topology=relabel_edge_ids(obj.topology),
        states={mapping[k]: v for k, v in obj.states.items()},
    )


if get_qrules_version() < (0, 10):
    relabel_edge_ids.register(StateTransition)(__relabel_st)
else:
    from qrules.topology import FrozenTransition

    relabel_edge_ids.register(FrozenTransition)(__relabel_st)


@relabel_edge_ids.register(Topology)
def _(obj: Topology) -> Topology:  # type: ignore[misc]
    mapping = __get_default_relabel_mapping()
    return obj.relabel_edges(mapping)


def __get_default_relabel_mapping() -> dict[int, int]:
    return {i - 1: i for i in range(5)}


def _collect_outer_state_helicities(
    reaction: ReactionInfo,
) -> dict[int, list[sp.Rational]]:
    outer_state_ids = get_outer_state_ids(reaction)
    return {
        i: sorted({
            sp.Rational(transition.states[i].spin_projection)
            for transition in reaction.transitions
        })
        for i in outer_state_ids
    }
