"""Spin alignment with the "axis-angle" method.

See :cite:`Marangotto:2019ucc` and `Wigner rotations
<https://en.wikipedia.org/wiki/Wigner_rotation>`_.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

import sympy as sp

from ampform.helicity import SpinAlignment
from ampform.helicity.align._spin import create_spin_range
from ampform.helicity.decay import (
    get_outer_state_ids,
    get_parent_id,
    get_sibling_state_id,
    group_by_topology,
    is_opposite_helicity_state,
)
from ampform.helicity.naming import (
    create_amplitude_base,
    create_helicity_symbol,
    create_spin_projection_symbol,
    get_helicity_angle_symbols,
    get_helicity_suffix,
)
from ampform.kinematics.angles import compute_wigner_angles
from ampform.kinematics.lorentz import create_four_momentum_symbols
from ampform.sympy import PoolSum

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from typing import Literal

    from qrules.topology import Topology
    from qrules.transition import ReactionInfo, StateTransition


class AxisAngleAlignment(SpinAlignment):
    """Alignment amplitudes with the "axis-angle" method.

    See :cite:`Marangotto:2019ucc` and `Wigner rotations
    <https://en.wikipedia.org/wiki/Wigner_rotation>`_.
    """

    @staticmethod
    def formulate_amplitude(reaction: ReactionInfo) -> sp.Expr:
        topology_groups = group_by_topology(reaction.transitions)
        outer_state_ids = get_outer_state_ids(reaction)
        amplitude = sp.S.Zero
        for topology, transitions in topology_groups.items():
            base = create_amplitude_base(topology)
            helicities = [
                get_opposite_helicity_sign(topology, i)
                * create_helicity_symbol(topology, i)
                for i in outer_state_ids
            ]
            amplitude_symbol = base[helicities]
            first_transition = transitions[0]
            alignment_sum = formulate_axis_angle_alignment(first_transition)
            amplitude += PoolSum(
                alignment_sum.expression * amplitude_symbol,
                *alignment_sum.indices,
            )
        return amplitude

    @staticmethod
    def define_symbols(reaction: ReactionInfo) -> dict[sp.Symbol, sp.Expr]:
        wigner_angles = {}
        for topology in group_by_topology(reaction.transitions):
            momenta = create_four_momentum_symbols(topology)
            wigner_rotation_ids = {
                i
                for i in topology.outgoing_edge_ids
                if get_parent_id(topology, i) != -1
            }
            for state_id in wigner_rotation_ids:
                angles = compute_wigner_angles(topology, momenta, state_id)
                wigner_angles.update(angles)
        return wigner_angles


def formulate_axis_angle_alignment(transition: StateTransition) -> PoolSum:
    """Generate all Wigner-:math:`D` combinations for a spin alignment sum.

    Generate all Wigner-:math:`D` function combinations that appear in
    :cite:`Marangotto:2019ucc`, Eq.(45), but for a generic multibody decay. Each element
    in the returned `list` is a `tuple` of Wigner-:math:`D` functions that appear in the
    summation, for a specific set of helicities were are summing over. To generate the
    full sum, make a multiply the Wigner-:math:`D` functions in each `tuple` and sum
    over all these products.
    """
    rotations = PoolSum(1)
    for rotated_state_id in transition.final_states:
        additional_rotations = formulate_rotation_chain(transition, rotated_state_id)
        rotations = __multiply_pool_sums([rotations, additional_rotations])
    return rotations


def formulate_rotation_chain(
    transition: StateTransition, rotated_state_id: int
) -> PoolSum:
    """Formulate the spin alignment sum for a specific chain.

    See Eq.(45) from :cite:`Marangotto:2019ucc`. The chain consists of a series of
    helicity rotations (see :func:`formulate_helicity_rotation_chain`) plus a Wigner
    rotation (see :func:`.formulate_wigner_rotation`) in case there is more than one
    helicity rotation.
    """
    helicity_symbol = create_spin_projection_symbol(rotated_state_id)
    helicity_rotations = formulate_helicity_rotation_chain(
        transition, rotated_state_id, helicity_symbol
    )
    if len(helicity_rotations.indices) == 1:
        return helicity_rotations
    idx_root = __GREEK_INDEX_NAMES[len(helicity_rotations.indices)]
    idx_suffix = get_helicity_suffix(transition.topology, rotated_state_id)
    wigner_rotation = formulate_wigner_rotation(
        transition,
        rotated_state_id,
        helicity_symbol=helicity_symbol,
        m_prime=sp.Symbol(f"{idx_root}{idx_suffix}", rational=True),
    )
    return __multiply_pool_sums([helicity_rotations, wigner_rotation])


def formulate_helicity_rotation_chain(
    transition: StateTransition,
    rotated_state_id: int,
    helicity_symbol: sp.Symbol,
) -> PoolSum:
    """Formulate a Wigner-:math:`D` for each helicity rotation up some state.

    The helicity rotations are performed going through the decay
    `~qrules.topology.Topology` starting from the initial state up some
    :code:`rotated_state_id`. Each rotation operates on the spin state and is therefore
    formulated as a `~sympy.physics.quantum.spin.WignerD` function (see
    :func:`.formulate_helicity_rotation`). See {doc}`/usage/helicity/spin-alignment` for
    more info.
    """
    topology = transition.topology
    rotated_state = transition.states[rotated_state_id]
    spin_magnitude = rotated_state.particle.spin
    idx_root_counter = 0
    idx_suffix = get_helicity_suffix(transition.topology, rotated_state_id)

    def get_helicity_rotation(state_id: int) -> Generator[PoolSum, None, None]:
        parent_id = get_parent_id(topology, state_id)
        if parent_id is None:
            return
        nonlocal idx_root_counter
        idx_root = __GREEK_INDEX_NAMES[idx_root_counter]
        next_idx_root = __GREEK_INDEX_NAMES[idx_root_counter + 1]
        idx_root_counter += 1
        if is_opposite_helicity_state(topology, state_id):
            state_id = get_sibling_state_id(topology, state_id)
        phi, theta = get_helicity_angle_symbols(topology, state_id)
        no_zero_spin = transition.states[rotated_state_id].particle.mass == 0.0
        yield formulate_helicity_rotation(
            spin_magnitude,
            spin_projection=sp.Symbol(f"{next_idx_root}{idx_suffix}", rational=True),
            m_prime=sp.Symbol(f"{idx_root}{idx_suffix}", rational=True),
            alpha=phi,
            beta=theta,
            gamma=0,
            no_zero_spin=no_zero_spin,
        )
        yield from get_helicity_rotation(parent_id)

    rotations = get_helicity_rotation(rotated_state_id)
    summation = __multiply_pool_sums(list(rotations))
    if len(summation.indices) == 1:
        idx_root = __GREEK_INDEX_NAMES[idx_root_counter]
        dangling_idx = sp.Symbol(f"{idx_root}{idx_suffix}", rational=True)
        return summation.subs(dangling_idx, helicity_symbol)
    return summation


def formulate_wigner_rotation(
    transition: StateTransition,
    rotated_state_id: int,
    helicity_symbol: sp.Symbol,
    m_prime: sp.Symbol,
) -> PoolSum:
    """Formulate the spin rotation matrices for a Wigner rotation.

    A **Wigner rotation** is the 'average' rotation that results form a chain of Lorentz
    boosts to a new reference frame with regard to a direct boost. See
    :cite:`Marangotto:2019ucc`, p.6, especially Eq.(36).

    Args:
        transition: The `~qrules.topology.Transition` in which you
            want to rotate one of the spin states.
        rotated_state_id: The state ID of a spin `~qrules.transition.State`
            that you want to rotate.
        helicity_symbol: Optional `~sympy.core.symbol.Symbol` for :math:`m` in
            :math:`D^s_{mm'}`. Falls back to the value of
            `~qrules.transition.State.spin_projection` embedded in the provided
            :code:`transition`.
        m_prime: The summation symbol :math:`m'` that should be used when
            summing over the Wigner-:math:`D` functions for this rotation.
    """
    state = transition.states[rotated_state_id]
    no_zero_spin = state.particle.mass == 0.0
    suffix = get_helicity_suffix(transition.topology, rotated_state_id)
    if helicity_symbol is None:
        spin_projection = state.spin_projection
    else:
        spin_projection = helicity_symbol
    return formulate_helicity_rotation(
        spin_magnitude=state.particle.spin,
        spin_projection=spin_projection,
        m_prime=m_prime,
        alpha=sp.Symbol(f"alpha{suffix}", real=True),
        beta=sp.Symbol(f"beta{suffix}", real=True),
        gamma=sp.Symbol(f"gamma{suffix}", real=True),
        no_zero_spin=no_zero_spin,
    )


def formulate_helicity_rotation(
    spin_magnitude,
    spin_projection,
    m_prime,
    alpha,
    beta,
    gamma,
    no_zero_spin: bool = False,
) -> PoolSum:
    r"""Formulate action of an Euler rotation on a spin state.

    When rotation a spin state :math:`\left|s,m\right\rangle` over `Euler angles
    <https://en.wikipedia.org/wiki/Euler_angles>`_ :math:`\alpha,\beta,\gamma`, the new
    state can be expressed in terms of other spin states :math:`\left|s,m'\right\rangle`
    with the help of Wigner-:math:`D` expansion coefficients:

    .. math::
        :label: formulate_helicity_rotation

        R(\alpha,\beta,\gamma)\left|s,m\right\rangle = \sum^s_{m'=-s}
        D^s_{m',m}\left(\alpha,\beta,\gamma\right) \left|s,m'\right\rangle

    See :cite:`Marangotto:2019ucc`, Eq.(B.5).

    This function gives the summation over these Wigner-:math:`D` functions and can be
    used for spin alignment following :cite:`Marangotto:2019ucc`, Eq.(45).

    Args:
        spin_magnitude: Spin magnitude :math:`s` of spin state that is being
            rotated.
        spin_projection: Spin projection component :math:`m` of the spin state
            that is being rotated.
        m_prime: A index `~sympy.core.symbol.Symbol` or `~sympy.core.symbol.Dummy`
            that represents :math:`m'` helicities in Eq.
            :eq:`formulate_helicity_rotation`.

        alpha: First Euler angle.
        beta: Second Euler angle.
        gamma: Third Euler angle.
        no_zero_spin: Skip value :code:`0.0` in the generated spin projection range.
            Useful for massless particles.

    Example
    -------
    >>> a, b, c, i = sp.symbols("a b c i")
    >>> formulate_helicity_rotation(0, 0, i, a, b, c)
    PoolSum(WignerD(0, 0, i, a, b, c), (i, (0,)))
    >>> formulate_helicity_rotation(1 / 2, -1 / 2, i, a, b, c)
    PoolSum(WignerD(1/2, -1/2, i, a, b, c), (i, (-1/2, 1/2)))
    """
    from sympy.physics.quantum.spin import Rotation as Wigner  # noqa: PLC0415

    helicities = map(sp.Rational, create_spin_range(spin_magnitude, no_zero_spin))
    return PoolSum(
        Wigner.D(
            j=__rationalize(spin_magnitude),
            m=__rationalize(spin_projection),
            mp=m_prime,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        ),
        (m_prime, list(helicities)),
    )


def get_opposite_helicity_sign(topology: Topology, state_id: int) -> Literal[-1, 1]:
    if state_id != -1 and is_opposite_helicity_state(topology, state_id):
        return -1
    return 1


def __multiply_pool_sums(sum_expressions: Sequence[PoolSum]) -> PoolSum:
    if len(sum_expressions) == 0:
        msg = f"Product needs at least one {PoolSum.__name__}"
        raise ValueError(msg)
    product = sp.Mul(*[pool_sum.expression for pool_sum in sum_expressions])
    combined_indices = []
    for pool_sum in sum_expressions:
        combined_indices.extend(pool_sum.indices)
    return PoolSum(product, *combined_indices)


_BasicType = TypeVar("_BasicType", bound=sp.Basic)


@overload
def __rationalize(value: _BasicType) -> _BasicType: ...


@overload
def __rationalize(value) -> sp.Rational:  # type: ignore[misc]
    ...


def __rationalize(value):
    if isinstance(value, sp.Basic):
        return value
    return sp.Rational(value)


__GREEK_INDEX_NAMES = ("lambda", "mu", "nu", "xi", "alpha", "beta", "gamma")
