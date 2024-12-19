"""Extract two-body decay info from a `~qrules.topology.Transition`."""

from __future__ import annotations

import collections
import sys
from functools import cache, singledispatch
from typing import TYPE_CHECKING

from attrs import frozen
from qrules.quantum_numbers import InteractionProperties
from qrules.transition import ReactionInfo, State, StateTransition

from ampform._qrules import get_qrules_version

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qrules.topology import Topology

from typing import Literal

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard


@frozen
class StateWithID(State):
    """Extension of `~qrules.transition.State` that embeds the state ID."""

    id: int

    @classmethod
    def from_transition(cls, transition: StateTransition, state_id: int) -> StateWithID:
        state = transition.states[state_id]
        return cls(
            id=state_id,
            particle=state.particle,
            spin_projection=state.spin_projection,
        )


@frozen
class TwoBodyDecay:
    """Two-body sub-decay in a `~qrules.topology.Transition`.

    This container class ensures that:

    1. a selected node in a `~qrules.topology.Transition` is indeed a 1-to-2 body
       decay

    2. its two `.children` are sorted by whether they decay further or not (see
       `.get_helicity_angle_symbols`, `.formulate_isobar_wigner_d`, and
       `.formulate_isobar_cg_coefficients`).

    3. the `.TwoBodyDecay` is hashable, so that it can be used as a key (see
       `.DynamicsSelector.assign`.)
    """

    parent: StateWithID
    children: tuple[StateWithID, StateWithID]
    interaction: InteractionProperties

    @staticmethod
    def create(obj) -> TwoBodyDecay:
        """Create a `TwoBodyDecay` instance from an arbitrary object.

        More implementations of :meth:`create` can be implemented with
        :func:`@ampform.helicity.decay._create_two_body_decay.register(TYPE)
        <functools.singledispatch>`.
        """
        return _create_two_body_decay(obj)

    @classmethod
    def from_transition(cls, transition: StateTransition, node_id: int) -> TwoBodyDecay:
        topology = transition.topology
        in_state_ids = topology.get_edge_ids_ingoing_to_node(node_id)
        out_state_ids = topology.get_edge_ids_outgoing_from_node(node_id)
        if len(in_state_ids) != 1 or len(out_state_ids) != 2:  # noqa: PLR2004
            msg = f"Node {node_id} does not represent a 1-to-2 body decay!"
            raise ValueError(msg)
        ingoing_state_id = next(iter(in_state_ids))
        out_state_id1, out_state_id2, *_ = tuple(out_state_ids)
        if is_opposite_helicity_state(topology, out_state_id1):
            out_state_id2, out_state_id1 = out_state_id1, out_state_id2
        return cls(
            parent=StateWithID.from_transition(transition, ingoing_state_id),
            children=(
                StateWithID.from_transition(transition, out_state_id1),
                StateWithID.from_transition(transition, out_state_id2),
            ),
            interaction=transition.interactions[node_id],
        )


@singledispatch
def _create_two_body_decay(obj) -> TwoBodyDecay:
    msg = f"Cannot create a {TwoBodyDecay.__name__} from a {type(obj).__name__}"
    raise NotImplementedError(msg)


@_create_two_body_decay.register(TwoBodyDecay)
def _(obj: TwoBodyDecay) -> TwoBodyDecay:
    return obj


@_create_two_body_decay.register(tuple)
def _(obj: tuple) -> TwoBodyDecay:
    if len(obj) == 2:  # noqa: PLR2004
        transition, node_id = obj
        if _is_qrules_state_transition(transition) and isinstance(node_id, int):
            return TwoBodyDecay.from_transition(transition, node_id)
    msg = f"Cannot create a {TwoBodyDecay.__name__} from {obj}"
    raise NotImplementedError(msg)


def _is_qrules_state_transition(obj) -> TypeGuard[StateTransition]:
    if get_qrules_version() >= (0, 10):
        from qrules.topology import FrozenTransition  # noqa: PLC0415

        if isinstance(obj, FrozenTransition):
            if any(not isinstance(s, State) for s in obj.states.values()):
                return False
            return all(
                isinstance(i, InteractionProperties) for i in obj.interactions.values()
            )
    return get_qrules_version() < (0, 10) and isinstance(obj, StateTransition)  # type: ignore[misc]


@cache
def is_opposite_helicity_state(topology: Topology, state_id: int) -> bool:
    """Determine if an edge is an "opposite helicity" state.

    This function provides a deterministic way of identifying states in a
    `~qrules.topology.Topology` as "opposite helicity" vs "helicity" state. It enforces
    that:

    1. state :code:`0` is never an opposite helicity state
    2. the sibling of an opposite helicity state is a helicity state.

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> for topology in topologies:
    ...     assert not is_opposite_helicity_state(topology, state_id=0)
    ...     for state_id in set(topology.edges) - topology.incoming_edge_ids:
    ...         sibling_id = get_sibling_state_id(topology, state_id)
    ...         assert is_opposite_helicity_state(
    ...             topology, state_id
    ...         ) != is_opposite_helicity_state(topology, sibling_id)

    The Wigner-:math:`D` function for a two-particle state treats one helicity with a
    negative sign. This sign originates from Eq.(13) in
    :cite:`Jacob:1959at` (see also Eq.(6) in
    :cite:`Marangotto:2019ucc`). Following :cite:`Marangotto:2019ucc`, we call the state
    that gets this minus sign the **"opposite helicity" state**. The other state is
    called **helicity state**. The choice of (opposite) helicity state affects not only
    the sign in the Wigner-:math:`D` function, but also the choice of angles: the
    argument of the Wigner-:math:`D` function returned by
    :func:`.formulate_isobar_wigner_d` are the angles of the helicity state.
    """
    sibling_id = get_sibling_state_id(topology, state_id)
    state_fs_ids = determine_attached_final_state(topology, state_id)
    sibling_fs_ids = determine_attached_final_state(topology, sibling_id)
    return tuple(state_fs_ids) > tuple(sibling_fs_ids)


def get_sibling_state_id(topology: Topology, state_id: int) -> int:
    r"""Get the sibling state ID for a state in an isobar decay.

    Example
    -------
    .. code-block::

        -- 3 -- 0
            \
             4 -- 1
              \
               2

    The sibling state of :code:`1` is :code:`2` and the sibling state of :code:`3` is
    :code:`4`.
    """
    parent_node = topology.edges[state_id].originating_node_id
    if parent_node is None:
        msg = f"State {state_id} is an incoming edge and does not have siblings."
        raise ValueError(msg)
    out_state_ids = topology.get_edge_ids_outgoing_from_node(parent_node)
    out_state_ids.remove(state_id)
    if len(out_state_ids) != 1:
        msg = "Not an isobar decay"
        raise ValueError(msg)
    return next(iter(out_state_ids))


@cache
def get_spectator_id(topology: Topology) -> Literal[1, 2, 3]:
    assert_three_body_decay(topology)
    decay_products = topology.get_edge_ids_outgoing_from_node(1)
    spectator_id_candidates = topology.outgoing_edge_ids - decay_products
    return next(iter(spectator_id_candidates))  # type: ignore[arg-type]


@cache
def get_decay_product_ids(
    topology: Topology,
) -> tuple[Literal[1, 2, 3], Literal[1, 2, 3]]:
    assert_three_body_decay(topology)
    decay_products = topology.get_edge_ids_outgoing_from_node(1)
    return tuple(sorted(decay_products))  # type: ignore[return-value]


def get_helicity_info(
    transition: StateTransition, node_id: int
) -> tuple[State, tuple[State, State]]:
    """Extract in- and outgoing states for a two-body decay node."""
    assert_two_body_decay(transition.topology, node_id)
    in_edge_ids = transition.topology.get_edge_ids_ingoing_to_node(node_id)
    out_edge_ids = transition.topology.get_edge_ids_outgoing_from_node(node_id)
    in_helicity_list = get_sorted_states(transition, in_edge_ids)
    out_helicity_list = get_sorted_states(transition, out_edge_ids)
    return (
        in_helicity_list[0],
        (out_helicity_list[0], out_helicity_list[1]),
    )


def get_parent_id(topology: Topology, state_id: int) -> int | None:
    """Get the edge ID of the edge from which this state decayed.

    .. warning:: This only works on 1-to-:math:`n` isobar topologies.

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(3)
    >>> topology = topologies[0]
    >>> get_parent_id(topology, state_id=0)
    -1
    >>> get_parent_id(topology, state_id=1)  # parent is the resonance
    3
    >>> get_parent_id(topology, state_id=2)
    3
    >>> get_parent_id(topology, state_id=3)
    -1
    >>> get_parent_id(topology, state_id=-1)  # already the top particle
    """
    edge = topology.edges[state_id]
    if edge.originating_node_id is None:
        return None
    incoming_edge_ids = tuple(
        topology.get_edge_ids_ingoing_to_node(edge.originating_node_id)
    )
    if len(incoming_edge_ids) != 1:
        msg = f"{StateTransition.__name__} is not an isobar decay"
        raise ValueError(msg)
    return incoming_edge_ids[0]


def list_decay_chain_ids(topology: Topology, state_id: int) -> list[int]:
    """Get the edge ID of the edge from which this state decayed.

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(3)
    >>> topology = topologies[0]
    >>> list_decay_chain_ids(topology, state_id=0)
    [0, -1]
    >>> list_decay_chain_ids(topology, state_id=1)
    [1, 3, -1]
    >>> list_decay_chain_ids(topology, state_id=2)
    [2, 3, -1]
    >>> list_decay_chain_ids(topology, state_id=-1)
    [-1]
    """
    assert_isobar_topology(topology)
    parent_list = []
    current_id: int | None = state_id
    while current_id is not None:
        parent_list.append(current_id)
        current_id = get_parent_id(topology, current_id)
    return parent_list


def get_sorted_states(
    transition: StateTransition, state_ids: Iterable[int]
) -> list[State]:
    """Get a sorted list of `~qrules.transition.State` instances.

    In order to ensure correct naming of amplitude coefficients the list has to be
    sorted by name. The same coefficient names have to be created for two transitions
    that only differ from a kinematic standpoint (swapped external edges).
    """
    states = [transition.states[i] for i in state_ids]
    return sorted(states, key=lambda s: s.particle.name)


def assert_isobar_topology(topology: Topology) -> None:
    for node_id in topology.nodes:
        assert_two_body_decay(topology, node_id)


@cache
def assert_three_body_decay(topology: Topology) -> None:
    n_initial = len(topology.incoming_edge_ids)
    n_final = len(topology.outgoing_edge_ids)
    if n_initial != 1 or n_final != 3:  # noqa: PLR2004
        msg = (
            "Only three-body decays are supported. This is a"
            f" {n_initial}-to-{n_final} decay."
        )
        raise ValueError(msg)
    if topology.incoming_edge_ids != {0} or topology.outgoing_edge_ids != {1, 2, 3}:
        msg = (
            "Please use `qrules.topology.Topology.relabel_edges()` to relabel the final"
            " states IDs to [1, 2, 3] and the initial state ID to 0."
        )
        raise ValueError(msg)


def assert_two_body_decay(topology: Topology, node_id: int) -> None:
    parent_state_ids = topology.get_edge_ids_ingoing_to_node(node_id)
    if len(parent_state_ids) != 1:
        msg = (
            f"Node {node_id} has {len(parent_state_ids)} parent states, so this is not"
            " an isobar decay"
        )
        raise ValueError(msg)
    child_state_ids = topology.get_edge_ids_outgoing_from_node(node_id)
    if len(child_state_ids) != 2:  # noqa: PLR2004
        msg = (
            f"Node {node_id} decays to {len(child_state_ids)} states, so this is not an"
            " isobar decay"
        )
        raise ValueError(msg)


def determine_attached_final_state(topology: Topology, state_id: int) -> list[int]:
    """Determine all final state particles of a transition.

    These are attached downward (forward in time) for a given edge (resembling the
    root).

    Example
    -------
    For **edge 5** in Figure :ref:`one-to-five-topology-0`, we get:

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> determine_attached_final_state(topologies[3], state_id=5)
    [0, 3, 4]
    >>> import pytest
    >>> from ampform._qrules import get_qrules_version
    >>> if get_qrules_version() < (0, 10):
    ...     pytest.skip("Doctest only works for qrules>=0.10")
    """
    edge = topology.edges[state_id]
    if edge.ending_node_id is None:
        return [state_id]
    return sorted(topology.get_originating_final_state_edge_ids(edge.ending_node_id))


@singledispatch
def get_outer_state_ids(obj: ReactionInfo | StateTransition) -> list[int]:
    msg = f"Cannot get outer state IDs from a {type(obj).__name__}"
    raise NotImplementedError(msg)


def __convert_state_transition(transition: StateTransition) -> list[int]:
    outer_state_ids = list(transition.initial_states)
    outer_state_ids += sorted(transition.final_states)
    return outer_state_ids


if get_qrules_version() < (0, 10):
    get_outer_state_ids.register(StateTransition)(__convert_state_transition)
else:
    from qrules.topology import FrozenTransition

    get_outer_state_ids.register(FrozenTransition)(__convert_state_transition)


@get_outer_state_ids.register(ReactionInfo)
def _(reaction: ReactionInfo) -> list[int]:
    return get_outer_state_ids(reaction.transitions[0])


def get_prefactor(transition: StateTransition) -> float:
    """Calculate the product of all prefactors defined in this transition.

    .. seealso:: `qrules.quantum_numbers.InteractionProperties.parity_prefactor`
    """
    prefactor = 1.0
    for node_id in transition.topology.nodes:
        interaction = transition.interactions[node_id]
        if interaction and interaction.parity_prefactor is not None:
            prefactor *= interaction.parity_prefactor
    return prefactor


def group_by_spin_projection(
    transitions: Iterable[StateTransition],
) -> list[list[StateTransition]]:
    """Match final and initial states in groups.

    Each `~qrules.topology.Transition` corresponds to a specific state transition
    amplitude. This function groups together transitions, which have the same initial
    and final state (including spin). This is needed to determine the coherency of the
    individual amplitude parts.
    """
    transition_groups: collections.defaultdict[
        tuple[
            tuple[tuple[str, float], ...],
            tuple[tuple[str, float], ...],
        ],
        list[StateTransition],
    ] = collections.defaultdict(list)
    for transition in transitions:
        initial_state = sorted(
            (
                transition.states[i].particle.name,
                transition.states[i].spin_projection,
            )
            for i in transition.topology.incoming_edge_ids
        )
        final_state = sorted(
            (
                transition.states[i].particle.name,
                transition.states[i].spin_projection,
            )
            for i in transition.topology.outgoing_edge_ids
        )
        group_key = (tuple(initial_state), tuple(final_state))
        transition_groups[group_key].append(transition)

    return list(transition_groups.values())


def group_by_topology(
    transitions: Iterable[StateTransition],
) -> dict[Topology, list[StateTransition]]:
    """Group state transitions by different `~qrules.topology.Topology`."""
    transition_groups = collections.defaultdict(list)
    for transition in transitions:
        transition_groups[transition.topology].append(transition)
    return dict(transition_groups)
