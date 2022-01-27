"""Extract two-body decay info from a `~qrules.transition.StateTransition`."""

from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import attr
from qrules.quantum_numbers import InteractionProperties
from qrules.topology import Topology
from qrules.transition import State, StateTransition

from ampform.kinematics import _assert_two_body_decay


@attr.frozen
class StateWithID(State):
    """Extension of `~qrules.transition.State` that embeds the state ID."""

    id: int  # noqa: A003

    @classmethod
    def from_transition(
        cls, transition: StateTransition, state_id: int
    ) -> "StateWithID":
        state = transition.states[state_id]
        return cls(
            id=state_id,
            particle=state.particle,
            spin_projection=state.spin_projection,
        )


def count_parents(topology: Topology, state_id: int) -> int:
    """Count the number of parents above a certain state.

    .. warning:: This only works on 1-to-:math:`n` isobar topologies.

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(3)
    >>> topology = topologies[0]
    >>> count_parents(topology, state_id=0)
    1
    >>> count_parents(topology, state_id=1)
    2
    >>> count_parents(topology, state_id=2)
    2
    >>> count_parents(topology, state_id=3)
    1
    >>> count_parents(topology, state_id=-1)
    0
    """
    n_parents = 0
    parent_id = get_parent_id(topology, state_id)
    while parent_id is not None:
        parent_id = get_parent_id(topology, parent_id)
        n_parents += 1
    return n_parents


@attr.frozen
class TwoBodyDecay:
    """Two-body sub-decay in a `~qrules.transition.StateTransition`.

    This container class ensures that:

    1. a selected node in a `~qrules.transition.StateTransition` is indeed a
       1-to-2 body decay

    2. its two `.children` are sorted by whether they decay further or not (see
       `.get_helicity_angle_label`, `.formulate_wigner_d`, and
       `.formulate_clebsch_gordan_coefficients`).

    3. the `.TwoBodyDecay` is hashable, so that it can be used as a key (see
       `.set_dynamics`.)
    """

    parent: StateWithID
    children: Tuple[StateWithID, StateWithID]
    interaction: InteractionProperties

    @classmethod
    def from_transition(
        cls, transition: StateTransition, node_id: int
    ) -> "TwoBodyDecay":
        topology = transition.topology
        in_state_ids = topology.get_edge_ids_ingoing_to_node(node_id)
        out_state_ids = topology.get_edge_ids_outgoing_from_node(node_id)
        if len(in_state_ids) != 1 or len(out_state_ids) != 2:
            raise ValueError(
                f"Node {node_id} does not represent a 1-to-2 body decay!"
            )
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

    def extract_angular_momentum(self) -> int:
        angular_momentum = self.interaction.l_magnitude
        if angular_momentum is not None:
            return angular_momentum
        spin_magnitude = self.parent.particle.spin
        if spin_magnitude.is_integer():
            return int(spin_magnitude)
        raise ValueError(
            f"Spin magnitude ({spin_magnitude}) of single particle state"
            " cannot be used as the angular momentum as it is not integral!"
        )


@lru_cache(maxsize=None)
def is_opposite_helicity_state(topology: Topology, state_id: int) -> bool:
    """Determine if an edge is an "opposite helicity" state.

    This function provides a deterministic way of identifying states in a
    `~qrules.topology.Topology` as "opposite helicity" vs "helicity" state.

    The Wigner-:math:`D` function for a two-particle state treats one helicity
    with a negative sign. This sign originates from Eq.(13) in
    :cite:`jacobGeneralTheoryCollisions1959` (see also Eq.(6) in
    :cite:`marangottoHelicityAmplitudesGeneric2020`). Following
    :cite:`marangottoHelicityAmplitudesGeneric2020`, we call the state that
    gets this minus sign the **"opposite helicity" state**. The other state is
    called **helicity state**. The choice of (opposite) helicity state affects
    not only the sign in the Wigner-:math:`D` function, but also the choice of
    angles: the argument of the Wigner-:math:`D` function returned by
    :func:`.formulate_wigner_d` are the angles of the helicity state.
    """
    parent_node = topology.edges[state_id].originating_node_id
    if parent_node is None:
        return False
    out_state_ids = topology.get_edge_ids_outgoing_from_node(parent_node)
    sorted_by_id = sorted(out_state_ids)
    final_state_ids = [
        i for i in sorted_by_id if i in topology.outgoing_edge_ids
    ]
    intermediate_state_ids = [
        i for i in sorted_by_id if i in topology.intermediate_edge_ids
    ]
    sorted_by_ending = tuple(intermediate_state_ids + final_state_ids)
    helicity_state_id = sorted_by_ending[0]
    if state_id == helicity_state_id:
        return False
    return True


def get_helicity_info(
    transition: StateTransition, node_id: int
) -> Tuple[State, Tuple[State, State]]:
    """Extract in- and outgoing states for a two-body decay node."""
    _assert_two_body_decay(transition.topology, node_id)
    in_edge_ids = transition.topology.get_edge_ids_ingoing_to_node(node_id)
    out_edge_ids = transition.topology.get_edge_ids_outgoing_from_node(node_id)
    in_helicity_list = get_sorted_states(transition, in_edge_ids)
    out_helicity_list = get_sorted_states(transition, out_edge_ids)
    return (
        in_helicity_list[0],
        (out_helicity_list[0], out_helicity_list[1]),
    )


def get_parent_id(topology: Topology, state_id: int) -> Optional[int]:
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
        raise ValueError(f"{StateTransition.__name__} is not an isobar decay")
    return incoming_edge_ids[0]


def get_sorted_states(
    transition: StateTransition, state_ids: Iterable[int]
) -> List[State]:
    """Get a sorted list of `~qrules.transition.State` instances.

    In order to ensure correct naming of amplitude coefficients the list has to
    be sorted by name. The same coefficient names have to be created for two
    transitions that only differ from a kinematic standpoint (swapped external
    edges).
    """
    states = [transition.states[i] for i in state_ids]
    return sorted(states, key=lambda s: s.particle.name)
