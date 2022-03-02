"""Extract two-body decay info from a `~qrules.transition.StateTransition`."""

from functools import lru_cache, singledispatch
from typing import Any, Iterable, List, Optional, Tuple

from attrs import frozen
from qrules.quantum_numbers import InteractionProperties
from qrules.topology import Topology
from qrules.transition import State, StateTransition


@frozen
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


@frozen
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

    @staticmethod
    def create(obj: Any) -> "TwoBodyDecay":
        """Create a `TwoBodyDecay` instance from an arbitrary object.

        More implementations of :meth:`create` can be implemented with
        :func:`@ampform.helicity.decay._create_two_body_decay.register(TYPE)
        <functools.singledispatch>`.
        """
        return _create_two_body_decay(obj)

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


@singledispatch
def _create_two_body_decay(obj: Any) -> TwoBodyDecay:
    raise NotImplementedError(
        f"Cannot create a {TwoBodyDecay.__name__} from a {type(obj).__name__}"
    )


@_create_two_body_decay.register(TwoBodyDecay)
def _(obj: TwoBodyDecay) -> TwoBodyDecay:
    return obj


@_create_two_body_decay.register(tuple)
def _(obj: tuple) -> TwoBodyDecay:
    if len(obj) == 2:
        if isinstance(obj[0], StateTransition) and isinstance(obj[1], int):
            return TwoBodyDecay.from_transition(*obj)
    raise NotImplementedError(
        f"Cannot create a {TwoBodyDecay.__name__} from {obj}"
    )


@lru_cache(maxsize=None)
def is_opposite_helicity_state(topology: Topology, state_id: int) -> bool:
    """Determine if an edge is an "opposite helicity" state.

    This function provides a deterministic way of identifying states in a
    `~qrules.topology.Topology` as "opposite helicity" vs "helicity" state.
    It enforces that:

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
    ...         ) != is_opposite_helicity_state(
    ...             topology, sibling_id
    ...         )

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
    sibling_id = get_sibling_state_id(topology, state_id)
    state_fs_ids = determine_attached_final_state(topology, state_id)
    sibling_fs_ids = determine_attached_final_state(topology, sibling_id)
    return tuple(state_fs_ids) > tuple(sibling_fs_ids)


@lru_cache(maxsize=None)
def collect_topologies(
    transitions: Tuple[StateTransition, ...]
) -> List[Topology]:
    return sorted({t.topology for t in transitions})


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

    The sibling state of :code:`1` is :code:`2` and the sibling state of
    :code:`3` is :code:`4`.
    """
    parent_node = topology.edges[state_id].originating_node_id
    if parent_node is None:
        raise ValueError(
            f"State {state_id} is an incoming edge and does not have siblings."
        )
    out_state_ids = topology.get_edge_ids_outgoing_from_node(parent_node)
    out_state_ids.remove(state_id)
    if len(out_state_ids) != 1:
        raise ValueError("Not an isobar decay")
    return next(iter(out_state_ids))


def get_helicity_info(
    transition: StateTransition, node_id: int
) -> Tuple[State, Tuple[State, State]]:
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


def list_decay_chain_ids(topology: Topology, state_id: int) -> List[int]:
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
    current_id: Optional[int] = state_id
    while current_id is not None:
        parent_list.append(current_id)
        current_id = get_parent_id(topology, current_id)
    return parent_list


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


def assert_isobar_topology(topology: Topology) -> None:
    for node_id in topology.nodes:
        assert_two_body_decay(topology, node_id)


def assert_two_body_decay(topology: Topology, node_id: int) -> None:
    parent_state_ids = topology.get_edge_ids_ingoing_to_node(node_id)
    if len(parent_state_ids) != 1:
        raise ValueError(
            f"Node {node_id} has {len(parent_state_ids)} parent states,"
            " so this is not an isobar decay"
        )
    child_state_ids = topology.get_edge_ids_outgoing_from_node(node_id)
    if len(child_state_ids) != 2:
        raise ValueError(
            f"Node {node_id} decays to {len(child_state_ids)} states,"
            " so this is not an isobar decay"
        )


def determine_attached_final_state(
    topology: Topology, state_id: int
) -> List[int]:
    """Determine all final state particles of a transition.

    These are attached downward (forward in time) for a given edge (resembling
    the root).

    Example
    -------
    For **edge 5** in Figure :ref:`one-to-five-topology-0`, we get:

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> determine_attached_final_state(topologies[0], state_id=5)
    [0, 3, 4]
    """
    edge = topology.edges[state_id]
    if edge.ending_node_id is None:
        return [state_id]
    return sorted(
        topology.get_originating_final_state_edge_ids(edge.ending_node_id)
    )
