"""Extract two-body decay info from a `~qrules.transition.StateTransition`."""

from functools import singledispatch
from typing import Any, Iterable, List, Tuple

from attrs import frozen
from qrules.quantum_numbers import InteractionProperties
from qrules.transition import State, StateTransition

from ampform.kinematics import _assert_two_body_decay


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

        sorted_by_id = sorted(out_state_ids)
        final_state_ids = [
            i for i in sorted_by_id if i in topology.outgoing_edge_ids
        ]
        intermediate_state_ids = [
            i for i in sorted_by_id if i in topology.intermediate_edge_ids
        ]
        sorted_by_ending = tuple(intermediate_state_ids + final_state_ids)

        ingoing_state_id = next(iter(in_state_ids))
        out_state_id1, out_state_id2, *_ = sorted_by_ending
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
