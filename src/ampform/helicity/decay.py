"""Extract two-body decay info from a `~qrules.transition.StateTransition`."""

from typing import Tuple

import attr
from qrules.particle import Spin
from qrules.quantum_numbers import InteractionProperties
from qrules.transition import State, StateTransition


@attr.s(auto_attribs=True, frozen=True)
class StateWithID(State):
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


@attr.s(auto_attribs=True, frozen=True)
class TwoBodyDecay:
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

    def extract_angular_momentum(self) -> int:
        angular_momentum = self.interaction.l_magnitude
        if angular_momentum is not None:
            return angular_momentum
        spin_magnitude = self.parent.particle.spin
        if spin_magnitude.is_integer():
            return int(spin_magnitude)
        raise ValueError(
            f"Spin magnitude ({spin_magnitude}) of single particle state cannot be"
            f" used as the angular momentum as it is not integral!"
        )


def get_angular_momentum(interaction: InteractionProperties) -> Spin:
    l_magnitude = interaction.l_magnitude
    l_projection = interaction.l_projection
    if l_magnitude is None or l_projection is None:
        raise TypeError(
            "Angular momentum L not defined!", l_magnitude, l_projection
        )
    return Spin(l_magnitude, l_projection)


def get_coupled_spin(interaction: InteractionProperties) -> Spin:
    s_magnitude = interaction.s_magnitude
    s_projection = interaction.s_projection
    if s_magnitude is None or s_projection is None:
        raise TypeError("Coupled spin S not defined!")
    return Spin(s_magnitude, s_projection)
