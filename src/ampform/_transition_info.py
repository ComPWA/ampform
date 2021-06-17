from typing import Dict, Iterable, List, Optional, Tuple, Union

from qrules import InteractionProperties, ParticleCollection
from qrules.particle import Spin
from qrules.topology import Topology
from qrules.transition import StateTransition


def group_transitions(
    transitions: Iterable[StateTransition],
) -> List[List[StateTransition]]:
    """Match final and initial states in groups.

    Each `~qrules.transition.StateTransition` corresponds to a specific state
    transition amplitude. This function groups together transitions, which have the
    same initial and final state (including spin). This is needed to determine
    the coherency of the individual amplitude parts.
    """
    transition_groups: Dict[Tuple[tuple, tuple], List[StateTransition]] = {}
    for transition in transitions:
        initial_state_ids = transition.topology.outgoing_edge_ids
        final_state_ids = transition.topology.incoming_edge_ids
        transition_group = (
            tuple(
                sorted(
                    [
                        (
                            transition.states[i].particle.name,
                            transition.states[i].spin_projection,
                        )
                        for i in initial_state_ids
                    ]
                )
            ),
            tuple(
                sorted(
                    [
                        (
                            transition.states[i].particle.name,
                            transition.states[i].spin_projection,
                        )
                        for i in final_state_ids
                    ]
                )
            ),
        )
        if transition_group not in transition_groups:
            transition_groups[transition_group] = []
        transition_groups[transition_group].append(transition)

    return list(transition_groups.values())


def determine_attached_final_state(
    topology: Topology, state_id: int
) -> List[int]:
    """Determine all final state particles of a transition.

    These are attached downward (forward in time) for a given edge (resembling
    the root).
    """
    edge = topology.edges[state_id]
    if edge.ending_node_id is None:
        return [state_id]
    return sorted(
        topology.get_originating_final_state_edge_ids(edge.ending_node_id)
    )


def get_prefactor(
    transition: StateTransition,
) -> float:
    """Calculate the product of all prefactors defined in this transition."""
    prefactor = 1.0
    for node_id in transition.topology.nodes:
        interaction = transition.interactions[node_id]
        if interaction:
            temp_prefactor = __validate_float_type(
                interaction.parity_prefactor
            )
            if temp_prefactor is not None:
                prefactor *= temp_prefactor
    return prefactor


def __validate_float_type(
    interaction_property: Optional[Union[Spin, float]]
) -> Optional[float]:
    if interaction_property is not None and not isinstance(
        interaction_property, (float, int)
    ):
        raise TypeError(
            f"{interaction_property.__class__.__name__} is not of type {float.__name__}"
        )
    return interaction_property


def generate_particle_collection(
    transitions: List[StateTransition],
) -> ParticleCollection:
    particles = ParticleCollection()
    for transition in transitions:
        for state in transition.states.values():
            if state.particle not in particles:
                particles.add(state.particle)
    return particles


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


def assert_isobar_topology(topology: Topology) -> None:
    for node_id in topology.nodes:
        parent_state_ids = topology.get_edge_ids_ingoing_to_node(node_id)
        if len(parent_state_ids) != 1:
            raise ValueError(
                f"Node {node_id} has {len(parent_state_ids)} parent edges,"
                " so this is not an isobar decay"
            )
        child_state_ids = topology.get_edge_ids_outgoing_from_node(node_id)
        if len(child_state_ids) != 2:
            raise ValueError(
                f"Node {node_id} decays to {len(child_state_ids)} edges,"
                " so this is not an isobar decay"
            )
