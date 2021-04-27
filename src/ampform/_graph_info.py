from typing import Dict, List, Optional, Tuple, Union

from qrules import InteractionProperties, ParticleCollection
from qrules.particle import ParticleWithSpin, Spin
from qrules.topology import StateTransitionGraph, Topology


def group_graphs_same_initial_and_final(
    graphs: List[StateTransitionGraph[ParticleWithSpin]],
) -> List[List[StateTransitionGraph[ParticleWithSpin]]]:
    """Match final and initial states in groups.

    Each graph corresponds to a specific state transition amplitude.
    This function groups together graphs, which have the same initial and final
    state (including spin). This is needed to determine the coherency of the
    individual amplitude parts.
    """
    graph_groups: Dict[
        Tuple[tuple, tuple], List[StateTransitionGraph[ParticleWithSpin]]
    ] = {}
    for graph in graphs:
        ise = graph.topology.outgoing_edge_ids
        fse = graph.topology.incoming_edge_ids
        graph_group = (
            tuple(
                sorted(
                    [
                        (
                            graph.get_edge_props(x)[0].name,
                            graph.get_edge_props(x)[1],
                        )
                        for x in ise
                    ]
                )
            ),
            tuple(
                sorted(
                    [
                        (
                            graph.get_edge_props(x)[0].name,
                            graph.get_edge_props(x)[1],
                        )
                        for x in fse
                    ]
                )
            ),
        )
        if graph_group not in graph_groups:
            graph_groups[graph_group] = []
        graph_groups[graph_group].append(graph)

    graph_group_list = list(graph_groups.values())
    return graph_group_list


def determine_attached_final_state(
    topology: Topology, edge_id: int
) -> List[int]:
    """Determine all final state particles of a graph.

    These are attached downward (forward in time) for a given edge (resembling
    the root).
    """
    edge = topology.edges[edge_id]
    if edge.ending_node_id is None:
        return [edge_id]
    return sorted(
        topology.get_originating_final_state_edge_ids(edge.ending_node_id)
    )


def get_prefactor(
    graph: StateTransitionGraph[ParticleWithSpin],
) -> float:
    """Calculate the product of all prefactors defined in this graph."""
    prefactor = 1.0
    for node_id in graph.topology.nodes:
        node_props = graph.get_node_props(node_id)
        if node_props:
            temp_prefactor = __validate_float_type(node_props.parity_prefactor)
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
    graphs: List[StateTransitionGraph[ParticleWithSpin]],
) -> ParticleCollection:
    particles = ParticleCollection()
    for graph in graphs:
        for edge_props in map(graph.get_edge_props, graph.topology.edges):
            particle, _ = edge_props
            if particle not in particles:
                particles.add(particle)
    return particles


def get_angular_momentum(node_props: InteractionProperties) -> Spin:
    l_magnitude = node_props.l_magnitude
    l_projection = node_props.l_projection
    if l_magnitude is None or l_projection is None:
        raise TypeError(
            "Angular momentum L not defined!", l_magnitude, l_projection
        )
    return Spin(l_magnitude, l_projection)


def get_coupled_spin(node_props: InteractionProperties) -> Spin:
    s_magnitude = node_props.s_magnitude
    s_projection = node_props.s_projection
    if s_magnitude is None or s_projection is None:
        raise TypeError("Coupled spin S not defined!")
    return Spin(s_magnitude, s_projection)


def assert_isobar_topology(topology: Topology) -> None:
    for node_id in topology.nodes:
        parent_edge_ids = topology.get_edge_ids_ingoing_to_node(node_id)
        if len(parent_edge_ids) != 1:
            raise ValueError(
                f"Node {node_id} has {len(parent_edge_ids)} parent edges,"
                " so this is not an isobar decay"
            )
        child_edge_ids = topology.get_edge_ids_outgoing_from_node(node_id)
        if len(child_edge_ids) != 2:
            raise ValueError(
                f"Node {node_id} decays to {len(child_edge_ids)} edges,"
                " so this is not an isobar decay"
            )
