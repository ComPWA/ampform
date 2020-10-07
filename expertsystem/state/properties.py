"""Collection of data structures and functions for particle information.

This module defines a particle as a collection of quantum numbers and things
related to this.
"""
from collections import OrderedDict
from copy import deepcopy
from itertools import permutations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

from numpy import arange

from expertsystem.data import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    NodeQuantumNumber,
    Parity,
    ParticleCollection,
    ParticleWithSpin,
    Spin,
)
from expertsystem.topology import StateTransitionGraph


def create_spin_domain(
    list_of_magnitudes: List[float], set_projection_zero: bool = False
) -> List[Spin]:
    domain_list = []
    for mag in list_of_magnitudes:
        if set_projection_zero:
            domain_list.append(
                Spin(mag, 0)
                if isinstance(mag, int) or mag.is_integer()
                else Spin(mag, mag)
            )
        else:
            for proj in arange(-mag, mag + 1, 1.0):  # type: ignore
                domain_list.append(Spin(mag, proj))
    return domain_list


def get_particle_property(
    edge_property: ParticleWithSpin, qn_type: Type[EdgeQuantumNumber]
) -> Optional[Union[float, int]]:
    """Convert a data member of `.Particle` into one of `.EdgeQuantumNumbers`.

    The `.solving` model requires a list of 'flat' values, such as `int` and
    `float`. It cannot handle `~.data.Spin` (which contains `~.Spin.magnitude` and
    `~.Spin.projection`). The `.solving` module also works with spin
    projection, which a general `.Particle` instance does not carry.
    """
    particle, spin_projection = edge_property
    value = None
    if hasattr(particle, qn_type.__name__):
        value = getattr(particle, qn_type.__name__)
    else:
        if qn_type is EdgeQuantumNumbers.spin_magnitude:
            value = particle.spin
        elif qn_type is EdgeQuantumNumbers.spin_projection:
            value = spin_projection
        if particle.isospin is not None:
            if qn_type is EdgeQuantumNumbers.isospin_magnitude:
                value = particle.isospin.magnitude
            elif qn_type is EdgeQuantumNumbers.isospin_projection:
                value = particle.isospin.projection

    if isinstance(value, Parity):
        return int(value)
    return value


def get_interaction_property(
    interaction_properties: Dict[Type[NodeQuantumNumber], Union[int, float]],
    qn_type: Type[NodeQuantumNumber],
) -> Optional[Union[int, float]]:
    found_prop = None
    if qn_type in interaction_properties:
        found_prop = interaction_properties[qn_type]

    return found_prop


class CompareGraphNodePropertiesFunctor:
    """Functor for comparing graph elements."""

    def __init__(
        self,
        ignored_qn_list: Optional[Set[Type[NodeQuantumNumber]]] = None,
    ) -> None:
        self.__ignored_qn_list = ignored_qn_list if ignored_qn_list else set()

    def __call__(
        self,
        node_props1: Dict[int, Dict[str, Any]],
        node_props2: Dict[int, Dict[str, Any]],
    ) -> bool:
        for node_id, node_props in node_props1.items():
            other_node_props = node_props2[node_id]
            if {
                k: v
                for k, v in node_props.items()
                if k not in self.__ignored_qn_list
            } != {
                k: v
                for k, v in other_node_props.items()
                if k not in self.__ignored_qn_list
            }:
                return False
        return True


def filter_particles(
    particle_db: ParticleCollection,
    allowed_particle_names: List[str],
) -> ParticleCollection:
    """Filters `.ParticleCollection` based on the allowed particle names."""
    allowed_particles = ParticleCollection()
    if len(allowed_particle_names) == 0:
        return particle_db

    for particle_name in allowed_particle_names:
        # if isinstance(particle_label, int):
        #     allowed_particles.add(particle_db.find(particle_label))
        # elif isinstance(particle_label, str):
        subset = particle_db.filter(
            lambda p: particle_name  # pylint: disable=cell-var-from-loop
            in p.name
        )
        allowed_particles.merge(subset)
    return allowed_particles


def match_external_edges(
    graphs: List[StateTransitionGraph[ParticleWithSpin]],
) -> None:
    if not isinstance(graphs, list):
        raise TypeError("graphs argument is not of type list!")
    if not graphs:
        return
    ref_graph_id = 0
    _match_external_edge_ids(graphs, ref_graph_id, "get_final_state_edges")
    _match_external_edge_ids(graphs, ref_graph_id, "get_initial_state_edges")


def _match_external_edge_ids(  # pylint: disable=too-many-locals
    graphs: List[StateTransitionGraph[ParticleWithSpin]],
    ref_graph_id: int,
    external_edge_getter_function: str,
) -> None:
    ref_graph = graphs[ref_graph_id]
    # create external edge to particle mapping
    ref_edge_id_particle_mapping = _create_edge_id_particle_mapping(
        ref_graph, external_edge_getter_function
    )

    for graph in graphs[:ref_graph_id] + graphs[ref_graph_id + 1 :]:
        edge_id_particle_mapping = _create_edge_id_particle_mapping(
            graph, external_edge_getter_function
        )
        # remove matching entries
        ref_mapping_copy = deepcopy(ref_edge_id_particle_mapping)
        edge_ids_mapping = {}
        for key, value in edge_id_particle_mapping.items():
            if key in ref_mapping_copy and value == ref_mapping_copy[key]:
                del ref_mapping_copy[key]
            else:
                for key_2, value_2 in ref_mapping_copy.items():
                    if value == value_2:
                        edge_ids_mapping[key] = key_2
                        del ref_mapping_copy[key_2]
                        break
        if len(ref_mapping_copy) != 0:
            raise ValueError(
                "Unable to match graphs, due to inherent graph"
                " structure mismatch"
            )
        swappings = _calculate_swappings(edge_ids_mapping)
        for edge_id1, edge_id2 in swappings.items():
            graph.swap_edges(edge_id1, edge_id2)


def perform_external_edge_identical_particle_combinatorics(
    graph: StateTransitionGraph,
) -> List[StateTransitionGraph]:
    """Create combinatorics clones of the `.StateTransitionGraph`.

    In case of identical particles in the initial or final state. Only
    identical particles, which do not enter or exit the same node allow for
    combinatorics!
    """
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph argument is not of type StateTransitionGraph!")
    temp_new_graphs = _external_edge_identical_particle_combinatorics(
        graph, "get_final_state_edges"
    )
    new_graphs = []
    for new_graph in temp_new_graphs:
        new_graphs.extend(
            _external_edge_identical_particle_combinatorics(
                new_graph, "get_initial_state_edges"
            )
        )
    return new_graphs


def _external_edge_identical_particle_combinatorics(
    graph: StateTransitionGraph[ParticleWithSpin],
    external_edge_getter_function: str,
) -> List[StateTransitionGraph]:
    # pylint: disable=too-many-locals
    new_graphs = [graph]
    edge_particle_mapping = _create_edge_id_particle_mapping(
        graph, external_edge_getter_function
    )
    identical_particle_groups: Dict[str, Set[int]] = {}
    for key, value in edge_particle_mapping.items():
        if value not in identical_particle_groups:
            identical_particle_groups[value] = set()
        identical_particle_groups[value].add(key)
    identical_particle_groups = {
        key: value
        for key, value in identical_particle_groups.items()
        if len(value) > 1
    }
    # now for each identical particle group perform all permutations
    for edge_group in identical_particle_groups.values():
        combinations = permutations(edge_group)
        graph_combinations = set()
        ext_edge_combinations = []
        ref_node_origin = graph.get_originating_node_list(edge_group)
        for comb in combinations:
            temp_edge_node_mapping = tuple(sorted(zip(comb, ref_node_origin)))
            if temp_edge_node_mapping not in graph_combinations:
                graph_combinations.add(temp_edge_node_mapping)
                ext_edge_combinations.append(dict(zip(edge_group, comb)))
        temp_new_graphs = []
        for new_graph in new_graphs:
            for combination in ext_edge_combinations:
                graph_copy = deepcopy(new_graph)
                swappings = _calculate_swappings(combination)
                for edge_id1, edge_id2 in swappings.items():
                    graph_copy.swap_edges(edge_id1, edge_id2)
                temp_new_graphs.append(graph_copy)
        new_graphs = temp_new_graphs
    return new_graphs


def _calculate_swappings(id_mapping: Dict[int, int]) -> OrderedDict:
    """Calculate edge id swappings.

    Its important to use an ordered dict as the swappings do not commute!
    """
    swappings: OrderedDict = OrderedDict()
    for key, value in id_mapping.items():
        # go through existing swappings and use them
        newkey = key
        while newkey in swappings:
            newkey = swappings[newkey]
        if value != newkey:
            swappings[value] = newkey
    return swappings


def _create_edge_id_particle_mapping(
    graph: StateTransitionGraph[ParticleWithSpin],
    external_edge_getter_function: str,
) -> Dict[int, str]:
    return {
        i: graph.edge_props[i][0].name
        for i in getattr(graph, external_edge_getter_function)()
    }
