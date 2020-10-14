"""Perform permutations on the edges of a `.StateTransitionGraph`.

In a `.StateTransitionGraph`, the edges represent quantum states, while the
nodes represent interactions. This module provides tools to permutate, modify
or extract these edge and node properties.
"""

from collections import OrderedDict
from copy import deepcopy
from decimal import Decimal
from itertools import permutations
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from expertsystem.particle import Particle, ParticleCollection

from .quantum_numbers import ParticleWithSpin
from .topology import StateTransitionGraph, Topology

StateWithSpins = Tuple[str, Sequence[float]]
StateDefinition = Union[str, StateWithSpins]


class _KinematicRepresentation:
    def __init__(
        self,
        final_state: Optional[Union[Sequence[List[Any]], List[Any]]] = None,
        initial_state: Optional[Union[Sequence[List[Any]], List[Any]]] = None,
    ) -> None:
        self.__initial_state: Optional[List[List[Any]]] = None
        self.__final_state: Optional[List[List[Any]]] = None
        if initial_state is not None:
            self.__initial_state = self.__import(initial_state)
        if final_state is not None:
            self.__final_state = self.__import(final_state)

    @property
    def initial_state(self) -> Optional[List[List[Any]]]:
        return self.__initial_state

    @property
    def final_state(self) -> Optional[List[List[Any]]]:
        return self.__final_state

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _KinematicRepresentation):
            return (
                self.initial_state == other.initial_state
                and self.final_state == other.final_state
            )
        raise ValueError(
            f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"initial_state={self.initial_state}, "
            f"final_state={self.final_state})"
        )

    def __contains__(self, other: object) -> bool:
        """Check if a `KinematicRepresentation` is contained within another.

        You can also compare with a `list` of `list` instances, such as:

        .. code-block::

            [["gamma", "pi0"], ["gamma", "pi0", "pi0"]]

        This list will be compared **only** with the
        `~KinematicRepresentation.final_state`!
        """

        def is_sublist(
            sub_representation: Optional[List[List[Any]]],
            main_representation: Optional[List[List[Any]]],
        ) -> bool:
            if main_representation is None:
                if sub_representation is None:
                    return True
                return False
            if sub_representation is None:
                return True
            for group in sub_representation:
                if group not in main_representation:
                    return False
            return True

        if isinstance(other, _KinematicRepresentation):
            return is_sublist(
                other.initial_state, self.initial_state
            ) and is_sublist(other.final_state, self.final_state)
        if isinstance(other, list):
            for item in other:
                if not isinstance(item, list):
                    raise ValueError(
                        "Comparison representation needs to be a list of lists"
                    )
            return is_sublist(other, self.final_state)
        raise ValueError(
            f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}"
        )

    def __import(
        self, nested_list: Union[Sequence[Sequence[Any]], Sequence[Any]]
    ) -> List[List[Any]]:
        return self.__sort(self.__prepare(nested_list))

    def __prepare(
        self, nested_list: Union[Sequence[Sequence[Any]], Sequence[Any]]
    ) -> List[List[Any]]:
        if len(nested_list) == 0 or not isinstance(nested_list[0], list):
            nested_list = [nested_list]
        return [
            [self.__extract_particle_name(item) for item in sub_list]
            for sub_list in nested_list
        ]

    @staticmethod
    def __sort(nested_list: List[List[Any]]) -> List[List[Any]]:
        return sorted([sorted(sub_list) for sub_list in nested_list])

    @staticmethod
    def __extract_particle_name(item: object) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, (tuple, list)) and isinstance(item[0], str):
            return item[0]
        if isinstance(item, Particle):
            return item.name
        if isinstance(item, dict) and "Name" in item:
            return str(item["Name"])
        raise ValueError(
            f"Cannot extract particle name from {item.__class__.__name__}"
        )


def _get_kinematic_representation(
    graph: StateTransitionGraph[StateWithSpins],
) -> _KinematicRepresentation:
    r"""Group final or initial states by node, sorted by length of the group.

    The resulting sorted groups can be used to check whether two
    `.StateTransitionGraph` instances are kinematically identical. For
    instance, the following two graphs:

    .. code-block::

        J/psi -- 0 -- pi0
                  \
                   1 -- gamma
                    \
                     2 -- gamma
                      \
                       pi0

        J/psi -- 0 -- pi0
                  \
                   1 -- gamma
                    \
                     2 -- pi0
                      \
                       gamma

    both result in:

    .. code-block::

        kinematic_representation.final_state == \
            [["gamma", "gamma"], ["gamma", "gamma", "pi0"], \
             ["gamma", "gamma", "pi0", "pi0"]]
        kinematic_representation.initial_state == \
            [["J/psi"], ["J/psi"]]

    and are therefore kinematically identical. The nested lists are sorted (by
    `list` length and element content) for comparisons.

    Note: more precisely, the states represented here by a `str` only also have
    a list of allowed spin projections, for instance, :code:`("J/psi", [-1,
    +1])`. Note that a `tuple` is also sortable.
    """

    def get_state_groupings(
        edge_per_node_getter: Callable[[int], List[int]]
    ) -> List[List[int]]:
        return [edge_per_node_getter(i) for i in graph.nodes]

    def fill_groupings(grouping_with_ids: List[List[Any]]) -> List[List[Any]]:
        return [
            [graph.edge_props[edge_id] for edge_id in group]
            for group in grouping_with_ids
        ]

    initial_state_edge_groups = fill_groupings(
        get_state_groupings(graph.get_originating_initial_state_edges)
    )
    final_state_edge_groups = fill_groupings(
        get_state_groupings(graph.get_originating_final_state_edges)
    )
    return _KinematicRepresentation(
        initial_state=initial_state_edge_groups,
        final_state=final_state_edge_groups,
    )


def initialize_graph(  # pylint: disable=too-many-locals
    topology: Topology,
    particles: ParticleCollection,
    initial_state: Sequence[StateDefinition],
    final_state: Sequence[StateDefinition],
    final_state_groupings: Optional[
        Union[List[List[List[str]]], List[List[str]], List[str]]
    ] = None,
) -> List[StateTransitionGraph[ParticleWithSpin]]:
    def embed_in_list(some_list: List[Any]) -> List[List[Any]]:
        if not isinstance(some_list[0], list):
            return [some_list]
        return some_list

    allowed_kinematic_groupings = None
    if final_state_groupings is not None:
        final_state_groupings = embed_in_list(final_state_groupings)
        final_state_groupings = embed_in_list(final_state_groupings)
        allowed_kinematic_groupings = [
            _KinematicRepresentation(final_state=grouping)
            for grouping in final_state_groupings
        ]

    kinematic_permutation_graphs = _generate_kinematic_permutations(
        topology=topology,
        particles=particles,
        initial_state=initial_state,
        final_state=final_state,
        allowed_kinematic_groupings=allowed_kinematic_groupings,
    )
    output_graphs = list()
    for kinematic_permutation in kinematic_permutation_graphs:
        spin_permutations = _generate_spin_permutations(
            kinematic_permutation, particles
        )
        output_graphs.extend(spin_permutations)
    return output_graphs


def _generate_kinematic_permutations(
    topology: Topology,
    particles: ParticleCollection,
    initial_state: Sequence[StateDefinition],
    final_state: Sequence[StateDefinition],
    allowed_kinematic_groupings: Optional[
        List[_KinematicRepresentation]
    ] = None,
) -> List[StateTransitionGraph[StateWithSpins]]:
    def assert_number_of_states(
        state_definitions: Sequence, edge_ids: Sequence[int]
    ) -> None:
        if len(state_definitions) != len(edge_ids):
            raise ValueError(
                "Number of state definitions is not same as number of edge IDs:"
                f"(len({state_definitions}) != len({edge_ids})"
            )

    assert_number_of_states(initial_state, topology.get_initial_state_edges())
    assert_number_of_states(final_state, topology.get_final_state_edges())

    def is_allowed_grouping(
        kinematic_representation: _KinematicRepresentation,
    ) -> bool:
        if allowed_kinematic_groupings is None:
            return True
        for allowed_kinematic_grouping in allowed_kinematic_groupings:
            if allowed_kinematic_grouping in kinematic_representation:
                return True
        return False

    initial_state_with_projections = _safe_set_spin_projections(
        initial_state, particles
    )
    final_state_with_projections = _safe_set_spin_projections(
        final_state, particles
    )

    graphs: List[StateTransitionGraph[StateWithSpins]] = list()
    kinematic_representations: List[_KinematicRepresentation] = list()
    for permutation in _generate_outer_edge_permutations(
        topology,
        initial_state_with_projections,
        final_state_with_projections,
    ):
        graph: StateTransitionGraph[
            StateWithSpins
        ] = StateTransitionGraph.from_topology(topology)
        graph.edge_props.update(permutation)
        kinematic_representation = _get_kinematic_representation(graph)
        if kinematic_representation in kinematic_representations:
            continue
        if not is_allowed_grouping(kinematic_representation):
            continue
        kinematic_representations.append(kinematic_representation)
        graphs.append(graph)

    return graphs


def _safe_set_spin_projections(
    list_of_states: Sequence[StateDefinition],
    particle_db: ParticleCollection,
) -> Sequence[StateWithSpins]:
    def safe_set_spin_projections(
        state: StateDefinition, particle_db: ParticleCollection
    ) -> StateWithSpins:
        if isinstance(state, str):
            particle_name = state
            particle = particle_db[state]
            spin_projections = list(
                arange(-particle.spin, particle.spin + 1, 1.0)
            )
            if particle.mass == 0.0:
                if 0.0 in spin_projections:
                    del spin_projections[spin_projections.index(0.0)]
            state = (particle_name, spin_projections)
        return state

    return [
        safe_set_spin_projections(state, particle_db)
        for state in list_of_states
    ]


def arange(
    x_1: float, x_2: float, delta: float
) -> Generator[float, None, None]:
    current = Decimal(x_1)
    while current < x_2:
        yield float(current)
        current += Decimal(delta)


def _generate_outer_edge_permutations(
    topology: Topology,
    initial_state: Sequence[StateWithSpins],
    final_state: Sequence[StateWithSpins],
) -> Generator[Dict[int, StateWithSpins], None, None]:
    initial_state_ids = topology.get_initial_state_edges()
    final_state_ids = topology.get_final_state_edges()
    for initial_state_permutation in permutations(initial_state):
        for final_state_permutation in permutations(final_state):
            yield dict(
                zip(
                    initial_state_ids + final_state_ids,
                    initial_state_permutation + final_state_permutation,
                )
            )


def _generate_spin_permutations(
    graph: StateTransitionGraph[StateWithSpins],
    particle_db: ParticleCollection,
) -> List[StateTransitionGraph[ParticleWithSpin]]:
    def populate_edge_with_spin_projections(
        uninitialized_graph: StateTransitionGraph[ParticleWithSpin],
        edge_id: int,
        state: StateWithSpins,
    ) -> List[StateTransitionGraph[ParticleWithSpin]]:
        particle_name, spin_projections = state
        particle = particle_db[particle_name]
        output_graph = []
        for projection in spin_projections:
            graph_copy = deepcopy(uninitialized_graph)
            graph_copy.edge_props[edge_id] = (particle, projection)
            output_graph.append(graph_copy)
        return output_graph

    edge_particle_dict = {
        edge_id: graph.edge_props[edge_id]
        for edge_id in graph.get_initial_state_edges()
        + graph.get_final_state_edges()
    }

    # now add more quantum numbers given by user (spin_projection)
    uninitialized_graph = StateTransitionGraph.from_topology(graph)
    output_graphs: List[StateTransitionGraph[ParticleWithSpin]] = [
        uninitialized_graph
    ]
    for edge_id, state in edge_particle_dict.items():
        temp_graphs = output_graphs
        output_graphs = []
        for temp_graph in temp_graphs:
            output_graphs.extend(
                populate_edge_with_spin_projections(temp_graph, edge_id, state)
            )

    return output_graphs


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
