"""Functions that steer operations of the `expertsystem`."""

import logging
from abc import ABC, abstractmethod
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
)

from expertsystem.data import NodeQuantumNumber, ParticleWithSpin, Spin
from expertsystem.nested_dicts import (
    InteractionQuantumNumberNames,
)
from expertsystem.solving import (
    EdgeSettings,
    GraphSettings,
    InteractionTypes,
    NodeSettings,
)
from expertsystem.solving.conservation_rules import Rule
from expertsystem.state.properties import (
    CompareGraphNodePropertiesFunctor,
    get_interaction_property,
)
from expertsystem.topology import StateTransitionGraph


Strength = float

GraphSettingsGroups = Dict[
    Strength, List[Tuple[StateTransitionGraph, GraphSettings]]
]


def _change_qn_domain(
    settings: Tuple[EdgeSettings, NodeSettings],
    qn_name: InteractionQuantumNumberNames,
    new_domain: List[Spin],
) -> None:
    if (
        not isinstance(settings, tuple)
        or not isinstance(settings[0], EdgeSettings)
        or not isinstance(settings[1], NodeSettings)
    ):
        raise TypeError(
            "graph_settings has to be of type Tuple[NodeSettings, EdgeSettings]"
        )

    def change_domain(qn_domains: dict) -> None:
        if qn_name in qn_domains:
            qn_domains.update({qn_name: new_domain})

    change_domain(settings[0].qn_domains)
    change_domain(settings[1].qn_domains)


def _remove_conservation_law(
    settings: Tuple[EdgeSettings, NodeSettings], cons_law: Rule
) -> None:
    if (
        not isinstance(settings, tuple)
        or not isinstance(settings[0], EdgeSettings)
        or not isinstance(settings[1], NodeSettings)
    ):
        raise TypeError(
            "graph_settings has to be of type Tuple[NodeSettings, EdgeSettings]"
        )

    def remove_rule(rule_set: Set[Rule]) -> None:
        for rule in rule_set:
            if str(rule) == str(cons_law):
                rule_set.remove(rule)
                break

    remove_rule(settings[0].conservation_rules)
    remove_rule(settings[1].conservation_rules)


def filter_interaction_types(
    valid_determined_interaction_types: List[InteractionTypes],
    allowed_interaction_types: List[InteractionTypes],
) -> List[InteractionTypes]:
    int_type_intersection = list(
        set(allowed_interaction_types)
        & set(valid_determined_interaction_types)
    )
    if int_type_intersection:
        return int_type_intersection
    logging.warning(
        "The specified list of interaction types %s"
        " does not intersect with the valid list of interaction types %s"
        ".\nUsing valid list instead.",
        allowed_interaction_types,
        valid_determined_interaction_types,
    )
    return valid_determined_interaction_types


class _InteractionDeterminationFunctorInterface(ABC):
    """Interface for interaction determination."""

    @abstractmethod
    def check(
        self,
        in_edge_props: List[ParticleWithSpin],
        out_edge_props: List[ParticleWithSpin],
        node_props: dict,
    ) -> List[InteractionTypes]:
        pass


class GammaCheck(_InteractionDeterminationFunctorInterface):
    """Conservation check for photons."""

    def check(
        self,
        in_edge_props: List[ParticleWithSpin],
        out_edge_props: List[ParticleWithSpin],
        node_props: dict,
    ) -> List[InteractionTypes]:
        int_types = list(InteractionTypes)
        for particle, _ in in_edge_props + out_edge_props:
            if "gamma" in particle.name:
                int_types = [InteractionTypes.EM, InteractionTypes.Weak]
                break
        return int_types


class LeptonCheck(_InteractionDeterminationFunctorInterface):
    """Conservation check lepton numbers."""

    def check(
        self,
        in_edge_props: List[ParticleWithSpin],
        out_edge_props: List[ParticleWithSpin],
        node_props: dict,
    ) -> List[InteractionTypes]:
        node_interaction_types = list(InteractionTypes)
        for particle, _ in in_edge_props + out_edge_props:
            if particle.is_lepton():
                if particle.name.startswith("nu("):
                    node_interaction_types = [InteractionTypes.Weak]
                    break
                node_interaction_types = [
                    InteractionTypes.EM,
                    InteractionTypes.Weak,
                ]
        return node_interaction_types


def remove_duplicate_solutions(
    solutions: List[StateTransitionGraph[ParticleWithSpin]],
    remove_qns_list: Optional[Set[Type[NodeQuantumNumber]]] = None,
    ignore_qns_list: Optional[Set[Type[NodeQuantumNumber]]] = None,
) -> List[StateTransitionGraph[ParticleWithSpin]]:
    if remove_qns_list is None:
        remove_qns_list = set()
    if ignore_qns_list is None:
        ignore_qns_list = set()
    logging.info("removing duplicate solutions...")
    logging.info(f"removing these qns from graphs: {remove_qns_list}")
    logging.info(f"ignoring qns in graph comparison: {ignore_qns_list}")

    filtered_solutions: List[StateTransitionGraph[ParticleWithSpin]] = list()
    remove_counter = 0
    for sol_graph in solutions:
        sol_graph = _remove_qns_from_graph(sol_graph, remove_qns_list)
        found_graph = _check_equal_ignoring_qns(
            sol_graph, filtered_solutions, ignore_qns_list
        )
        if found_graph is None:
            filtered_solutions.append(sol_graph)
        else:
            # check if found solution also has the prefactors
            # if not overwrite them
            remove_counter += 1

    logging.info(f"removed {remove_counter} solutions")
    return filtered_solutions


def _remove_qns_from_graph(  # pylint: disable=too-many-branches
    graph: StateTransitionGraph[ParticleWithSpin],
    qn_list: Set[Type[NodeQuantumNumber]],
) -> StateTransitionGraph[ParticleWithSpin]:
    for node_props in graph.node_props.values():
        for int_qn in qn_list:
            if int_qn in node_props:
                del node_props[int_qn]

    return graph


def _check_equal_ignoring_qns(
    ref_graph: StateTransitionGraph,
    solutions: List[StateTransitionGraph],
    ignored_qn_list: Set[Type[NodeQuantumNumber]],
) -> Optional[StateTransitionGraph]:
    """Define equal operator for the graphs ignoring certain quantum numbers."""
    if not isinstance(ref_graph, StateTransitionGraph):
        raise TypeError(
            "Reference graph has to be of type StateTransitionGraph"
        )
    found_graph = None
    old_comparator = ref_graph.graph_node_properties_comparator
    ref_graph.graph_node_properties_comparator = (
        CompareGraphNodePropertiesFunctor(ignored_qn_list)
    )
    for graph in solutions:
        if isinstance(graph, StateTransitionGraph):
            if ref_graph == graph:
                found_graph = graph
                break
    ref_graph.graph_node_properties_comparator = old_comparator
    return found_graph


def filter_graphs(
    graphs: List[StateTransitionGraph], filters: List[Callable]
) -> List[StateTransitionGraph]:
    r"""Implement filtering of a list of `.StateTransitionGraph` 's.

    This function can be used to select a subset of
    `.StateTransitionGraph` 's from a list. Only the graphs passing
    all supplied filters will be returned.

    Note:
        For the more advanced user, lambda functions can be used as filters.

    Args:
        graphs ([`.StateTransitionGraph`]): list of graphs to be
            filtered
        filters (list): list of functions, which take a single
            `.StateTransitionGraph` as an argument
    Returns:
        [`.StateTransitionGraph`]: filtered list of graphs

    Example:
        Selecting only the solutions, in which the :math:`\rho` decays via
        p-wave:

        >>> my_filter = require_interaction_property(
                'rho', InteractionQuantumNumberNames.L,
                create_spin_domain([1], True))
        >>> filtered_solutions = filter_graphs(solutions, [my_filter])
    """
    filtered_graphs = graphs
    for filter_ in filters:
        if not filtered_graphs:
            break
        filtered_graphs = list(filter(filter_, filtered_graphs))
    return filtered_graphs


def require_interaction_property(
    ingoing_particle_name: str,
    interaction_qn: Type[NodeQuantumNumber],
    allowed_values: List,
) -> Callable[[StateTransitionGraph], bool]:
    """Filter function.

    Closure, which can be used as a filter function in :func:`.filter_graphs`.

    It selects graphs based on a requirement on the property of specific
    interaction nodes.

    Args:
        ingoing_particle_name (str): name of particle, used to find nodes which
            have a particle with this name as "ingoing"
        interaction_qn (:class:`.NodeQuantumNumber`):
            interaction quantum number
        allowed_values (list): list of allowed values, that the interaction
            quantum number may take

    Return:
        Callable[Any, bool]:
            - *True* if the graph has nodes with an ingoing particle of the
              given name, and the graph fullfills the quantum number
              requirement
            - *False* otherwise
    """

    def check(graph: StateTransitionGraph) -> bool:
        node_ids = _find_node_ids_with_ingoing_particle_name(
            graph, ingoing_particle_name
        )
        if not node_ids:
            return False
        for i in node_ids:
            if (
                get_interaction_property(graph.node_props[i], interaction_qn)
                not in allowed_values
            ):
                return False
        return True

    return check


def _find_node_ids_with_ingoing_particle_name(
    graph: StateTransitionGraph, ingoing_particle_name: str
) -> List[int]:
    found_node_ids = []
    for node_id in graph.nodes:
        edge_ids = graph.get_edges_ingoing_to_node(node_id)
        for edge_id in edge_ids:
            edge_props = graph.edge_props[edge_id]
            edge_particle_name = edge_props[0].name
            if str(ingoing_particle_name) in str(edge_particle_name):
                found_node_ids.append(node_id)
                break
    return found_node_ids


def group_by_strength(
    graph_node_setting_pairs: List[Tuple[StateTransitionGraph, GraphSettings]]
) -> GraphSettingsGroups:
    graph_settings_groups: GraphSettingsGroups = {}
    for (instance, graph_settings) in graph_node_setting_pairs:
        strength = _calculate_strength(graph_settings.node_settings)
        if strength not in graph_settings_groups:
            graph_settings_groups[strength] = []
        graph_settings_groups[strength].append((instance, graph_settings))
    return graph_settings_groups


def _calculate_strength(
    node_interaction_settings: Dict[int, NodeSettings]
) -> float:
    strength = 1.0
    for int_setting in node_interaction_settings.values():
        strength *= int_setting.interaction_strength
    return strength
