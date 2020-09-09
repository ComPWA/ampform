"""Functions that steer operations of the `expertsystem`."""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from expertsystem.data import Spin
from expertsystem.state import particle
from expertsystem.state.conservation_rules import Rule
from expertsystem.state.particle import (
    CompareGraphElementPropertiesFunctor,
    InteractionQuantumNumberNames,
    ParticlePropertyNames,
    StateQuantumNumberNames,
    get_interaction_property,
    get_particle_property,
)
from expertsystem.state.propagation import (
    InteractionNodeSettings,
    InteractionTypes,
)
from expertsystem.topology import StateTransitionGraph


Strength = float

GraphSettings = Tuple[StateTransitionGraph, Dict[int, InteractionNodeSettings]]
GraphSettingsGroups = Dict[Strength, List[GraphSettings]]
NodeSettings = Dict[int, List[InteractionNodeSettings]]

ViolatedLaws = Dict[int, List[Rule]]
SolutionMapping = Dict[
    Strength, List[Tuple[List[StateTransitionGraph], ViolatedLaws]],
]


def _change_qn_domain(
    interaction_settings: InteractionNodeSettings,
    qn_name: InteractionQuantumNumberNames,
    new_domain: List[Spin],
) -> None:
    if not isinstance(interaction_settings, InteractionNodeSettings):
        raise TypeError(
            "interaction_settings has to be of type InteractionNodeSettings"
        )
    interaction_settings.qn_domains.update({qn_name: new_domain})


def _remove_conservation_law(
    interaction_settings: InteractionNodeSettings, cons_law: Rule
) -> None:
    if not isinstance(interaction_settings, InteractionNodeSettings):
        raise TypeError(
            "interaction_settings has to be of type InteractionNodeSettings"
        )
    for i, law in enumerate(interaction_settings.conservation_laws):
        if str(law) == str(cons_law):
            del interaction_settings.conservation_laws[i]
            break


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
        in_edge_props: List[dict],
        out_edge_props: List[dict],
        node_props: dict,
    ) -> List[InteractionTypes]:
        pass


class GammaCheck(_InteractionDeterminationFunctorInterface):
    """Conservation check for photons."""

    name_label = particle.Labels.Name.name

    def check(
        self,
        in_edge_props: List[dict],
        out_edge_props: List[dict],
        node_props: dict,
    ) -> List[InteractionTypes]:
        int_types = list(InteractionTypes)
        for edge_props in in_edge_props + out_edge_props:
            if "gamma" in edge_props[self.name_label]:
                int_types = [InteractionTypes.EM, InteractionTypes.Weak]
                break
        return int_types


class LeptonCheck(_InteractionDeterminationFunctorInterface):
    """Conservation check lepton numbers."""

    lepton_flavor_labels = [
        StateQuantumNumberNames.ElectronLN,
        StateQuantumNumberNames.MuonLN,
        StateQuantumNumberNames.TauLN,
    ]
    name_label = particle.Labels.Name.name
    qns_label = particle.Labels.QuantumNumber.name

    def check(
        self,
        in_edge_props: List[dict],
        out_edge_props: List[dict],
        node_props: dict,
    ) -> List[InteractionTypes]:
        node_interaction_types = list(InteractionTypes)
        for edge_props in in_edge_props + out_edge_props:
            if sum(
                [
                    get_particle_property(edge_props, x)
                    for x in self.lepton_flavor_labels
                    if get_particle_property(edge_props, x) is not None
                ]
            ):
                if [
                    x
                    for x in [
                        "nu(e)",
                        "nu(e)~",
                        "nu(mu)",
                        "nu(mu)~",
                        "nu(tau)",
                        "nu(tau)~",
                    ]
                    if x == edge_props[self.name_label]
                ]:
                    node_interaction_types = [InteractionTypes.Weak]
                    break
                if edge_props[self.qns_label] != 0:
                    node_interaction_types = [
                        InteractionTypes.EM,
                        InteractionTypes.Weak,
                    ]
        return node_interaction_types


def remove_duplicate_solutions(
    results: SolutionMapping,
    remove_qns_list: Optional[Any] = None,
    ignore_qns_list: Optional[Any] = None,
) -> SolutionMapping:
    if remove_qns_list is None:
        remove_qns_list = []
    if ignore_qns_list is None:
        ignore_qns_list = []
    logging.info("removing duplicate solutions...")
    logging.info(f"removing these qns from graphs: {remove_qns_list}")
    logging.info(f"ignoring qns in graph comparison: {ignore_qns_list}")
    filtered_results: SolutionMapping = {}
    solutions: List[StateTransitionGraph] = list()
    remove_counter = 0
    for strength, group_results in results.items():
        for (sol_graphs, rule_violations) in group_results:
            temp_graphs = []
            for sol_graph in sol_graphs:
                sol_graph = _remove_qns_from_graph(sol_graph, remove_qns_list)
                found_graph = _check_equal_ignoring_qns(
                    sol_graph, solutions, ignore_qns_list
                )
                if found_graph is None:
                    solutions.append(sol_graph)
                    temp_graphs.append(sol_graph)
                else:
                    # check if found solution also has the prefactors
                    # if not overwrite them
                    remove_counter += 1

            if strength not in filtered_results:
                filtered_results[strength] = []
            filtered_results[strength].append((temp_graphs, rule_violations))
    logging.info(f"removed {remove_counter} solutions")
    return filtered_results


def _remove_qns_from_graph(  # pylint: disable=too-many-branches
    graph: StateTransitionGraph,
    qn_list: List[
        Union[
            InteractionQuantumNumberNames,
            StateQuantumNumberNames,
            ParticlePropertyNames,
        ]
    ],
) -> StateTransitionGraph:
    qns_label = particle.Labels.QuantumNumber.name
    type_label = particle.Labels.Type.name

    int_qns = [
        x for x in qn_list if isinstance(x, InteractionQuantumNumberNames)
    ]
    state_qns = [x for x in qn_list if isinstance(x, StateQuantumNumberNames)]
    part_props = [x for x in qn_list if isinstance(x, ParticlePropertyNames)]

    graph_copy = deepcopy(graph)

    for int_qn in int_qns:
        for props in graph_copy.node_props.values():
            if qns_label in props:
                for qn_entry in props[qns_label]:
                    if (
                        InteractionQuantumNumberNames[qn_entry[type_label]]
                        is int_qn
                    ):
                        del props[qns_label][props[qns_label].index(qn_entry)]
                        break

    for state_qn in state_qns:
        for props in graph_copy.edge_props.values():
            if qns_label in props:
                for qn_entry in props[qns_label]:
                    if (
                        StateQuantumNumberNames[qn_entry[type_label]]
                        is state_qn
                    ):
                        del props[qns_label][props[qns_label].index(qn_entry)]
                        break

    for part_prop in part_props:
        for props in graph_copy.edge_props.values():
            if qns_label in props:
                for qn_entry in graph_copy.edge_props[qns_label]:  # type: ignore
                    if (
                        ParticlePropertyNames[qn_entry[type_label]]
                        is part_prop
                    ):
                        del props[qns_label][props[qns_label].index(qn_entry)]
                        break
    return graph_copy


def _check_equal_ignoring_qns(
    ref_graph: StateTransitionGraph,
    solutions: List[StateTransitionGraph],
    ignored_qn_list: List[
        Union[StateQuantumNumberNames, InteractionQuantumNumberNames]
    ],
) -> Optional[StateTransitionGraph]:
    """Define equal operator for the graphs ignoring certain quantum numbers."""
    if not isinstance(ref_graph, StateTransitionGraph):
        raise TypeError(
            "Reference graph has to be of type StateTransitionGraph"
        )
    found_graph = None
    old_comparator = ref_graph.graph_element_properties_comparator
    ref_graph.set_graph_element_properties_comparator(
        CompareGraphElementPropertiesFunctor(ignored_qn_list)
    )
    for graph in solutions:
        if isinstance(graph, StateTransitionGraph):
            if ref_graph == graph:
                found_graph = graph
                break
    ref_graph.set_graph_element_properties_comparator(old_comparator)
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
    interaction_qn: InteractionQuantumNumberNames,
    allowed_values: List,
) -> Callable[[StateTransitionGraph], bool]:
    """Filter function.

    Closure, which can be used as a filter function in :func:`.filter_graphs`.

    It selects graphs based on a requirement on the property of specific
    interaction nodes.

    Args:
        ingoing_particle_name (str): name of particle, used to find nodes which
            have a particle with this name as "ingoing"
        interaction_qn (:class:`.InteractionQuantumNumberNames`):
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
    name_label = particle.Labels.Name.name
    found_node_ids = []
    for node_id in graph.nodes:
        edge_ids = graph.get_edges_ingoing_to_node(node_id)
        for edge_id in edge_ids:
            edge_props = graph.edge_props[edge_id]
            if name_label in edge_props:
                edge_particle_name = edge_props[name_label]
                if str(ingoing_particle_name) in str(edge_particle_name):
                    found_node_ids.append(node_id)
                    break
    return found_node_ids


def analyse_solution_failure(
    violated_laws_per_node_and_graph: List[ViolatedLaws],
) -> List[str]:
    # try to find rules that are just always violated
    violated_laws: List[str] = []
    scoresheet: Dict[str, int] = {}

    for violated_laws_per_node in violated_laws_per_node_and_graph:
        temp_violated_laws = set()
        for laws in violated_laws_per_node.values():
            for law in laws:
                temp_violated_laws.add(str(law))
        for law_name in temp_violated_laws:
            if law_name not in scoresheet:
                scoresheet[law_name] = 0
            scoresheet[law_name] += 1

    for rule_name, violation_count in scoresheet.items():
        if violation_count == len(violated_laws_per_node_and_graph):
            violated_laws.append(rule_name)

    logging.debug(
        "no solutions could be found, because the following rules are violated:\n%r",
        violated_laws,
    )

    return violated_laws


def create_interaction_setting_groups(
    graph_node_setting_pairs: List[Tuple[StateTransitionGraph, NodeSettings]]
) -> GraphSettingsGroups:
    graph_settings_groups: GraphSettingsGroups = {}
    for (instance, node_settings) in graph_node_setting_pairs:
        setting_combinations = _create_setting_combinations(node_settings)
        for setting in setting_combinations:
            strength = _calculate_strength(setting)
            if strength not in graph_settings_groups:
                graph_settings_groups[strength] = []
            graph_settings_groups[strength].append((instance, setting))
    return graph_settings_groups


def _create_setting_combinations(
    node_settings: NodeSettings,
) -> List[Dict[int, InteractionNodeSettings]]:
    return [
        dict(zip(node_settings.keys(), x))
        for x in product(*node_settings.values())  # type: ignore
    ]


def _calculate_strength(
    node_interaction_settings: Dict[int, InteractionNodeSettings]
) -> float:
    strength = 1.0
    for int_setting in node_interaction_settings.values():
        strength *= int_setting.interaction_strength
    return strength
