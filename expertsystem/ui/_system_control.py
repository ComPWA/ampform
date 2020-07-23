"""Functions that steer operations of the `expertsystem`."""

# cspell:ignore vebar, vmubar, vtau, vtaubar

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from itertools import (
    permutations,
    product,
)

from expertsystem.state import particle
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
from expertsystem.topology.graph import (
    StateTransitionGraph,
    get_edges_ingoing_to_node,
    get_final_state_edges,
    get_initial_state_edges,
)


def _change_qn_domain(interaction_settings, qn_name, new_domain):
    if not isinstance(interaction_settings, InteractionNodeSettings):
        raise TypeError(
            "interaction_settings has to be of type InteractionNodeSettings"
        )
    interaction_settings.qn_domains.update({qn_name: new_domain})


def _remove_conservation_law(interaction_settings, cons_law):
    if not isinstance(interaction_settings, InteractionNodeSettings):
        raise TypeError(
            "interaction_settings has to be of type InteractionNodeSettings"
        )
    for i, law in enumerate(interaction_settings.conservation_laws):
        if law.__class__.__name__ == cons_law.__class__.__name__:
            del interaction_settings.conservation_laws[i]
            break


def filter_interaction_types(
    valid_determined_interaction_types, allowed_interaction_types
):
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
    def check(self, in_edge_props, out_edge_props, node_props):
        pass


class GammaCheck(_InteractionDeterminationFunctorInterface):
    """Conservation check for photons."""

    name_label = particle.Labels.Name.name

    def check(self, in_edge_props, out_edge_props, node_props):
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

    def check(self, in_edge_props, out_edge_props, node_props):
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
                        "ve",
                        "vebar",
                        "vmu",
                        "vmubar",
                        "vtau",
                        "vtaubar",
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
    results, remove_qns_list=None, ignore_qns_list=None
):
    if remove_qns_list is None:
        remove_qns_list = []
    if ignore_qns_list is None:
        ignore_qns_list = []
    logging.info("removing duplicate solutions...")
    logging.info(f"removing these qns from graphs: {remove_qns_list}")
    logging.info(f"ignoring qns in graph comparison: {ignore_qns_list}")
    filtered_results = {}
    solutions = []
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


def _remove_qns_from_graph(
    graph, qn_list
):  # pylint: disable=too-many-branches
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
                for qn_entry in graph_copy.edge_props[qns_label]:
                    if (
                        ParticlePropertyNames[qn_entry[type_label]]
                        is part_prop
                    ):
                        del props[qns_label][props[qns_label].index(qn_entry)]
                        break
    return graph_copy


def _check_equal_ignoring_qns(ref_graph, solutions, ignored_qn_list):
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


def filter_graphs(graphs, filters):
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
    ingoing_particle_name, interaction_qn, allowed_values
):
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
        bool:
            - *True* if the graph has nodes with an ingoing particle of the
              given name, and the graph fullfills the quantum number
              requirement
            - *False* otherwise
    """

    def check(graph):
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


def _find_node_ids_with_ingoing_particle_name(graph, ingoing_particle_name):
    name_label = particle.Labels.Name.name
    found_node_ids = []
    for node_id in graph.nodes:
        edge_ids = get_edges_ingoing_to_node(graph, node_id)
        for edge_id in edge_ids:
            edge_props = graph.edge_props[edge_id]
            if name_label in edge_props:
                edge_particle_name = edge_props[name_label]
                if str(ingoing_particle_name) in str(edge_particle_name):
                    found_node_ids.append(node_id)
                    break
    return found_node_ids


def analyse_solution_failure(violated_laws_per_node_and_graph):
    # try to find rules that are just always violated
    violated_laws = []
    scoresheet = {}

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


def create_interaction_setting_groups(graph_node_setting_pairs):
    graph_settings_groups = {}
    for (instance, node_settings) in graph_node_setting_pairs:
        setting_combinations = _create_setting_combinations(node_settings)
        for setting in setting_combinations:
            strength = _calculate_strength(setting)
            if strength not in graph_settings_groups:
                graph_settings_groups[strength] = []
            graph_settings_groups[strength].append((instance, setting))
    return graph_settings_groups


def _create_setting_combinations(node_settings):
    return [
        dict(zip(node_settings.keys(), x))
        for x in product(*node_settings.values())
    ]


def _calculate_strength(node_interaction_settings):
    strength = 1.0
    for int_setting in node_interaction_settings.values():
        strength *= int_setting.interaction_strength
    return strength


def match_external_edges(graphs):
    if not isinstance(graphs, list):
        raise TypeError("graphs argument is not of type list!")
    if not graphs:
        return
    ref_graph_id = 0
    _match_external_edge_ids(graphs, ref_graph_id, get_final_state_edges)
    _match_external_edge_ids(graphs, ref_graph_id, get_initial_state_edges)


def _match_external_edge_ids(  # pylint: disable=too-many-locals
    graphs, ref_graph_id, external_edge_getter_function
):
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


def _calculate_swappings(id_mapping):
    """Calculate edge id swappings.

    Its important to use an ordered dict as the swappings do not commute!
    """
    swappings = OrderedDict()
    for key, value in id_mapping.items():
        # go through existing swappings and use them
        newkey = key
        while newkey in swappings:
            newkey = swappings[newkey]
        if value != newkey:
            swappings[value] = newkey
    return swappings


def _create_edge_id_particle_mapping(graph, external_edge_getter_function):
    name_label = particle.Labels.Name.name
    return {
        i: graph.edge_props[i][name_label]
        for i in external_edge_getter_function(graph)
    }


def perform_external_edge_identical_particle_combinatorics(graph):
    """Create combinatorics clones of the `.StateTransitionGraph`.

    In case of identical particles in the initial or final state. Only
    identical particles, which do not enter or exit the same node allow for
    combinatorics!
    """
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph argument is not of type StateTransitionGraph!")
    temp_new_graphs = _external_edge_identical_particle_combinatorics(
        graph, get_final_state_edges
    )
    new_graphs = []
    for new_graph in temp_new_graphs:
        new_graphs.extend(
            _external_edge_identical_particle_combinatorics(
                new_graph, get_initial_state_edges
            )
        )
    return new_graphs


def _external_edge_identical_particle_combinatorics(
    graph, external_edge_getter_function
):
    # pylint: disable=too-many-locals
    new_graphs = [graph]
    edge_particle_mapping = _create_edge_id_particle_mapping(
        graph, external_edge_getter_function
    )
    identical_particle_groups = {}
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
