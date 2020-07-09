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
from multiprocessing import Pool
from os import path
from typing import Callable

from progress.bar import IncrementalBar

import expertsystem
from expertsystem.state import particle
from expertsystem.state.particle import (
    CompareGraphElementPropertiesFunctor,
    InteractionQuantumNumberNames,
    ParticlePropertyNames,
    StateQuantumNumberNames,
    get_interaction_property,
    get_particle_property,
    initialize_graph,
)
from expertsystem.state.propagation import (
    FullPropagator,
    InteractionNodeSettings,
    InteractionTypes,
)
from expertsystem.topology.graph import (
    InteractionNode,
    StateTransitionGraph,
    get_edges_ingoing_to_node,
    get_edges_outgoing_to_node,
    get_final_state_edges,
    get_initial_state_edges,
)
from expertsystem.topology.topology_builder import (
    SimpleStateTransitionTopologyBuilder,
)

from .default_settings import create_default_interaction_settings


_EXPERT_SYSTEM_PATH = path.dirname(path.realpath(expertsystem.__file__))


def change_qn_domain(interaction_settings, qn_name, new_domain):
    if not isinstance(interaction_settings, InteractionNodeSettings):
        raise TypeError(
            "interaction_settings has to be of type InteractionNodeSettings"
        )
    interaction_settings.qn_domains.update({qn_name: new_domain})


def remove_conservation_law(interaction_settings, cons_law):
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


class InteractionDeterminationFunctorInterface(ABC):
    """Interface for interaction determination."""

    @abstractmethod
    def check(self, in_edge_props, out_edge_props, node_props):
        pass


class GammaCheck(InteractionDeterminationFunctorInterface):
    """Conservation check for photons."""

    name_label = particle.Labels.Name.name

    def check(self, in_edge_props, out_edge_props, node_props):
        int_types = list(InteractionTypes)
        for edge_props in in_edge_props + out_edge_props:
            if "gamma" in edge_props[self.name_label]:
                int_types = [InteractionTypes.EM, InteractionTypes.Weak]
                break

        return int_types


class LeptonCheck(InteractionDeterminationFunctorInterface):
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
                sol_graph = remove_qns_from_graph(sol_graph, remove_qns_list)
                found_graph = check_equal_ignoring_qns(
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


def remove_qns_from_graph(graph, qn_list):  # pylint: disable=too-many-branches
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


def check_equal_ignoring_qns(ref_graph, solutions, ignored_qn_list):
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
        Selecting only the solutions, in which the :math:`\\rho` decays via
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


def create_setting_combinations(node_settings):
    return [
        dict(zip(node_settings.keys(), x))
        for x in product(*node_settings.values())
    ]


def calculate_strength(node_interaction_settings):
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
    match_external_edge_ids(graphs, ref_graph_id, get_final_state_edges)
    match_external_edge_ids(graphs, ref_graph_id, get_initial_state_edges)


def match_external_edge_ids(  # pylint: disable=too-many-locals
    graphs, ref_graph_id, external_edge_getter_function
):
    ref_graph = graphs[ref_graph_id]
    # create external edge to particle mapping
    ref_edge_id_particle_mapping = create_edge_id_particle_mapping(
        ref_graph, external_edge_getter_function
    )

    for graph in graphs[:ref_graph_id] + graphs[ref_graph_id + 1 :]:
        edge_id_particle_mapping = create_edge_id_particle_mapping(
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
        swappings = calculate_swappings(edge_ids_mapping)
        for edge_id1, edge_id2 in swappings.items():
            graph.swap_edges(edge_id1, edge_id2)


def calculate_swappings(id_mapping):
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


def create_edge_id_particle_mapping(graph, external_edge_getter_function):
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
    temp_new_graphs = external_edge_identical_particle_combinatorics(
        graph, get_final_state_edges
    )
    new_graphs = []
    for new_graph in temp_new_graphs:
        new_graphs.extend(
            external_edge_identical_particle_combinatorics(
                new_graph, get_initial_state_edges
            )
        )
    return new_graphs


def external_edge_identical_particle_combinatorics(
    graph, external_edge_getter_function
):
    # pylint: disable=too-many-locals
    new_graphs = [graph]
    edge_particle_mapping = create_edge_id_particle_mapping(
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
                swappings = calculate_swappings(combination)
                for edge_id1, edge_id2 in swappings.items():
                    graph_copy.swap_edges(edge_id1, edge_id2)
                temp_new_graphs.append(graph_copy)
        new_graphs = temp_new_graphs
    return new_graphs


class StateTransitionManager:  # pylint: disable=too-many-instance-attributes
    """Main handler for decay topologies."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        initial_state,
        final_state,
        allowed_intermediate_particles=None,
        interaction_type_settings=None,
        formalism_type="helicity",
        topology_building="isobar",
        number_of_threads=4,
        propagation_mode="fast",
    ):
        if allowed_intermediate_particles is None:
            allowed_intermediate_particles = []
        if interaction_type_settings is None:
            interaction_type_settings = {}
        allowed_formalism_types = [
            "helicity",
            "canonical-helicity",
            "canonical",
        ]
        if formalism_type not in allowed_formalism_types:
            raise NotImplementedError(
                f"Formalism type {formalism_type} not implemented."
                f" Use {allowed_formalism_types} instead."
            )
        self.number_of_threads = number_of_threads
        self.propagation_mode = propagation_mode
        self.initial_state = initial_state
        self.final_state = final_state
        self.interaction_type_settings = interaction_type_settings
        if not self.interaction_type_settings:
            self.interaction_type_settings = create_default_interaction_settings(
                formalism_type
            )
        self.interaction_determinators = [LeptonCheck(), GammaCheck()]
        self.allowed_intermediate_particles = allowed_intermediate_particles
        self.final_state_groupings = []
        self.allowed_interaction_types = [
            InteractionTypes.Strong,
            InteractionTypes.EM,
            InteractionTypes.Weak,
        ]
        self.filter_remove_qns = []
        self.filter_ignore_qns = []
        if formalism_type == "helicity":
            self.filter_remove_qns = [
                InteractionQuantumNumberNames.S,
                InteractionQuantumNumberNames.L,
            ]
        if "helicity" in formalism_type:
            self.filter_ignore_qns = [
                InteractionQuantumNumberNames.ParityPrefactor
            ]
        int_nodes = []
        if topology_building == "isobar":
            if len(initial_state) == 1:
                int_nodes.append(InteractionNode("TwoBodyDecay", 1, 2))
        else:
            int_nodes.append(
                InteractionNode(
                    "NBodyScattering", len(initial_state), len(final_state)
                )
            )
            # turn of mass conservation, in case more than one initial state
            # particle is present
            if len(initial_state) > 1:
                self.interaction_type_settings = create_default_interaction_settings(
                    formalism_type, False
                )
        self.topology_builder = SimpleStateTransitionTopologyBuilder(int_nodes)

        load_default_particle_list()

    def set_topology_builder(self, topology_builder):
        self.topology_builder = topology_builder

    def add_final_state_grouping(self, fs_group):
        if not isinstance(fs_group, list):
            raise ValueError(
                "The final state grouping has to be of type list."
            )
        if len(fs_group) > 0:
            if not isinstance(fs_group[0], list):
                fs_group = [fs_group]
            self.final_state_groupings.append(fs_group)

    def set_allowed_interaction_types(self, allowed_interaction_types):
        # verify order
        for allowed_types in allowed_interaction_types:
            if not isinstance(allowed_types, InteractionTypes):
                raise TypeError(
                    "allowed interaction types must be of type"
                    "[InteractionTypes]"
                )
            if allowed_types not in self.interaction_type_settings:
                logging.info(self.interaction_type_settings.keys())
                raise ValueError(
                    f"interaction {allowed_types} not found in settings"
                )
        self.allowed_interaction_types = allowed_interaction_types

    def prepare_graphs(self):
        topology_graphs = self.build_topologies()
        init_graphs = self.create_seed_graphs(topology_graphs)
        graph_node_setting_pairs = self.determine_node_settings(init_graphs)
        # create groups of settings ordered by "probability"
        graph_settings_groups = self.create_interaction_setting_groups(
            graph_node_setting_pairs
        )
        return graph_settings_groups

    def build_topologies(self):
        all_graphs = self.topology_builder.build_graphs(
            len(self.initial_state), len(self.final_state)
        )
        logging.info(f"number of topology graphs: {len(all_graphs)}")
        return all_graphs

    def create_seed_graphs(self, topology_graphs):
        # initialize the graph edges (initial and final state)
        init_graphs = []
        for topology_graph in topology_graphs:
            topology_graph.set_graph_element_properties_comparator(
                CompareGraphElementPropertiesFunctor()
            )
            init_graphs.extend(
                initialize_graph(
                    topology_graph,
                    self.initial_state,
                    self.final_state,
                    self.final_state_groupings,
                )
            )

        logging.info(f"initialized {len(init_graphs)} graphs!")
        return init_graphs

    def determine_node_settings(self, graphs):
        # pylint: disable=too-many-locals
        graph_node_setting_pairs = []
        for graph in graphs:
            final_state_edges = get_final_state_edges(graph)
            initial_state_edges = get_initial_state_edges(graph)
            node_settings = {}
            for node_id in graph.nodes:
                node_int_types = []
                out_edge_ids = get_edges_outgoing_to_node(graph, node_id)
                in_edge_ids = get_edges_outgoing_to_node(graph, node_id)
                in_edge_props = [
                    graph.edge_props[edge_id]
                    for edge_id in [
                        x for x in in_edge_ids if x in initial_state_edges
                    ]
                ]
                out_edge_props = [
                    graph.edge_props[edge_id]
                    for edge_id in [
                        x for x in out_edge_ids if x in final_state_edges
                    ]
                ]
                node_props = {}
                if node_id in graph.node_props:
                    node_props = graph.node_props[node_id]
                for int_det in self.interaction_determinators:
                    determined_interactions = int_det.check(
                        in_edge_props, out_edge_props, node_props
                    )
                    if node_int_types:
                        node_int_types = list(
                            set(determined_interactions) & set(node_int_types)
                        )
                    else:
                        node_int_types = determined_interactions
                node_int_types = filter_interaction_types(
                    node_int_types, self.allowed_interaction_types
                )
                logging.debug(
                    "using %s interaction order for node: %s",
                    str(node_int_types),
                    str(node_id),
                )
                node_settings[node_id] = [
                    deepcopy(self.interaction_type_settings[x])
                    for x in node_int_types
                ]
            graph_node_setting_pairs.append((graph, node_settings))
        return graph_node_setting_pairs

    @staticmethod
    def create_interaction_setting_groups(graph_node_setting_pairs):
        graph_settings_groups = {}
        for (graph, node_settings) in graph_node_setting_pairs:
            setting_combinations = create_setting_combinations(node_settings)
            for setting in setting_combinations:
                strength = calculate_strength(setting)
                if strength not in graph_settings_groups:
                    graph_settings_groups[strength] = []
                graph_settings_groups[strength].append((graph, setting))
        return graph_settings_groups

    def find_solutions(
        self, graph_setting_groups
    ):  # pylint: disable=too-many-locals
        """Check for solutions for a specific set of interaction settings."""
        results = {}
        logging.info(
            "Number of interaction settings groups being processed: %d",
            len(graph_setting_groups),
        )
        for strength, graph_setting_group in sorted(
            graph_setting_groups.items(), reverse=True
        ):
            logging.info(
                "processing interaction settings group with "
                f"strength {strength}",
            )
            logging.info(f"{graph_setting_group} entries in this group")
            logging.info(f"running with {self.number_of_threads} threads...")

            temp_results = []
            progress_bar = IncrementalBar(
                "Propagating quantum numbers...", max=len(graph_setting_group)
            )
            progress_bar.update()
            if self.number_of_threads > 1:
                with Pool(self.number_of_threads) as pool:
                    for result in pool.imap_unordered(
                        self.propagate_quantum_numbers, graph_setting_group, 1
                    ):
                        temp_results.append(result)
                        progress_bar.next()
            else:
                for graph_setting_pair in graph_setting_group:
                    temp_results.append(
                        self.propagate_quantum_numbers(graph_setting_pair)
                    )
                    progress_bar.next()
            progress_bar.finish()
            logging.info("Finished!")
            if strength not in results:
                results[strength] = []
            results[strength].extend(temp_results)

        for key, value in results.items():
            logging.info(
                f"number of solutions for strength ({key}) "
                f"after qn propagation: {sum([len(x[0]) for x in value])}",
            )

        # remove duplicate solutions, which only differ in the interaction qn S
        results = remove_duplicate_solutions(
            results, self.filter_remove_qns, self.filter_ignore_qns
        )

        node_non_satisfied_rules = []
        solutions = []
        for result in results.values():
            for (tempsolutions, non_satisfied_laws) in result:
                solutions.extend(tempsolutions)
                node_non_satisfied_rules.append(non_satisfied_laws)
        logging.info(f"total number of found solutions: {len(solutions)}")
        violated_laws = []
        if len(solutions) == 0:
            violated_laws = analyse_solution_failure(node_non_satisfied_rules)
            logging.info(f"violated rules: {violated_laws}")

        # finally perform combinatorics of identical external edges
        # (initial or final state edges) and prepare graphs for
        # amplitude generation
        match_external_edges(solutions)
        final_solutions = []
        for sol in solutions:
            final_solutions.extend(
                perform_external_edge_identical_particle_combinatorics(sol)
            )

        return (final_solutions, violated_laws)

    def propagate_quantum_numbers(self, state_graph_node_settings_pair):
        propagator = self.initialize_qn_propagator(
            state_graph_node_settings_pair[0],
            state_graph_node_settings_pair[1],
        )
        solutions = propagator.find_solutions()
        return (solutions, propagator.get_non_satisfied_conservation_laws())

    def initialize_qn_propagator(self, state_graph, node_settings):
        propagator = FullPropagator(state_graph, self.propagation_mode)
        for node_id, interaction_settings in node_settings.items():
            propagator.assign_settings_to_node(node_id, interaction_settings)
        # specify set of particles which are allowed to be intermediate
        # particles. If list is empty, then all particles in the default
        # particle list are used
        propagator.set_allowed_intermediate_particles(
            self.allowed_intermediate_particles
        )

        return propagator


def load_default_particle_list(
    method: Callable = particle.load_particle_list_from_xml,
) -> None:
    """Load the default particle list that comes with the expertsystem."""
    if len(particle.DATABASE) != 0:
        return
    if method is particle.load_particle_list_from_xml:
        particle_list_file = "particle_list.xml"
    elif method is particle.load_particle_list_from_yaml:
        particle_list_file = "particle_list.yml"
    else:
        raise NotImplementedError
    particle_list_path = path.join(_EXPERT_SYSTEM_PATH, particle_list_file)
    if not path.exists(particle_list_path):
        raise FileNotFoundError(
            "\n  Failed to load {particle_list_file}!"
            "\n  Please contact the developers: https://github.com/ComPWA"
        )
    method(particle_list_path)
    logging.info(
        f"Loaded {len(particle.DATABASE)} particles from {particle_list_file}!"
    )
