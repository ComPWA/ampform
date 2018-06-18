import logging
from copy import deepcopy
from itertools import product, permutations
from abc import ABC, abstractmethod
from multiprocessing import Pool

from expertsystem.topology.graph import (StateTransitionGraph,
                                         InteractionNode,
                                         get_edges_outgoing_to_node,
                                         get_final_state_edges,
                                         get_initial_state_edges,
                                         get_originating_final_state_edges)
from expertsystem.topology.topologybuilder import (
    SimpleStateTransitionTopologyBuilder)

from expertsystem.state.particle import (
    load_particle_list_from_xml, particle_list, initialize_graph,
    get_particle_property, XMLLabelConstants, get_xml_label,
    StateQuantumNumberNames, InteractionQuantumNumberNames,
    ParticlePropertyNames, compare_graph_element_properties)

from expertsystem.state.propagation import (
    FullPropagator, InteractionTypes, InteractionNodeSettings)

from expertsystem.ui.default_settings import (
    create_default_interaction_settings
)


def change_qn_domain(interaction_settings, qn_name, new_domain):
    if not isinstance(interaction_settings, InteractionNodeSettings):
        raise TypeError(
            "interaction_settings has to be of type InteractionNodeSettings")
    interaction_settings.qn_domains.update({qn_name: new_domain})


def filter_interaction_types(interaction_types, allowed_interaction_types):
    current_lowest_type = InteractionTypes.Strong.value
    for int_type in interaction_types:
        if int_type.value > current_lowest_type:
            current_lowest_type = int_type.value

    return [x for x in InteractionTypes if x.value >= current_lowest_type
            and x in allowed_interaction_types]


class InteractionDeterminationFunctorInterface(ABC):
    @abstractmethod
    def check(self, in_edge_props, out_edge_props, node_props):
        pass


class GammaCheck(InteractionDeterminationFunctorInterface):
    name_label = get_xml_label(XMLLabelConstants.Name)

    def check(self, in_edge_props, out_edge_props, node_props):
        int_type = InteractionTypes.Undefined
        for edge_props in in_edge_props + out_edge_props:
            if ('gamma' in edge_props[self.name_label]):
                int_type = InteractionTypes.EM
                break

        return int_type


class LeptonCheck(InteractionDeterminationFunctorInterface):
    lepton_flavour_labels = [
        StateQuantumNumberNames.ElectronLN,
        StateQuantumNumberNames.MuonLN,
        StateQuantumNumberNames.TauLN
    ]
    name_label = get_xml_label(XMLLabelConstants.Name)
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)

    def check(self, in_edge_props, out_edge_props, node_props):
        node_interaction_type = InteractionTypes.Undefined
        for edge_props in in_edge_props + out_edge_props:
            if sum([get_particle_property(edge_props, x)
                    for x in self.lepton_flavour_labels
                    if get_particle_property(edge_props, x) is not None]):
                if [x for x in
                        ['ve', 'vebar', 'vmu', 'vmubar', 'vtau', 'vtaubar']
                        if x == edge_props[self.name_label]]:
                    node_interaction_type = InteractionTypes.Weak
                    break
                if edge_props[self.qns_label] != 0:
                    node_interaction_type = InteractionTypes.EM
        return node_interaction_type


def convert_fs_names_to_edge_ids(graph, list_of_particle_name_lists):
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    name_label = get_xml_label(XMLLabelConstants.Name)
    fsp_name_id_map = {}
    for i in get_final_state_edges(graph):
        if graph.edge_props[i][name_label] not in fsp_name_id_map:
            fsp_name_id_map[graph.edge_props[i][name_label]] = []
        fsp_name_id_map[graph.edge_props[i][name_label]].append(i)

    fsp_name_id_maps = [{}]
    for k, v in fsp_name_id_map.items():
        all_permutations = permutations(v)
        temp_fsp_name_id_maps = []
        for current_fsp_permutation in product(fsp_name_id_maps,
                                               all_permutations):
            temp_fsp_name_id_map = deepcopy(current_fsp_permutation[0])
            temp_fsp_name_id_map[k] = list(current_fsp_permutation[1])
            temp_fsp_name_id_maps.append(temp_fsp_name_id_map)
        fsp_name_id_maps = temp_fsp_name_id_maps

    edge_lists_combinations = []
    for fsp_name_id_map in fsp_name_id_maps:
        current_edge_list_combination = []
        current_fsp_name_id_map = deepcopy(fsp_name_id_map)
        for particle_name_list in list_of_particle_name_lists:
            current_fs_group = set()
            for particle_name in particle_name_list:
                if (particle_name not in current_fsp_name_id_map
                        or len(current_fsp_name_id_map[particle_name]) == 0):
                    raise ValueError(
                        "Too many final state particles with name "
                        + particle_name + " were requested.\nThe existing "
                        + "final state particles are:\n"
                        + str(fsp_name_id_map))
                possible_edge_ids = current_fsp_name_id_map[particle_name]
                current_fs_group.add(possible_edge_ids[0])
                del possible_edge_ids[0]
            current_edge_list_combination.append(current_fs_group)
        edge_lists_combinations.append(
            current_edge_list_combination)
    return edge_lists_combinations


def remove_qns_from_graph(graph, qn_list):
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
    type_label = get_xml_label(XMLLabelConstants.Type)

    int_qns = [x for x in qn_list if isinstance(
        x, InteractionQuantumNumberNames)]
    state_qns = [x for x in qn_list if isinstance(
        x, StateQuantumNumberNames)]
    part_props = [x for x in qn_list if isinstance(
        x, ParticlePropertyNames)]

    graph_copy = deepcopy(graph)

    for int_qn in int_qns:
        for props in graph_copy.node_props.values():
            if qns_label in props:
                for qn_entry in props[qns_label]:
                    if (InteractionQuantumNumberNames[qn_entry[type_label]]
                            is int_qn):
                        del props[qns_label][props[qns_label].index(qn_entry)]
                        break

    for state_qn in state_qns:
        for props in graph_copy.edge_props.values():
            if qns_label in props:
                for qn_entry in props[qns_label]:
                    if (StateQuantumNumberNames[qn_entry[type_label]]
                            is state_qn):
                        del props[qns_label][props[qns_label].index(qn_entry)]
                        break

    for part_prop in part_props:
        for props in graph_copy.edge_props.values():
            if qns_label in props:
                for qn_entry in graph_copy.edge_props[qns_label]:
                    if (ParticlePropertyNames[qn_entry[type_label]]
                            is part_prop):
                        del props[qns_label][props[qns_label].index(qn_entry)]
                        break

    return graph_copy


def check_equal_ignoring_qns(ref_graph, solutions, ignored_qn_list):
    """
    defines the equal operator for the graphs ignoring certain quantum numbers
    """
    if not isinstance(ref_graph, StateTransitionGraph):
        raise TypeError(
            "Reference graph has to be of type StateTransitionGraph")

    ref_graph_copy = remove_qns_from_graph(ref_graph, ignored_qn_list)
    found_graph = None

    for graph in solutions:
        if isinstance(graph, StateTransitionGraph):
            # first copy prop dicts and remove ignored qns
            graph_copy = remove_qns_from_graph(graph, ignored_qn_list)
            if ref_graph_copy == graph_copy:
                found_graph = graph
                break

    return found_graph


def filter_solutions(results, remove_qns_list, ingore_qns_list):
    logging.info("filtering solutions...")
    logging.info("removing these qns from graphs: " + str(remove_qns_list))
    logging.info("ignoring qns in graph comparison: " + str(ingore_qns_list))
    filtered_results = {}
    solutions = []
    remove_counter = 0
    for strength, group_results in results.items():
        for (sol_graphs, rule_violations) in group_results:
            temp_graphs = []
            for sol_graph in sol_graphs:
                sol_graph = remove_qns_from_graph(sol_graph, remove_qns_list)

                found_graph = check_equal_ignoring_qns(sol_graph, solutions,
                                                       ingore_qns_list)
                if found_graph is None:
                    solutions.append(sol_graph)
                    temp_graphs.append(sol_graph)
                else:
                    # check if found solution also has the prefactors
                    # if not overwrite them
                    remove_counter += 1

            if strength not in filtered_results:
                filtered_results[strength] = []
            filtered_results[strength].append(
                (temp_graphs, rule_violations))
    logging.info("removed " + str(remove_counter) + " solutions")
    return filtered_results


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

    logging.debug("no solutions could be found, because the following " +
                  "rules are violated:")
    logging.debug(violated_laws)

    return violated_laws


def create_setting_combinations(node_settings):
    return [dict(zip(node_settings.keys(), x))
            for x in product(*node_settings.values())]


def calculate_strength(node_interaction_settings):
    strength = 1.0
    for int_setting in node_interaction_settings.values():
        strength *= int_setting.interaction_strength
    return strength


class StateTransitionManager():
    def __init__(self, initial_state, final_state,
                 allowed_intermediate_particles=[],
                 interaction_type_settings={},
                 formalism_type='helicity',
                 topology_building='isobar',
                 number_of_threads=4):
        self.number_of_threads = number_of_threads
        self.initial_state = initial_state
        self.final_state = final_state
        self.interaction_type_settings = interaction_type_settings
        if not self.interaction_type_settings:
            self.interaction_type_settings = create_default_interaction_settings(
                formalism_type)
        self.interaction_determinators = [LeptonCheck(), GammaCheck()]
        self.allowed_intermediate_particles = allowed_intermediate_particles
        self.final_state_groupings = []
        self.allowed_interaction_types = [InteractionTypes.Strong,
                                          InteractionTypes.EM,
                                          InteractionTypes.Weak]
        self.filter_remove_qns = []
        self.filter_ignore_qns = []
        if formalism_type == 'helicity':
            self.filter_remove_qns = [InteractionQuantumNumberNames.S]
            self.filter_ignore_qns = [
                InteractionQuantumNumberNames.ParityPrefactor]
        int_nodes = []
        if topology_building == 'isobar':
            if len(initial_state) == 1:
                int_nodes.append(InteractionNode("TwoBodyDecay", 1, 2))
        else:
            int_nodes.append(InteractionNode(
                "NBodyScattering",
                len(initial_state),
                len(final_state)))
            # turn of mass conservation, in case more than one initial state
            # particle is present
            if len(initial_state) > 1:
                self.interaction_type_settings = create_default_interaction_settings(
                    formalism_type,
                    False)
        self.topology_builder = SimpleStateTransitionTopologyBuilder(
            int_nodes)

        # load default particles from database/file
        if len(particle_list) == 0:
            load_particle_list_from_xml('../particle_list.xml')
            # print(particle_list)
            logging.info("loaded " + str(len(particle_list))
                         + " particles from xml file!")

    def set_topology_builder(self, topology_builder):
        self.topology_builder = topology_builder

    def add_final_state_grouping(self, fs_group):
        if not isinstance(fs_group, list):
            raise ValueError(
                "The final state grouping has to be of type list.")
        if len(fs_group) > 0:
            if not isinstance(fs_group[0], list):
                fs_group = [fs_group]
            self.final_state_groupings.append(fs_group)

    def set_allowed_interaction_types(self, allowed_interaction_types):
        # verify order
        for x in allowed_interaction_types:
            if not isinstance(x, InteractionTypes):
                raise TypeError("allowed interaction types must be of type"
                                "[InteractionTypes]")
            if x not in self.interaction_type_settings:
                logging.info(self.interaction_type_settings.keys())
                raise ValueError("interaction " + str(x) +
                                 " not found in settings")
        self.allowed_interaction_types = allowed_interaction_types

    def prepare_graphs(self):
        topology_graphs = self.build_topologies()
        init_graphs = self.create_seed_graphs(topology_graphs)
        graph_node_setting_pairs = self.determine_node_settings(init_graphs)
        # create groups of settings ordered by "probablity"
        graph_settings_groups = self.create_interaction_setting_groups(
            graph_node_setting_pairs)
        return graph_settings_groups

    def build_topologies(self):
        all_graphs = self.topology_builder.build_graphs(
            len(self.initial_state), len(self.final_state))
        logging.info("number of tolopogy graphs: " + str(len(all_graphs)))
        return all_graphs

    def create_seed_graphs(self, topology_graphs):
        # initialize the graph edges (intial and final state)
        init_graphs = []
        for tgraph in topology_graphs:
            tgraph.set_graph_element_properties_comparator(
                compare_graph_element_properties)
            init_graphs.extend(initialize_graph(
                tgraph, self.initial_state, self.final_state))

        logging.info("initialized " + str(len(init_graphs)) + " graphs!")
        logging.info("Now removing graphs based on required final state"
                     " groupings")
        graphs_to_remove = []
        # remove graphs which do not show the final state groupings
        if self.final_state_groupings:
            for igraph in init_graphs:
                valid_groupings = []
                for fs_grouping in self.final_state_groupings:
                    # check if this grouping is available in this graph
                    valid_grouping = False

                    possible_fs_groupings = convert_fs_names_to_edge_ids(
                        igraph, fs_grouping)

                    for possible_fs_grouping in possible_fs_groupings:
                        valid_current_fs_grouping_val = True
                        for group_fs_list in possible_fs_grouping:
                            fs_group_found = False
                            for node_id in igraph.nodes:
                                node_fs_list = set(
                                    get_originating_final_state_edges(
                                        igraph, node_id))
                                if group_fs_list == node_fs_list:
                                    fs_group_found = True
                                    break
                            if not fs_group_found:
                                valid_current_fs_grouping_val = False
                                break
                        if valid_current_fs_grouping_val:
                            valid_grouping = True
                            valid_groupings.append(
                                self.final_state_groupings.index(fs_grouping)
                            )
                            break
                    if valid_grouping:
                        break
                if len(valid_groupings) == 0:
                    graphs_to_remove.append(init_graphs.index(igraph))
        graphs_to_remove.sort(reverse=True)
        for i in graphs_to_remove:
            del init_graphs[i]

        logging.info("initialized " + str(len(init_graphs)) + " graphs!")
        return init_graphs

    def determine_node_settings(self, graphs):
        graph_node_setting_pairs = []
        for graph in graphs:
            final_state_edges = get_final_state_edges(graph)
            initial_state_edges = get_initial_state_edges(graph)
            node_settings = {}
            for node_id in graph.nodes:
                node_int_types = []
                out_edge_ids = get_edges_outgoing_to_node(graph, node_id)
                in_edge_ids = get_edges_outgoing_to_node(graph, node_id)
                in_edge_props = [graph.edge_props[edge_id] for edge_id in
                                 [x for x in in_edge_ids
                                  if x in initial_state_edges]]
                out_edge_props = [graph.edge_props[edge_id] for edge_id in
                                  [x for x in out_edge_ids
                                   if x in final_state_edges]]
                node_props = {}
                if node_id in graph.node_props:
                    node_props = graph.node_props[node_id]
                for int_det in self.interaction_determinators:
                    node_int_types.append(
                        int_det.check(in_edge_props,
                                      out_edge_props,
                                      node_props)
                    )
                node_int_types = filter_interaction_types(
                    node_int_types,
                    self.allowed_interaction_types)
                logging.debug(
                    "using " + str(node_int_types)
                    + " interaction order for node: " + str(node_id))
                node_settings[node_id] = [
                    deepcopy(self.interaction_type_settings[x])
                    for x in node_int_types]
            graph_node_setting_pairs.append((graph, node_settings))
        return graph_node_setting_pairs

    def create_interaction_setting_groups(self, graph_node_setting_pairs):
        graph_settings_groups = {}
        for (graph, node_settings) in graph_node_setting_pairs:
            setting_combinations = create_setting_combinations(node_settings)
            for setting in setting_combinations:
                strength = calculate_strength(setting)
                if strength not in graph_settings_groups:
                    graph_settings_groups[strength] = []
                graph_settings_groups[strength].append((graph, setting))
        return graph_settings_groups

    def find_solutions(self, graph_setting_groups):
        results = {}
        # check for solutions for a specific set of interaction settings
        logging.info("Number of interaction settings groups being processed: "
                     + str(len(graph_setting_groups)))
        for strength, graph_setting_group in sorted(
                graph_setting_groups.items(), reverse=True):
            logging.info("processing interaction settings group with "
                         "strength " + str(strength))
            logging.info(str(len(graph_setting_group)) +
                         " entries in this group")
            logging.info("running with " +
                         str(self.number_of_threads) + " threads...")

            temp_results = []
            if self.number_of_threads > 1:
                with Pool(self.number_of_threads) as p:
                    temp_results = p.imap_unordered(
                        self.propagate_quantum_numbers, graph_setting_group, 1)
                    p.close()
                    p.join()
            else:
                for graph_setting_pair in graph_setting_group:
                    temp_results.append(self.propagate_quantum_numbers(
                        graph_setting_pair))

            if strength not in results:
                results[strength] = []
            results[strength].extend(temp_results)

        # filter solutions, by removing those which only differ in
        # the interaction S qn
        results = filter_solutions(results, self.filter_remove_qns,
                                   self.filter_ignore_qns)

        node_non_satisfied_rules = []
        solutions = []
        for result in results.values():
            for (tempsolutions, non_satisfied_laws) in result:
                solutions.extend(tempsolutions)
                node_non_satisfied_rules.append(non_satisfied_laws)
        logging.info("total number of found solutions: " +
                     str(len(solutions)))
        violated_laws = []
        if len(solutions) == 0:
            violated_laws = analyse_solution_failure(node_non_satisfied_rules)

        return (solutions, violated_laws)

    def propagate_quantum_numbers(self, state_graph_node_settings_pair):
        propagator = self.initialize_qn_propagator(
            state_graph_node_settings_pair[0],
            state_graph_node_settings_pair[1])
        solutions = propagator.find_solutions()
        return (solutions, propagator.get_non_satisfied_conservation_laws())

    def initialize_qn_propagator(self, state_graph, node_settings):
        propagator = FullPropagator(state_graph)
        for node_id, interaction_settings in node_settings.items():
            propagator.assign_settings_to_node(
                node_id, interaction_settings)
        # specify set of particles which are allowed to be intermediate
        # particles. If list is empty, then all particles in the default
        # particle list are used
        propagator.set_allowed_intermediate_particles(
            self.allowed_intermediate_particles)

        return propagator
