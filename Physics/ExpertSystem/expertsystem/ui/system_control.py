import logging
from copy import deepcopy
from itertools import product
from abc import ABC, abstractmethod
from enum import Enum
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
    StateQuantumNumberNames, InteractionQuantumNumberNames, create_spin_domain)

from expertsystem.state.propagation import (FullPropagator)

from expertsystem.state.conservationrules import (
    AdditiveQuantumNumberConservation,
    ParityConservation,
    ParityConservationHelicity,
    IdenticalParticleSymmetrization,
    SpinConservation,
    HelicityConservation,
    CParityConservation,
    GParityConservation,
    GellMannNishijimaRule,
    MassConservation)


InteractionTypes = Enum('InteractionTypes', 'Undefined Strong EM Weak')


def create_default_settings(formalism_type, use_mass_conservation=True):
    '''
    Create a container, which holds the settings for the various interactions
    (e.g.: strong, em and weak interaction).
    Each settings is a tuple of four values:
        1. list of strict conservation laws
        2. list of non-strict conservation laws
        2. list of quantum number domains
        3. strength scale parameter (higher value means stronger force)
    '''
    InteractionTypeSettings = {}
    formalism_conservation_laws = []
    formalism_qn_domains = {}
    formalism_type = formalism_type
    if formalism_type is 'helicity':
        formalism_conservation_laws = [
            SpinConservation(StateQuantumNumberNames.Spin, False),
            HelicityConservation()]
        formalism_qn_domains = {
            InteractionQuantumNumberNames.L: create_spin_domain(
                [0, 1, 2], True),
            InteractionQuantumNumberNames.S: create_spin_domain(
                [0, 0.5, 1, 1.5, 2], True)}
    elif formalism_type is 'canonical':
        formalism_conservation_laws = [
            SpinConservation(StateQuantumNumberNames.Spin)]
        formalism_qn_domains = {
            InteractionQuantumNumberNames.L: create_spin_domain(
                [0, 1, 2]),
            InteractionQuantumNumberNames.S: create_spin_domain(
                [0, 0.5, 1, 2])}
    if use_mass_conservation:
        formalism_conservation_laws.append(MassConservation())

    weak_conservation_laws = formalism_conservation_laws
    weak_conservation_laws.extend([
        GellMannNishijimaRule(),
        AdditiveQuantumNumberConservation(
            StateQuantumNumberNames.Charge),
        AdditiveQuantumNumberConservation(
            StateQuantumNumberNames.ElectronLN),
        AdditiveQuantumNumberConservation(
            StateQuantumNumberNames.MuonLN),
        AdditiveQuantumNumberConservation(
            StateQuantumNumberNames.TauLN),
        AdditiveQuantumNumberConservation(
            StateQuantumNumberNames.BaryonNumber),
        IdenticalParticleSymmetrization()
    ])
    qn_domains = {
        StateQuantumNumberNames.Charge: [-2, -1, 0, 1, 2],
        StateQuantumNumberNames.BaryonNumber: [-1, 0, 1],
        StateQuantumNumberNames.ElectronLN: [-1, 0, 1],
        StateQuantumNumberNames.MuonLN: [-1, 0, 1],
        StateQuantumNumberNames.TauLN: [-1, 0, 1],
        StateQuantumNumberNames.Parity: [-1, 1],
        StateQuantumNumberNames.Cparity: [-1, 1, None],
        StateQuantumNumberNames.Gparity: [-1, 1, None],
        StateQuantumNumberNames.Spin: create_spin_domain(
            [0, 0.5, 1, 1.5, 2]),
        StateQuantumNumberNames.IsoSpin: create_spin_domain(
            [0, 0.5, 1, 1.5]),
        StateQuantumNumberNames.Charm: [-1, 0, 1],
        StateQuantumNumberNames.Strangeness: [-1, 0, 1]
    }
    qn_domains.update(formalism_qn_domains)

    InteractionTypeSettings[InteractionTypes.Weak] = (
        weak_conservation_laws,
        [],
        qn_domains,
        10**(-4)
    )
    em_cons_law_list = deepcopy(InteractionTypeSettings[
        InteractionTypes.Weak][0])
    em_cons_law_list.extend(
        [AdditiveQuantumNumberConservation(
            StateQuantumNumberNames.Charm),
            AdditiveQuantumNumberConservation(
            StateQuantumNumberNames.Strangeness),
            ParityConservation(),
            CParityConservation()
         ]
    )
    em_qn_domains = InteractionTypeSettings[
        InteractionTypes.Weak][2]
    if formalism_type == 'helicity':
        em_cons_law_list.append(ParityConservationHelicity())
        em_qn_domains.update({
            InteractionQuantumNumberNames.ParityPrefactor: [-1, 1]
        })

    InteractionTypeSettings[InteractionTypes.EM] = (
        em_cons_law_list,
        InteractionTypeSettings[InteractionTypes.Weak][1],
        em_qn_domains,
        1
    )
    strong_cons_law_list = deepcopy(InteractionTypeSettings[
        InteractionTypes.EM][0])
    strong_cons_law_list.extend(
        [SpinConservation(
            StateQuantumNumberNames.IsoSpin),
            GParityConservation()]
    )
    InteractionTypeSettings[InteractionTypes.Strong] = (
        strong_cons_law_list,
        InteractionTypeSettings[InteractionTypes.EM][1],
        InteractionTypeSettings[InteractionTypes.EM][2],
        60
    )
    return InteractionTypeSettings


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


def get_final_state_edge_ids(graph, list_of_particle_names):
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    name_label = get_xml_label(XMLLabelConstants.Name)
    fsp_names = {graph.edge_props[i][name_label]: i
                 for i in get_final_state_edges(graph)}
    edge_list = []
    for particle_name in list_of_particle_names:
        if particle_name in fsp_names:
            edge_list.append(fsp_names[particle_name])
    return edge_list


def filter_solutions(results):
    filtered_results = {}
    int_spin_label = InteractionQuantumNumberNames.S
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
    type_label = get_xml_label(XMLLabelConstants.Type)
    solutions = []
    remove_counter = 0
    for strength, group_results in results.items():
        for (sol_graphs, rule_violations) in group_results:
            temp_graphs = []
            for sol_graph in sol_graphs:
                for props in sol_graph.node_props.values():
                    if qns_label in props:
                        for qn_entry in props[qns_label]:
                            if (InteractionQuantumNumberNames[
                                    qn_entry[type_label]] is int_spin_label):
                                del props[qns_label][props[qns_label].index(
                                    qn_entry)]
                                break
                if sol_graph not in solutions:
                    solutions.append(sol_graph)
                    temp_graphs.append(sol_graph)
                else:
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
        strength *= int_setting[3]
    return strength


class StateTransitionManager():
    def __init__(self, initial_state, final_state,
                 allowed_intermediate_particles=[],
                 formalism_type='helicity',
                 topology_building='isobar',
                 number_of_threads=4):
        self.number_of_threads = number_of_threads
        self.initial_state = initial_state
        self.final_state = final_state
        self.InteractionTypeSettings = create_default_settings(formalism_type)
        self.interaction_determinators = [LeptonCheck(), GammaCheck()]
        self.allowed_intermediate_particles = allowed_intermediate_particles
        self.final_state_groupings = []
        self.allowed_interaction_types = [InteractionTypes.Strong,
                                          InteractionTypes.EM,
                                          InteractionTypes.Weak]
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
                self.InteractionTypeSettings = create_default_settings(
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
        self.final_state_groupings.append(fs_group)

    def set_interaction_settings(self, allowed_interaction_types):
        # verify order
        for x in allowed_interaction_types:
            if not isinstance(x, InteractionTypes):
                raise TypeError("allowed interaction types must be of type"
                                "[InteractionTypes]")
            if x not in self.InteractionTypeSettings:
                logging.info(self.InteractionTypeSettings.keys())
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
            init_graphs.extend(initialize_graph(
                tgraph, self.initial_state, self.final_state))

        graphs_to_remove = []
        # remove graphs which do not show the final state groupings
        if self.final_state_groupings:
            for igraph in init_graphs:
                valid_groupings = []
                for fs_grouping in self.final_state_groupings:
                    # check if this grouping is available in this graph
                    valid_grouping = True
                    for fs_group in fs_grouping:
                        group_fs_list = set(get_final_state_edge_ids(igraph,
                                                                     fs_group))
                        fs_group_found = False
                        for node_id in igraph.nodes:
                            node_fs_list = set(
                                get_originating_final_state_edges(
                                    igraph, node_id))
                            if group_fs_list == node_fs_list:
                                fs_group_found = True
                                break
                        if not fs_group_found:
                            valid_grouping = False
                            break
                    if valid_grouping:
                        valid_groupings.append(
                            self.final_state_groupings.index(fs_grouping)
                        )
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
                node_settings[node_id] = [self.InteractionTypeSettings[x]
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
                graph_settings_groups[strength].append((graph,
                                                        setting))
        return graph_settings_groups

    def find_solutions(self, graph_setting_groups):
        results = {}
        # check for solutions for a specific set of interaction settings
        for strength, graph_setting_group in sorted(
                graph_setting_groups.items(), reverse=True):
            logging.debug("processing interaction settings group with "
                          "strength " + str(strength))
            logging.info(str(len(graph_setting_group)) +
                         " entries in this group")
            logging.info("running with " +
                         str(self.number_of_threads) + " threads...")

            temp_results = []
            with Pool(self.number_of_threads) as p:
                temp_results = p.imap_unordered(self.propagate_quantum_numbers,
                                          graph_setting_group, 1)
                p.close()
                p.join()
            
            if strength not in results:
                results[strength] = []
            results[strength].extend(temp_results)

        # filter solutions, by removing those which only differ in
        # the interaction S qn
        results = filter_solutions(results)

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
            propagator.assign_conservation_laws_to_node(
                node_id,
                interaction_settings[0],
                True)
            propagator.assign_conservation_laws_to_node(
                node_id,
                interaction_settings[1],
                False)
            propagator.assign_qn_domains_to_node(
                node_id,
                interaction_settings[2])
        # specify set of particles which are allowed to be intermediate
        # particles. If list is empty, then all particles in the default
        # particle list are used
        propagator.set_allowed_intermediate_particles(
            self.allowed_intermediate_particles)

        return propagator
