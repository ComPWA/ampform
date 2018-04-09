import logging
from copy import deepcopy
from collections import OrderedDict

import xmltodict

from core.topology.graph import (StateTransitionGraph,
                                 InteractionNode,
                                 get_edges_outgoing_to_node,
                                 get_edges_ingoing_to_node,
                                 get_final_state_edges,
                                 get_initial_state_edges,
                                 get_originating_final_state_edges)
from core.topology.topologybuilder import SimpleStateTransitionTopologyBuilder

from core.state.particle import (
    load_particle_list_from_xml, particle_list, initialize_graph,
    get_particle_property,
    StateQuantumNumberNames, InteractionQuantumNumberNames, create_spin_domain,
    XMLLabelConstants, get_xml_label)
from core.state.propagation import (FullPropagator)
from core.state.conservationrules import (AdditiveQuantumNumberConservation,
                                          ParityConservation,
                                          IdenticalParticleSymmetrization,
                                          SpinConservation,
                                          HelicityConservation,
                                          CParityConservation,
                                          GParityConservation,
                                          GellMannNishijimaRule,
                                          MassConservation)


def group_graphs_same_initial_and_final(graphs):
    return [graphs]


def get_helicity_from_edge_props(edge_props):
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
    type_label = get_xml_label(XMLLabelConstants.Type)
    spin_label = StateQuantumNumberNames.Spin
    proj_label = get_xml_label(XMLLabelConstants.Projection)
    for qn in edge_props[qns_label]:
        if qn[type_label] == spin_label.name:
            return qn[proj_label]
    print(edge_props[qns_label])
    raise ValueError("Could not find spin projection quantum number!")


def determine_final_state(graph, edge_id):
    return "0"


def get_recoil_edge(graph, edge_id):
    return 0


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


class HelicityDecayAmplitudeGeneratorXML():
    def __init__(self, graphs):
        self.particle_list = {}
        if len(graphs) <= 0:
            raise ValueError(
                "Number of solution graphs is not larger than zero!")
        self.graphs = graphs
        self.helicity_amplitudes = {}
        self.kinematics = {}
        self.generate_particle_list()
        self.generate_kinematics()
        self.generate_amplitude_info()

    def generate_particle_list(self):
        # create particle entries
        temp_particle_names = []
        particles = []
        for g in self.graphs:
            for edge_props in g.edge_props.values():
                par_name = edge_props[get_xml_label(XMLLabelConstants.Name)]
                if par_name not in temp_particle_names:
                    particles.append(edge_props)
                    temp_particle_names.append(par_name)
        self.particle_list = {'ParticleList': {'Particle': particles}}

    def generate_amplitude_info(self):
        graph_groups = group_graphs_same_initial_and_final(self.graphs)

        # for each graph group we create a coherent amplitude
        coherent_amplitudes = []
        for graph_group in graph_groups:
            seq_partial_decays = []
            for graph in graph_group:
                partial_decays = []
                for node_id in graph.nodes:
                    decay_products = []
                    for out_edge_id in get_edges_outgoing_to_node(graph,
                                                                  node_id):
                        decay_products.append({
                            '@Name': graph.edge_props[out_edge_id]['@Name'],
                            '@FinalState': determine_final_state(graph,
                                                                 out_edge_id),
                            '@Helicity': get_helicity_from_edge_props(
                                graph.edge_props[out_edge_id])
                        })
                    in_edge_ids = get_edges_ingoing_to_node(graph, node_id)
                    if len(in_edge_ids) != 1:
                        raise ValueError(
                            "This node does not represent a two body decay!")
                    dec_part = graph.edge_props[in_edge_ids[0]]
                    dec_part_name = dec_part['@Name']
                    partial_decays.append({
                        'Parameter': [{'@Class': "Double", '@Type': "Magnitude",
                                       '@Name': "Magnitude_" + dec_part_name,
                                       'Value': 1.0, 'Fix': True},
                                      {'@Class': "Double", '@Type': "Phase",
                                       '@Name': "Phase_" + dec_part_name,
                                       'Value': 0.0, 'Fix': True}],
                        'DecayParticle': {
                            '@Name': dec_part['@Name'],
                            '@Helicity': get_helicity_from_edge_props(dec_part)
                        },
                        'RecoilSystem': {'@FinalState':
                                         determine_final_state(
                                             graph,
                                             get_recoil_edge(
                                                 graph, in_edge_ids[0])
                                         )
                                         },
                        'DecayProducts': {'Particle': decay_products}
                    })

                seq_decay_amp_name = "asdfsadf"
                seq_partial_decays.append({
                    '@Class': "SequentialPartialAmplitude",
                    '@Name': seq_decay_amp_name,
                    'Parameter': [{'@Class': "Double", '@Type': "Magnitude",
                                   '@Name': "Magnitude_" + seq_decay_amp_name,
                                   'Value': 1.0, 'Fix': True},
                                  {'@Class': "Double", '@Type': "Phase",
                                   '@Name': "Phase_" + seq_decay_amp_name,
                                   'Value': 0.0, 'Fix': True}],
                    'PartialAmplitude': partial_decays
                })

            # and in each coherent amplitude we create a product of partial decays
            coherent_amp_name = "coherent"
            coherent_amplitudes.append({
                '@Class': 'Coherent', '@Name': coherent_amp_name,
                'Parameter': {'@Class': "Double", '@Type': "Strength",
                              '@Name': "strength_" + coherent_amp_name,
                              'Value': 1, 'Fix': True},
                'Amplitude': seq_partial_decays
            })

        # now wrap it with an incoherent intensity
        incoherent_amp_name = "incoherent"
        self.helicity_amplitudes = {
            'Intensity': {
                '@Class': "Incoherent", '@Name': incoherent_amp_name,
                'Parameter': {'@Class': "Double", '@Type': "Strength",
                              '@Name': "strength_" + incoherent_amp_name,
                              'Value': 1, 'Fix': True},
                'Intensity': coherent_amplitudes
            }
        }

    def generate_kinematics(self):
        tempdict = {
            #<PhspVolume>0.541493</PhspVolume>
            'InitialState': {'Particle': []}, 'FinalState': {'Particle': []}
        }
        is_edge_ids = get_initial_state_edges(self.graphs[0])
        for x in is_edge_ids:
            tempdict['InitialState']['Particle'].append(
                {'@Name': self.graphs[0].edge_props[x]['@Name'], '@Id': x})
        fs_edge_ids = get_final_state_edges(self.graphs[0])
        for x in fs_edge_ids:
            tempdict['FinalState']['Particle'].append(
                {'@Name': self.graphs[0].edge_props[x]['@Name'], '@Id': x})
        self.kinematics = {'HelicityKinematics': tempdict}

    def write_to_xml(self, output_file):
        with open(output_file, mode='w') as xmlfile:
            full_dict = self.particle_list
            full_dict.update(self.kinematics)
            full_dict.update(self.helicity_amplitudes)
            xmltodict.unparse(OrderedDict(
                {'root': full_dict}), output=xmlfile, pretty=True)
            xmlfile.close()


class InteractionTypeSettings:
    def __init__(self, formalism_type, use_mass_conservation=True):
        self.interaction_settings = {}
        self.formalism_conservation_laws = []
        if formalism_type is 'helicity':
            self.formalism_conservation_laws = [
                SpinConservation(StateQuantumNumberNames.Spin, False),
                HelicityConservation()]
        if use_mass_conservation:
            self.formalism_conservation_laws.append(MassConservation())
        self.create_default_settings()

    def create_default_settings(self):
        weak_conservation_laws = self.formalism_conservation_laws
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
        self.interaction_settings['weak'] = (
            weak_conservation_laws,
            [],
            {
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
                StateQuantumNumberNames.Strangeness: [-1, 0, 1],
                InteractionQuantumNumberNames.L: create_spin_domain(
                    [0, 1, 2], True),
                InteractionQuantumNumberNames.S: create_spin_domain(
                    [0, 0.5, 1], True)
            }
        )
        em_cons_law_list = deepcopy(self.interaction_settings['weak'][0])
        em_cons_law_list.extend(
            [AdditiveQuantumNumberConservation(
                StateQuantumNumberNames.Charm),
             AdditiveQuantumNumberConservation(
                StateQuantumNumberNames.Strangeness),
             ParityConservation(),
             CParityConservation()]
        )
        self.interaction_settings['em'] = (
            em_cons_law_list,
            self.interaction_settings['weak'][1],
            self.interaction_settings['weak'][2]
        )
        strong_cons_law_list = deepcopy(self.interaction_settings['em'][0])
        strong_cons_law_list.extend(
            [SpinConservation(
                StateQuantumNumberNames.IsoSpin),
             GParityConservation()]
        )
        self.interaction_settings['strong'] = (
            strong_cons_law_list,
            self.interaction_settings['em'][1],
            self.interaction_settings['em'][2]
        )

    def get_settings(self):
        return self.interaction_settings


class StateTransitionManager():
    def __init__(self, initial_state, final_state,
                 allowed_intermediate_particles=[],
                 formalism_type='helicity',
                 default_interaction_type='strong',
                 topology_building='isobar'):
        self.initial_state = initial_state
        self.final_state = final_state
        self.interaction_settings = InteractionTypeSettings(
            formalism_type).get_settings()
        self.default_interaction_type = default_interaction_type
        self.allowed_intermediate_particles = allowed_intermediate_particles
        self.final_state_groupings = []
        int_nodes = []
        if topology_building == 'isobar':
            if len(initial_state) == 1:
                int_nodes.append(InteractionNode("TwoBodyDecay", 1, 2))
        else:
            int_nodes.append(InteractionNode(
                "NBodyScattering",
                len(initial_state),
                len(final_state)))
            self.interaction_settings = InteractionTypeSettings(
                formalism_type, False).get_settings()
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

    def prepare_graphs(self):
        topology_graphs = self.build_topologies()
        init_graphs = self.create_seed_graphs(topology_graphs)
        graph_node_setting_pairs = self.determine_node_settings(init_graphs)
        return graph_node_setting_pairs

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
        eln_label = StateQuantumNumberNames.ElectronLN
        mln_label = StateQuantumNumberNames.MuonLN
        tln_label = StateQuantumNumberNames.TauLN
        name_label = get_xml_label(XMLLabelConstants.Name)
        qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)

        for graph in graphs:
            final_state_edges = get_final_state_edges(graph)
            initial_state_edges = get_initial_state_edges(graph)
            node_settings = {}
            for node_id in graph.nodes:
                node_interaction_type = 'default'
                out_edge_ids = get_edges_outgoing_to_node(graph, node_id)
                in_edge_ids = get_edges_outgoing_to_node(graph, node_id)
                for edge_id in [x for x in out_edge_ids + in_edge_ids
                                if x in final_state_edges +
                                initial_state_edges]:
                    edge_props = graph.edge_props[edge_id]
                    if ('gamma' in graph.edge_props[edge_id][name_label]):
                        node_interaction_type = 'em'
                    elif sum([get_particle_property(edge_props, x)
                              for x in [eln_label, mln_label, tln_label]
                              if get_particle_property(edge_props, x) is not None]):
                        if [x for x in
                            ['ve', 'vebar', 'vmu', 'vmubar', 'vtau', 'vtaubar']
                                if x == graph.edge_props[edge_id][name_label]]:
                            node_interaction_type = 'weak'
                            break
                        if graph.edge_props[edge_id][qns_label] != 0:
                            node_interaction_type = 'em'

                if node_interaction_type == 'em':
                    node_settings[node_id] = self.interaction_settings['em']
                    logging.debug(
                        "using em interaction settings for node: "
                        + str(node_id))
                elif node_interaction_type == 'weak':
                    node_settings[node_id] = self.interaction_settings['weak']
                    logging.debug(
                        "using weak interaction settings for node: "
                        + str(node_id))
                else:
                    node_settings[node_id] = self.interaction_settings[
                        self.default_interaction_type]
                    logging.debug(
                        "using " + str(self.default_interaction_type)
                        + " interaction settings for node: "
                        + str(node_id))
            graph_node_setting_pairs.append((graph, node_settings))
        return graph_node_setting_pairs

    def find_solutions(self, graph_node_setting_pairs):
        solutions = []
        violated_rules = []
        node_non_satisfied_rules = []
        int_spin_label = InteractionQuantumNumberNames.S
        qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
        type_label = get_xml_label(XMLLabelConstants.Type)
        for (igraph, node_settings) in graph_node_setting_pairs:
            result = self.propagate_quantum_numbers(igraph, node_settings)
            temp_solutions = result[0]
            node_non_satisfied_rules.append(result[1])
            # we want to remove solutions which only differ
            # in the interaction S qn
            for sol_graph in temp_solutions:
                for node_id, props in sol_graph.node_props.items():
                    if qns_label in props:
                        for qn_entry in props[qns_label]:
                            if (InteractionQuantumNumberNames[qn_entry[type_label]]
                                    is int_spin_label):
                                del props[qns_label][props[qns_label].index(
                                    qn_entry)]
                                break
                if sol_graph not in solutions:
                    solutions.append(sol_graph)
        logging.info("total number of found solutions: " +
                     str(len(solutions)))
        if len(solutions) == 0:
            violated_rules = self.analyse_solution_failure(
                node_non_satisfied_rules)
        return (solutions, violated_rules)

    def propagate_quantum_numbers(self, state_graph, node_settings):
        propagator = self.initialize_qn_propagator(state_graph, node_settings)
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
            propagator.assign_qn_domains_to_node(node_id,
                                                 interaction_settings[2])
        # specify set of particles which are allowed to be intermediate
        # particles. If list is empty, then all particles in the default
        # particle list are used
        propagator.set_allowed_intermediate_particles(
            self.allowed_intermediate_particles)

        return propagator

    def analyse_solution_failure(self, violated_laws_per_node_and_graph):
        # try to find rules that are just always violated
        violated_laws = []
        scoresheet = {}

        for violated_laws_per_node in violated_laws_per_node_and_graph:
            temp_violated_laws = set()
            for node_id, laws in violated_laws_per_node.items():
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
