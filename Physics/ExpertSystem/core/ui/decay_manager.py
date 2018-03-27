import logging
from copy import deepcopy

from core.topology.graph import (InteractionNode,
                                 get_edges_outgoing_to_node,
                                 get_edges_ingoing_to_node,
                                 get_final_state_edges,
                                 get_initial_state_edges)
from core.topology.topologybuilder import SimpleStateTransitionTopologyBuilder

from core.state.particle import (
    load_particle_list_from_xml, particle_list, initialize_graph,
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


class InteractionTypeSettings:
    def __init__(self, formalism_type):
        self.interaction_settings = {}
        self.formalism_conservation_laws = []
        if formalism_type is 'helicity':
            self.formalism_conservation_laws = [
                SpinConservation(StateQuantumNumberNames.Spin, False),
                HelicityConservation()]
        self.create_default_settings()

    def create_default_settings(self):
        weak_conservation_laws = self.formalism_conservation_laws
        weak_conservation_laws.extend([
            GellMannNishijimaRule(),
            AdditiveQuantumNumberConservation(
                StateQuantumNumberNames.Charge),
            AdditiveQuantumNumberConservation(
                StateQuantumNumberNames.BaryonNumber),
            IdenticalParticleSymmetrization(),
            MassConservation()
        ])
        self.interaction_settings['weak'] = (
            weak_conservation_laws,
            [],
            {
                StateQuantumNumberNames.Charge: [-2, -1, 0, 1, 2],
                StateQuantumNumberNames.BaryonNumber: [-2, -1, 0, 1, 2],
                #  ParticleQuantumNumberNames.LeptonNumber: [-2, -1, 0, 1, 2],
                StateQuantumNumberNames.Parity: [-1, 1],
                StateQuantumNumberNames.Cparity: [-1, 1, None],
                StateQuantumNumberNames.Gparity: [-1, 1, None],
                StateQuantumNumberNames.Spin: create_spin_domain([0, 1, 2]),
                StateQuantumNumberNames.IsoSpin: create_spin_domain([0, 0.5, 1]),
                StateQuantumNumberNames.Charm: [-1, 0, 1],
                StateQuantumNumberNames.Strangeness: [-1, 0, 1],
                InteractionQuantumNumberNames.L: create_spin_domain([0, 1, 2],
                                                                    True),
                InteractionQuantumNumberNames.S: create_spin_domain([0, 1], True)
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


class TwoBodyDecayManager():
    def __init__(self, initial_state, final_state,
                 allowed_intermediate_particles=[],
                 default_interaction_type='strong',
                 formalism_type='helicity'):
        self.initial_state = initial_state
        self.final_state = final_state
        self.interaction_settings = InteractionTypeSettings(
            formalism_type).get_settings()
        self.default_interaction_type = default_interaction_type
        self.allowed_intermediate_particles = allowed_intermediate_particles

    def prepare_graphs(self):
        init_graphs = self.create_seed_graphs()
        graph_node_setting_pairs = self.determine_node_settings(init_graphs)
        return graph_node_setting_pairs

    def create_seed_graphs(self):
        topology_graphs = self.build_topologies()
        # load default particles from database/file
        load_particle_list_from_xml('../particle_list.xml')
        # print(particle_list)
        logging.info("loaded " + str(len(particle_list))
                     + " particles from xml file!")

        # initialize the graph edges (intial and final state)
        init_graphs = []
        for tgraph in topology_graphs:
            init_graphs.extend(initialize_graph(
                tgraph, self.initial_state, self.final_state))
        logging.info("initialized " + str(len(init_graphs)) + " graphs!")
        return init_graphs

    def build_topologies(self):
        two_body_decay_node = InteractionNode("TwoBodyDecay", 1, 2)
        simple_builder = SimpleStateTransitionTopologyBuilder(
            [two_body_decay_node])
        all_graphs = simple_builder.build_graphs(
            len(self.initial_state), len(self.final_state))
        logging.info("number of tolopogy graphs: " + str(len(all_graphs)))
        return all_graphs

    def determine_node_settings(self, graphs):
        graph_node_setting_pairs = []
        leptonnum_label = StateQuantumNumberNames.LeptonNumber
        qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
        # TODO: try to guess interaction type for each node correctly
        for graph in graphs:
            final_state_edges = get_final_state_edges(graph)
            initial_state_edges = get_initial_state_edges(graph)
            node_settings = {}
            for node_id in graph.nodes:
                out_edge_ids = get_edges_outgoing_to_node(graph, node_id)
                in_edge_ids = get_edges_outgoing_to_node(graph, node_id)
                for edge_id in [x for x in out_edge_ids + in_edge_ids
                                if x in final_state_edges +
                                initial_state_edges]:
                    if ('gamma' in graph.edge_props[edge_id][
                            get_xml_label(XMLLabelConstants.Name)]):
                        node_settings[node_id] = self.interaction_settings['em']
                        logging.debug(
                            "using em interaction settings for node: "
                            + str(node_id))
                        break
                    elif leptonnum_label in graph.edge_props[edge_id][qns_label]:
                        if graph.edge_props[edge_id][qns_label] != 0:
                            node_settings[node_id] = self.interaction_settings['em']
                            logging.debug(
                                "using em interaction settings for node: "
                                + str(node_id))
                            break
                if node_id not in node_settings:
                    node_settings[node_id] = self.interaction_settings[self.default_interaction_type]
                    logging.debug(
                        "using " + str(self.default_interaction_type)
                        + " interaction settings for node: "
                        + str(node_id))
            graph_node_setting_pairs.append((graph, node_settings))
        return graph_node_setting_pairs

    def find_solutions(self, graphs_node_setting_pairs):
        solutions = []
        int_spin_label = InteractionQuantumNumberNames.S
        qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
        type_label = get_xml_label(XMLLabelConstants.Type)
        for (igraph, node_settings) in graphs_node_setting_pairs:
            temp_solutions = self.propagate_quantum_numbers(
                igraph, node_settings)
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
        return solutions

    def propagate_quantum_numbers(self, state_graph, node_settings):
        propagator = self.initialize_qn_propagator(state_graph, node_settings)
        solutions = propagator.find_solutions()
        return solutions

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
