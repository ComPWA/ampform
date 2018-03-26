import logging
from copy import deepcopy

from core.topology.graph import InteractionNode
from core.topology.topologybuilder import SimpleStateTransitionTopologyBuilder

from core.state.particle import (
    load_particle_list_from_xml, particle_list, initialize_graph,
    StateQuantumNumberNames, InteractionQuantumNumberNames, create_spin_domain)
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
                # StateQuantumNumberNames.Gparity: [-1, 1, None],
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
                StateQuantumNumberNames.IsoSpin)]
        )
        self.interaction_settings['strong'] = (
            strong_cons_law_list,
            self.interaction_settings['em'][1],
            self.interaction_settings['em'][2]
        )

    def get_settings(self):
        return self.interaction_settings


class TwoBodyDecayManager():
    def __init__(self, initial_state, final_state, formalism_type='helicity',
                 default_interaction_type='strong'):
        self.initial_state = initial_state
        self.final_state = final_state
        self.interaction_settings = InteractionTypeSettings(
            formalism_type).get_settings()
        self.default_interaction_type = default_interaction_type

    def find_solutions(self):
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

        solutions = []
        for igraph in init_graphs:
            solutions.extend(self.propagate_quantum_numbers(igraph))
        logging.info("total number of found solutions: " +
                     str(len(solutions)))
        return solutions

    def build_topologies(self):
        two_body_decay_node = InteractionNode("TwoBodyDecay", 1, 2)
        simple_builder = SimpleStateTransitionTopologyBuilder(
            [two_body_decay_node])
        all_graphs = simple_builder.build_graphs(
            len(self.initial_state), len(self.final_state))
        logging.info("number of tolopogy graphs: " + str(len(all_graphs)))
        return all_graphs

    def propagate_quantum_numbers(self, state_graph):
        propagator = self.initialize_qn_propagator(state_graph)
        solutions = propagator.find_solutions()
        return solutions

    def initialize_qn_propagator(self, state_graph):
        # TODO: try to guess interaction type for each node correctly
        # for now we just take the default type specified by the user
        interaction_settings = self.interaction_settings[
            self.default_interaction_type]
        strict_conservation_rules = interaction_settings[0]
        non_strict_conservation_rules = interaction_settings[1]
        quantum_number_domains = interaction_settings[2]

        propagator = FullPropagator(state_graph)
        propagator.assign_conservation_laws_to_all_nodes(
            strict_conservation_rules)
        propagator.assign_conservation_laws_to_all_nodes(
            non_strict_conservation_rules, False)
        propagator.assign_qn_domains_to_all_nodes(quantum_number_domains)
        # specify set of particles which are allowed to be intermediate
        # particles. If list is empty, then all particles in the default
        # particle list are used
        allowed_intermediate_particles = []
        propagator.set_allowed_intermediate_particles(
            allowed_intermediate_particles)

        return propagator
