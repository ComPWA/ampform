""" sample script for the testing purposes using the decay
    e+e- -> D0 D0bar pi+ pi-
"""
import logging

from core.topology.graph import InteractionNode
from core.topology.topologybuilder import SimpleStateTransitionTopologyBuilder

from core.state.particle import (
    load_particle_list_from_xml, particle_list,
    initialize_graph, StateQuantumNumberNames, InteractionQuantumNumberNames,
    create_spin_domain)
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

# uncomment line below to enable debug output
#logging.basicConfig(level=logging.DEBUG)

# ------------------ Creation of topology graphs ------------------

TwoBodyDecayNode = InteractionNode("TwoBodyDecay", 1, 2)

SimpleBuilder = SimpleStateTransitionTopologyBuilder([TwoBodyDecayNode])

all_graphs = SimpleBuilder.build_graphs(1, 4)

print("we have " + str(len(all_graphs)) + " tolopogy graphs!\n")
print(all_graphs)

# ------------------ first stage of QN propagation ------------------

# load default particles from database/file
load_particle_list_from_xml('../particle_list.xml')
# print(particle_list)
print("loaded " + str(len(particle_list)) + " particles from xml file!")

# initialize the graph edges (initial and final state)
initial_state = [("EpEm", [-1, 1])]
final_state = [("D0", [0]), ("D0bar", [0]), ("pi+", [0]), ("pi-", [0])]
init_graphs = initialize_graph(
    all_graphs[1], initial_state, final_state)
print("initialized " + str(len(init_graphs)) + " graphs!")

test_graph = init_graphs[12]
print("pick one test graph:")
print(test_graph)

strict_conservation_rules = [
    GellMannNishijimaRule(),
    AdditiveQuantumNumberConservation(StateQuantumNumberNames.Charge),
    SpinConservation(
        StateQuantumNumberNames.IsoSpin),
    AdditiveQuantumNumberConservation(StateQuantumNumberNames.Charm),
    AdditiveQuantumNumberConservation(
        StateQuantumNumberNames.BaryonNumber),
    #  AdditiveQuantumNumberConservation(
    #      ParticleQuantumNumberNames.LeptonNumber),
    SpinConservation(StateQuantumNumberNames.Spin, False),
    HelicityConservation(),
    ParityConservation(),
    CParityConservation(),
    IdenticalParticleSymmetrization(),
    MassConservation()
]

non_strict_conservation_rules = [
]

quantum_number_domains = {
    StateQuantumNumberNames.Charge: [-1, 0, 1],
    StateQuantumNumberNames.BaryonNumber: [0],
    #  ParticleQuantumNumberNames.LeptonNumber: [-2, -1, 0, 1, 2],
    StateQuantumNumberNames.Parity: [-1, 1],
    StateQuantumNumberNames.Cparity: [-1, 1, None],
    StateQuantumNumberNames.Spin: create_spin_domain([0, 1, 2]),
    StateQuantumNumberNames.IsoSpin: create_spin_domain([0, 0.5, 1]),
    StateQuantumNumberNames.Charm: [-1, 0, 1],
    InteractionQuantumNumberNames.L: create_spin_domain([0, 1, 2], True),
    InteractionQuantumNumberNames.S: create_spin_domain([0, 1], True)
}

propagator = FullPropagator(test_graph)
propagator.assign_conservation_laws_to_all_nodes(
    strict_conservation_rules)
propagator.assign_conservation_laws_to_all_nodes(
    non_strict_conservation_rules, False)
propagator.assign_qn_domains_to_all_nodes(quantum_number_domains)

# specify set of particles which are allowed to be intermediate particles
# if list is empty, then all particles in the default particle list are used
allowed_intermediate_particles = []
propagator.set_allowed_intermediate_particles(allowed_intermediate_particles)

solutions = propagator.find_solutions()

print("found " + str(len(solutions)) + " solutions!")
'''
for g in solutions:
    print(g.node_props[0])
    print(g.node_props[1])
    print(g.node_props[2])
    print(g.edge_props[1])
    print(g.edge_props[3])'''