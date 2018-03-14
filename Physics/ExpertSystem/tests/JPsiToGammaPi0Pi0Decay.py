""" sample script for the testing purposes using the decay
    JPsi -> gamma pi0 pi0
"""
import os

from core.topology.graph import InteractionNode
from core.topology.topologybuilder import SimpleStateTransitionTopologyBuilder

from core.state.particle import (
    load_particle_list_from_xml, particle_list,
    initialize_graph, initialize_graphs_with_particles,
    StateQuantumNumberNames, InteractionQuantumNumberNames,
    create_spin_domain)
from core.state.propagation import (CSPPropagator)
from core.state.conservationrules import (AdditiveQuantumNumberConservation,
                                          ParityConservation,
                                          IdenticalParticleSymmetrization,
                                          SpinConservation,
                                          HelicityConservation,
                                          CParityConservation,
                                          GParityConservation,
                                          GellMannNishijimaRule)


# ------------------ Creation of topology graphs ------------------

TwoBodyDecayNode = InteractionNode("TwoBodyDecay", 1, 2)

SimpleBuilder = SimpleStateTransitionTopologyBuilder([TwoBodyDecayNode])

all_graphs = SimpleBuilder.build_graphs(1, 3)

print("we have " + str(len(all_graphs)) + " tolopogy graphs!\n")
print(all_graphs)

# ------------------ first stage of QN propagation ------------------

# load default particles from database/file
load_particle_list_from_xml('../particle_list.xml')
# print(particle_list)
print("loaded " + str(len(particle_list)) + " particles from xml file!")

# initialize the graph edges (intial and final state)
initial_state = [("J/psi", [-1, 1])]
final_state = [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])]
init_graphs = initialize_graph(
    all_graphs[0], initial_state, final_state)
print("initialized " + str(len(init_graphs)) + " graphs!")
test_graph = init_graphs[0]


strict_conservation_rules = [
    AdditiveQuantumNumberConservation(StateQuantumNumberNames.Charge),
    AdditiveQuantumNumberConservation(
        StateQuantumNumberNames.BaryonNumber),
    #  AdditiveQuantumNumberConservation(
    #      ParticleQuantumNumberNames.LeptonNumber),
    ParityConservation(),
    CParityConservation(),
    IdenticalParticleSymmetrization(),
    SpinConservation(StateQuantumNumberNames.Spin, False),
    HelicityConservation(),
    GellMannNishijimaRule()
]

non_strict_conservation_rules = [
    SpinConservation(
        StateQuantumNumberNames.IsoSpin)
]

quantum_number_domains = {
    StateQuantumNumberNames.Charge: [-2, -1, 0, 1, 2],
    StateQuantumNumberNames.BaryonNumber: [-2, -1, 0, 1, 2],
    #  ParticleQuantumNumberNames.LeptonNumber: [-2, -1, 0, 1, 2],
    StateQuantumNumberNames.Parity: [-1, 1],
    StateQuantumNumberNames.Cparity: [-1, 1],
    StateQuantumNumberNames.Spin: create_spin_domain([0, 1, 2]),
    StateQuantumNumberNames.IsoSpin: create_spin_domain([0]),
    InteractionQuantumNumberNames.L: create_spin_domain([0, 1, 2], True),
    InteractionQuantumNumberNames.S: create_spin_domain([0, 1], True)
}

propagator = CSPPropagator(test_graph)
propagator.assign_conservation_laws_to_all_nodes(
    strict_conservation_rules)
propagator.assign_conservation_laws_to_all_nodes(
    non_strict_conservation_rules, False)
propagator.assign_qn_domains_to_all_nodes(quantum_number_domains)
solutions = propagator.find_solutions()

print("found " + str(len(solutions)) + " solutions!")

for g in solutions:
    print(g.node_props[0])
    print(g.node_props[1])
    print(g.edge_props[1])

# ------------------ second stage of QN propagation ------------------

# specify set of particles which are allowed to be intermediate particles
# if list is empty, then all particles in the default particle list are used
allowed_intermediate_particles = []

full_particle_graphs = initialize_graphs_with_particles(
    solutions, allowed_intermediate_particles)
print("Number of initialized graphs: " + str(len(full_particle_graphs)))
for g in full_particle_graphs:
    print(g)
