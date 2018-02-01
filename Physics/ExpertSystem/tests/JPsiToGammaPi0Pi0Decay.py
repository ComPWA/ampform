""" sample script for the testing purposes using the decay
    JPsi -> gamma pi0 pi0
"""

from core.topology.graph import InteractionNode
from core.topology.topologybuilder import SimpleStateTransitionTopologyBuilder

from core.state.particle import (
    load_particle_list_from_xml, particle_list,
    initialize_graph, ParticleQuantumNumberNames,
    InteractionQuantumNumberNames)
from core.state.propagation import (CSPPropagator)
from core.state.conservationrules import (ChargeConservation,
                                          ParityConservation,
                                          IdenticalParticleSymmetrization,
                                          SpinConservation,
                                          HelicityConservation,
                                          CParityConservation)

TwoBodyDecayNode = InteractionNode("TwoBodyDecay", 1, 2)

SimpleBuilder = SimpleStateTransitionTopologyBuilder([TwoBodyDecayNode])

all_graphs = SimpleBuilder.build_graphs(1, 3)

print("we have " + str(len(all_graphs)) + " tolopogy graphs!\n")
print(all_graphs)

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

conservation_rules = {'strict':
                      [ChargeConservation(),
                       ParityConservation(),
                       IdenticalParticleSymmetrization(),
                       SpinConservation(
                          ParticleQuantumNumberNames.Spin, False),
                       HelicityConservation(),
                       CParityConservation()],
                      'non-strict':
                      [SpinConservation(
                          ParticleQuantumNumberNames.IsoSpin)]
                      }
quantum_number_domains = {ParticleQuantumNumberNames.Charge: [-2, -1, 0, 1, 2],
                          ParticleQuantumNumberNames.Parity: [-1, 1],
                          ParticleQuantumNumberNames.Cparity: [-1, 1],
                          ParticleQuantumNumberNames.Spin: [0, 1, 2],
                          ParticleQuantumNumberNames.IsoSpin: [0, 1],
                          InteractionQuantumNumberNames.L: [0, 1, 2]}

propagator = CSPPropagator(test_graph)
propagator.assign_conservation_laws_to_all_nodes(
    conservation_rules, quantum_number_domains)
solutions = propagator.find_solutions()
print("found " + str(len(solutions)) + " solutions!")

for g in solutions:
    print(g.node_props[0])
    print(g.node_props[1])
    print(g.edge_props[1])
