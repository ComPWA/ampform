""" sample script for the testing purposes using the decay
    JPsi -> gamma pi0 pi0
"""
import logging

from core.ui.decay_manager import TwoBodyDecayManager

logging.basicConfig(level=logging.DEBUG)

# initialize the graph edges (intial and final state)
initial_state = [("J/psi", [-1, 1])]
final_state = [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])]

tbd_manager = TwoBodyDecayManager(initial_state, final_state)

solutions = tbd_manager.find_solutions()
print("found " + str(len(solutions)) + " solutions!")

for g in solutions:
    #print(g.node_props[0])
    #print(g.node_props[1])
    print(g.edge_props[1]['@Name'])