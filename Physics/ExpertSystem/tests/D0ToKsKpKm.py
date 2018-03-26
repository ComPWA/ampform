""" sample script for the testing purposes using the decay
    JPsi -> gamma pi0 pi0
"""
import logging

from core.ui.decay_manager import TwoBodyDecayManager

# logging.basicConfig(level=logging.DEBUG)

# initialize the graph edges (intial and final state)
initial_state = [("D0", [0])]
final_state = [("K_S0", [0]), ("K+", [0]), ("K-", [0])]

tbd_manager = TwoBodyDecayManager(initial_state, final_state,
                                  'helicity', 'weak')

solutions = tbd_manager.find_solutions()
print("found " + str(len(solutions)) + " solutions!")

for g in solutions:
    # print(g.node_props[0])
    # print(g.node_props[1])
    print(g.edge_props[1]['@Name'])
