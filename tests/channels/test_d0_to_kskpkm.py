""" sample script for the testing purposes using the decay
    D0 -> K0bar K+ K-
"""

import logging

from expertsystem.ui import StateTransitionManager


def test_script():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    # initialize the graph edges (initial and final state)
    initial_state = [("D0", [0])]
    final_state = [("K0bar", [0]), ("K+", [0]), ("K-", [0])]

    stm = StateTransitionManager(
        initial_state, final_state, ["a0", "phi", "a2(1320)-"]
    )
    stm.number_of_threads = 1

    graph_interaction_settings_groups = stm.prepare_graphs()
    solutions, _ = stm.find_solutions(graph_interaction_settings_groups)

    print("found " + str(len(solutions)) + " solutions!")
    assert len(solutions) == 5

    # print intermediate state particle names
    for solution in solutions:
        print(solution.edge_props[1]["Name"])

    stm.write_amplitude_model(solutions, "D0ToKs0KpKm.xml")


if __name__ == "__main__":
    test_script()
