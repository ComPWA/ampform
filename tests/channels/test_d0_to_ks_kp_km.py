""" sample script for the testing purposes using the decay
    D0 -> K~0 K+ K-
"""

import logging

from expertsystem.ui import StateTransitionManager

logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def test_script():
    stm = StateTransitionManager(
        initial_state=[("D0", [0])],
        final_state=[("K~0", [0]), ("K+", [0]), ("K-", [0])],
        allowed_intermediate_particles=[
            "a(0)(980)",
            "a(2)(1320)-",
            "phi(1020)",
        ],
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
