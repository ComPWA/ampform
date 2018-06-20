import pytest

from expertsystem.ui.system_control import (
    StateTransitionManager, InteractionTypes)


@pytest.mark.parametrize(
    "initial_state,final_state,final_state_groupings,result_graph_count",
    [
        ([("Y", [-1])],
         [("D0", [0]), ("D0bar", [0]),
          ("pi0", [0]), ("pi0", [0])],
         [[['D0', 'pi0'], ['D0bar', 'pi0']]], 1),
        ([("Y", [-1, 1])],
         [("D0", [0]), ("D0bar", [0]),
          ("pi0", [0]), ("pi0", [0])],
         [[['D0', 'pi0'], ['D0bar', 'pi0']]], 2),
        ([("Y", [1])],
         [("D0", [0]), ("D0bar", [0]),
          ("pi0", [0]), ("pi0", [0])],
         [], 9),
        ([("Y", [-1, 1])],
         [("D0", [0]), ("D0bar", [0]),
          ("pi0", [0]), ("pi0", [0])],
         [], 18),
        ([("Y", [1])],
         [("D0", [0]), ("D0bar", [0]),
          ("pi0", [0]), ("pi0", [0])],
         [[['D0', 'pi0'], ['D0bar', 'pi0']], ['D0', 'pi0']], 3),
        ([("J/psi", [-1, 1])],
         [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
         [['pi0', 'pi0']], 4),
        ([("J/psi", [-1, 1])],
         [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
         [], 8),
        ([("J/psi", [-1, 1])],
         [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
         [["pi0", "pi-"]], 0)
    ])
def test_external_edge_initialization(initial_state, final_state,
                                      final_state_groupings,
                                      result_graph_count):
    tbd_manager = StateTransitionManager(initial_state, final_state, [],
                                         formalism_type='helicity')

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    for x in final_state_groupings:
        tbd_manager.add_final_state_grouping(x)
    tbd_manager.number_of_threads = 1

    topology_graphs = tbd_manager.build_topologies()

    init_graphs = tbd_manager.create_seed_graphs(topology_graphs)
    assert len(init_graphs) == result_graph_count
