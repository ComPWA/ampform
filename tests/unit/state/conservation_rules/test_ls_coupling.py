import pytest

from expertsystem.data import NodeQuantumNumbers, Spin
from expertsystem.solving.conservation_rules import ParityConservationHelicity
from expertsystem.ui import (
    InteractionTypes,
    StateTransitionManager,
)
from expertsystem.ui._default_settings import (
    create_default_interaction_settings,
)
from expertsystem.ui._system_control import _remove_conservation_law


@pytest.mark.parametrize(
    "initial_state, final_state, ang_mom, spin, solution_count",
    [
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [0]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(0, 0),
            1,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [1]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(0, 0),
            0,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [1]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(1, 0),
            1,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [0]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(1, 0),
            0,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [1]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(2, 0),
            1,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [0]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(2, 0),
            1,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [0]), ("D*(2007)0", [1])],
            Spin(1, 0),
            Spin(0, 0),
            0,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [0]), ("D*(2007)0", [1])],
            Spin(1, 0),
            Spin(1, 0),
            1,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [0]), ("D*(2007)0", [1])],
            Spin(1, 0),
            Spin(2, 0),
            1,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [0]), ("D*(2007)0", [-1])],
            Spin(1, 0),
            Spin(0, 0),
            0,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [0]), ("D*(2007)0", [-1])],
            Spin(1, 0),
            Spin(1, 0),
            1,
        ),
        (
            [("Y(4260)", [1])],
            [("D*(2007)~0", [0]), ("D*(2007)0", [-1])],
            Spin(1, 0),
            Spin(2, 0),
            1,
        ),  # pylint: disable=too-many-locals
    ],
)
def test_canonical_clebsch_gordan_ls_coupling(  # pylint: disable=too-many-arguments
    initial_state: list,
    final_state: list,
    ang_mom: Spin,
    spin: Spin,
    solution_count: int,
):
    # because the amount of solutions is too big we change the default domains
    formalism_type = "canonical-helicity"
    int_settings = create_default_interaction_settings(formalism_type)

    _remove_conservation_law(
        int_settings[InteractionTypes.Strong], ParityConservationHelicity()
    )

    stm = StateTransitionManager(
        initial_state,
        final_state,
        interaction_type_settings=int_settings,
        formalism_type=formalism_type,
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    stm.number_of_threads = 2
    stm.filter_remove_qns = set()

    node_props = {
        0: {
            NodeQuantumNumbers.l_magnitude: ang_mom.magnitude,
            NodeQuantumNumbers.l_projection: ang_mom.projection,
            NodeQuantumNumbers.s_magnitude: spin.magnitude,
            NodeQuantumNumbers.s_projection: spin.projection,
        }
    }
    graph_node_setting_pairs = stm.prepare_graphs()
    for graph_node_settings in graph_node_setting_pairs.values():
        for graph, _ in graph_node_settings:
            graph.node_props = node_props

    result = stm.find_solutions(graph_node_setting_pairs)

    assert len(result.solutions) == solution_count
