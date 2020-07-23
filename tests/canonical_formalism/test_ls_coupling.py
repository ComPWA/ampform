import pytest

from expertsystem.state import particle
from expertsystem.state.conservation_rules import ParityConservationHelicity
from expertsystem.state.particle import (
    InteractionQuantumNumberNames,
    Spin,
    _SpinQNConverter,
)
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
            [("Y", [1])],
            [("D*(2007)0bar", [0]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(0, 0),
            1,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [1]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(0, 0),
            0,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [1]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(1, 0),
            1,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [0]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(1, 0),
            0,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [1]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(2, 0),
            1,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [0]), ("D*(2007)0", [0])],
            Spin(1, 0),
            Spin(2, 0),
            1,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [0]), ("D*(2007)0", [1])],
            Spin(1, 0),
            Spin(0, 0),
            0,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [0]), ("D*(2007)0", [1])],
            Spin(1, 0),
            Spin(1, 0),
            1,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [0]), ("D*(2007)0", [1])],
            Spin(1, 0),
            Spin(2, 0),
            1,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [0]), ("D*(2007)0", [-1])],
            Spin(1, 0),
            Spin(0, 0),
            0,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [0]), ("D*(2007)0", [-1])],
            Spin(1, 0),
            Spin(1, 0),
            1,
        ),
        (
            [("Y", [1])],
            [("D*(2007)0bar", [0]), ("D*(2007)0", [-1])],
            Spin(1, 0),
            Spin(2, 0),
            1,
        ),  # pylint: disable=too-many-locals
    ],
)
def test_canonical_clebsch_gordan_ls_coupling(
    initial_state, final_state, ang_mom, spin, solution_count
):
    # because the amount of solutions is too big we change the default domains
    formalism_type = "canonical-helicity"
    int_settings = create_default_interaction_settings(formalism_type)

    _remove_conservation_law(
        int_settings[InteractionTypes.Strong], ParityConservationHelicity()
    )

    tbd_manager = StateTransitionManager(
        initial_state,
        final_state,
        [],
        interaction_type_settings=int_settings,
        formalism_type=formalism_type,
    )

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    tbd_manager.number_of_threads = 2
    tbd_manager.filter_remove_qns = []

    l_label = InteractionQuantumNumberNames.L
    s_label = InteractionQuantumNumberNames.S
    qn_label = particle.Labels.QuantumNumber

    spin_converter = _SpinQNConverter()
    node_props = {
        0: {
            qn_label.name: [
                spin_converter.convert_to_dict(l_label, ang_mom),
                spin_converter.convert_to_dict(s_label, spin),
            ]
        }
    }
    graph_node_setting_pairs = tbd_manager.prepare_graphs()
    for value in graph_node_setting_pairs.values():
        for edge in value:
            edge[0].node_props = node_props

    solutions = tbd_manager.find_solutions(graph_node_setting_pairs)[0]

    assert len(solutions) == solution_count
