# pylint: disable=protected-access
import pytest

from expertsystem.particle import Particle
from expertsystem.reaction import InteractionTypes, StateTransitionManager
from expertsystem.reaction._system_control import (
    CompareGraphNodePropertiesFunctor,
    filter_graphs,
    remove_duplicate_solutions,
    require_interaction_property,
)
from expertsystem.reaction.combinatorics import (
    _create_edge_id_particle_mapping,
    match_external_edges,
    perform_external_edge_identical_particle_combinatorics,
)
from expertsystem.reaction.quantum_numbers import (
    InteractionProperties,
    NodeQuantumNumbers,
)
from expertsystem.reaction.topology import StateTransitionGraph


@pytest.mark.parametrize(
    "initial_state,final_state,final_state_groupings,result_graph_count",
    [
        (
            [("Y(4260)", [-1])],
            [("D0", [0]), ("D~0", [0]), ("pi0", [0]), ("pi0", [0])],
            [[["D0", "pi0"], ["D~0", "pi0"]]],
            1,
        ),
        (
            [("Y(4260)", [-1, 1])],
            [("D0", [0]), ("D~0", [0]), ("pi0", [0]), ("pi0", [0])],
            [[["D0", "pi0"], ["D~0", "pi0"]]],
            2,
        ),
        (
            [("Y(4260)", [1])],
            [("D0", [0]), ("D~0", [0]), ("pi0", [0]), ("pi0", [0])],
            [],
            9,
        ),
        (
            [("Y(4260)", [-1, 1])],
            [("D0", [0]), ("D~0", [0]), ("pi0", [0]), ("pi0", [0])],
            [],
            18,
        ),
        (
            [("Y(4260)", [1])],
            [("D0", [0]), ("D~0", [0]), ("pi0", [0]), ("pi0", [0])],
            [[["D0", "pi0"], ["D~0", "pi0"]], ["D0", "pi0"]],
            3,
        ),
        (
            [("J/psi(1S)", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "pi0"]],
            4,
        ),
        (
            [("J/psi(1S)", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "gamma"]],
            4,
        ),
        (
            [("J/psi(1S)", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [],
            8,
        ),
        (
            [("J/psi(1S)", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "pi-"]],
            0,
        ),
    ],
)
def test_external_edge_initialization(
    particle_database,
    initial_state,
    final_state,
    final_state_groupings,
    result_graph_count,
):
    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        formalism_type="helicity",
        number_of_threads=1,
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    for group in final_state_groupings:
        stm.add_final_state_grouping(group)

    topology_graphs = stm._build_topologies()

    init_graphs = stm._create_seed_graphs(topology_graphs)
    assert len(init_graphs) == result_graph_count


def make_ls_test_graph(angular_momentum_magnitude, coupled_spin_magnitude):
    graph = StateTransitionGraph[dict]()
    graph.graph_node_properties_comparator = (
        CompareGraphNodePropertiesFunctor()
    )
    graph.add_node(0)
    graph.node_props[0] = InteractionProperties(
        s_magnitude=coupled_spin_magnitude,
        l_magnitude=angular_momentum_magnitude,
    )
    return graph


def make_ls_test_graph_scrambled(
    angular_momentum_magnitude, coupled_spin_magnitude
):
    graph = StateTransitionGraph[dict]()
    graph.graph_node_properties_comparator = (
        CompareGraphNodePropertiesFunctor()
    )
    graph.add_node(0)
    graph.node_props[0] = InteractionProperties(
        l_magnitude=angular_momentum_magnitude,
        s_magnitude=coupled_spin_magnitude,
    )
    return graph


class TestSolutionFilter:  # pylint: disable=no-self-use
    @pytest.mark.parametrize(
        "ls_pairs, result",
        [
            ([(1, 0), (1, 1)], 2),
            ([(1, 0), (1, 0)], 1),
        ],
    )
    def test_remove_duplicates(self, ls_pairs, result):
        graphs: list = []
        for ls_pair in ls_pairs:
            graphs.append(make_ls_test_graph(ls_pair[0], ls_pair[1]))

        results = remove_duplicate_solutions(graphs)
        assert len(results) == result

        for ls_pair in ls_pairs:
            graphs.append(make_ls_test_graph_scrambled(ls_pair[0], ls_pair[1]))
        results = remove_duplicate_solutions(graphs)
        assert len(results) == result

    @pytest.mark.parametrize(
        "input_values,filter_parameters,result",
        [
            (
                [("foo", (1, 0)), ("foo", (1, 1))],
                (
                    "foo",
                    NodeQuantumNumbers.l_magnitude,
                    [1],
                ),
                2,
            ),
            (
                [("foo", (1, 0)), ("foo", (2, 1))],
                (
                    "foo",
                    NodeQuantumNumbers.l_magnitude,
                    [1],
                ),
                1,
            ),
            (
                [("foo", (1, 0)), ("foo", (1, 1))],
                (
                    "foo~",
                    NodeQuantumNumbers.l_magnitude,
                    [1],
                ),
                0,
            ),
            (
                [("foo", (0, 0)), ("foo", (1, 1)), ("foo", (2, 1))],
                (
                    "foo",
                    NodeQuantumNumbers.l_magnitude,
                    [1, 2],
                ),
                2,
            ),
            (
                [("foo", (1, 0)), ("foo", (1, 1))],
                (
                    "foo",
                    NodeQuantumNumbers.s_magnitude,
                    [1],
                ),
                1,
            ),
        ],
    )
    def test_filter_graphs_for_interaction_qns(
        self, input_values, filter_parameters, result
    ):
        graphs = []

        for value in input_values:
            tempgraph = make_ls_test_graph(value[1][0], value[1][1])
            tempgraph.add_edges([0])
            tempgraph.attach_edges_to_node_ingoing([0], 0)
            tempgraph.edge_props[0] = (
                Particle(name=value[0], pid=0, mass=1.0, spin=1.0),
                0.0,
            )
            graphs.append(tempgraph)

        my_filter = require_interaction_property(*filter_parameters)
        filtered_graphs = filter_graphs(graphs, [my_filter])
        assert len(filtered_graphs) == result


@pytest.mark.parametrize(
    "initial_state,final_state",
    [
        (
            [("Y(4260)", [-1])],
            [("D0", [0]), ("D~0", [0]), ("pi0", [0]), ("pi0", [0])],
        ),
    ],
)
def test_edge_swap(particle_database, initial_state, final_state):
    # pylint: disable=too-many-locals
    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        formalism_type="helicity",
        number_of_threads=1,
    )
    stm.set_allowed_interaction_types([InteractionTypes.Strong])

    topology_graphs = stm._build_topologies()
    init_graphs = stm._create_seed_graphs(topology_graphs)

    for graph in init_graphs:
        ref_mapping = _create_edge_id_particle_mapping(
            graph, "get_final_state_edges"
        )
        edge_keys = list(ref_mapping.keys())
        edge1 = edge_keys[0]
        edge1_val = graph.edges[edge1]
        edge1_props = graph.edge_props[edge1]
        edge2 = edge_keys[1]
        edge2_val = graph.edges[edge2]
        edge2_props = graph.edge_props[edge2]
        graph.swap_edges(edge1, edge2)
        assert graph.edges[edge1] == edge2_val
        assert graph.edges[edge2] == edge1_val
        assert graph.edge_props[edge1] == edge2_props
        assert graph.edge_props[edge2] == edge1_props


@pytest.mark.parametrize(
    "initial_state,final_state",
    [
        (
            [("Y(4260)", [-1])],
            [("D0", [0]), ("D~0", [0]), ("pi0", [0]), ("pi0", [0])],
        ),
        (
            [("J/psi(1S)", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
        ),
    ],
)
def test_match_external_edges(particle_database, initial_state, final_state):
    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        formalism_type="helicity",
        number_of_threads=1,
    )

    stm.set_allowed_interaction_types([InteractionTypes.Strong])

    topology_graphs = stm._build_topologies()
    init_graphs = stm._create_seed_graphs(topology_graphs)

    match_external_edges(init_graphs)

    ref_mapping_fs = _create_edge_id_particle_mapping(
        init_graphs[0], "get_final_state_edges"
    )
    ref_mapping_is = _create_edge_id_particle_mapping(
        init_graphs[0], "get_initial_state_edges"
    )

    for graph in init_graphs[1:]:
        assert ref_mapping_fs == _create_edge_id_particle_mapping(
            graph, "get_final_state_edges"
        )
        assert ref_mapping_is == _create_edge_id_particle_mapping(
            graph, "get_initial_state_edges"
        )


@pytest.mark.parametrize(
    "initial_state,final_state,final_state_groupings,result_graph_count",
    [
        (
            [("Y(4260)", [1])],
            [("D0", [0]), ("D~0", [0]), ("pi0", [0]), ("pi0", [0])],
            [[["D0", "pi0"], ["D~0", "pi0"]]],
            2,
        ),
        (
            [("Y(4260)", [1])],
            [("D0", [0]), ("D~0", [0]), ("pi0", [0]), ("pi0", [0])],
            [["D0", "pi0"]],
            6,
        ),
        (
            [("J/psi(1S)", [1])],
            [("gamma", [1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "pi0"]],
            1,
        ),
        (
            [("J/psi(1S)", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [],
            12,
        ),
        (
            [("J/psi(1S)", [1])],
            [("gamma", [1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "gamma"]],
            2,
        ),
    ],
)
def test_external_edge_identical_particle_combinatorics(
    particle_database,
    initial_state,
    final_state,
    final_state_groupings,
    result_graph_count,
):
    stm = StateTransitionManager(
        initial_state,
        final_state,
        particle_database,
        formalism_type="helicity",
        number_of_threads=1,
    )
    stm.set_allowed_interaction_types([InteractionTypes.Strong])
    for group in final_state_groupings:
        stm.add_final_state_grouping(group)

    topology_graphs = stm._build_topologies()

    init_graphs = stm._create_seed_graphs(topology_graphs)

    match_external_edges(init_graphs)

    comb_graphs = []
    for group in init_graphs:
        comb_graphs.extend(
            perform_external_edge_identical_particle_combinatorics(group)
        )
    assert len(comb_graphs) == result_graph_count

    ref_mapping_fs = _create_edge_id_particle_mapping(
        comb_graphs[0], "get_final_state_edges"
    )
    ref_mapping_is = _create_edge_id_particle_mapping(
        comb_graphs[0], "get_initial_state_edges"
    )

    for group in comb_graphs[1:]:
        assert ref_mapping_fs == _create_edge_id_particle_mapping(
            group, "get_final_state_edges"
        )
        assert ref_mapping_is == _create_edge_id_particle_mapping(
            group, "get_initial_state_edges"
        )
