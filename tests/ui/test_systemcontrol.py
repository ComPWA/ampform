import pytest

from expertsystem.state import particle
from expertsystem.state.particle import (
    InteractionQuantumNumberNames,
    create_spin_domain,
)
from expertsystem.topology.graph import (
    StateTransitionGraph,
    get_final_state_edges,
    get_initial_state_edges,
)
from expertsystem.ui.system_control import (
    CompareGraphElementPropertiesFunctor,
    InteractionTypes,
    StateTransitionManager,
    create_edge_id_particle_mapping,
    filter_graphs,
    match_external_edges,
    perform_external_edge_identical_particle_combinatorics,
    remove_duplicate_solutions,
    require_interaction_property,
)


@pytest.mark.parametrize(
    "initial_state,final_state,final_state_groupings,result_graph_count",
    [
        (
            [("Y", [-1])],
            [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])],
            [[["D0", "pi0"], ["D0bar", "pi0"]]],
            1,
        ),
        (
            [("Y", [-1, 1])],
            [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])],
            [[["D0", "pi0"], ["D0bar", "pi0"]]],
            2,
        ),
        (
            [("Y", [1])],
            [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])],
            [],
            9,
        ),
        (
            [("Y", [-1, 1])],
            [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])],
            [],
            18,
        ),
        (
            [("Y", [1])],
            [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])],
            [[["D0", "pi0"], ["D0bar", "pi0"]], ["D0", "pi0"]],
            3,
        ),
        (
            [("J/psi", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "pi0"]],
            4,
        ),
        (
            [("J/psi", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "gamma"]],
            4,
        ),
        (
            [("J/psi", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [],
            8,
        ),
        (
            [("J/psi", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "pi-"]],
            0,
        ),
    ],
)
def test_external_edge_initialization(
    initial_state, final_state, final_state_groupings, result_graph_count
):
    tbd_manager = StateTransitionManager(
        initial_state, final_state, [], formalism_type="helicity"
    )

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    for group in final_state_groupings:
        tbd_manager.add_final_state_grouping(group)
    tbd_manager.number_of_threads = 1

    topology_graphs = tbd_manager.build_topologies()

    init_graphs = tbd_manager.create_seed_graphs(topology_graphs)
    assert len(init_graphs) == result_graph_count


def make_ls_test_graph(angular_momentum_magnitude, coupled_spin_magnitude):
    graph = StateTransitionGraph()
    graph.set_graph_element_properties_comparator(
        CompareGraphElementPropertiesFunctor()
    )
    graph.nodes.append(0)
    graph.node_props[0] = {
        "QuantumNumber": [
            {
                "Value": str(coupled_spin_magnitude),
                "Type": "S",
                "Projection": "0.0",
                "Class": "Spin",
            },
            {
                "Value": str(angular_momentum_magnitude),
                "Type": "L",
                "Projection": "0.0",
                "Class": "Spin",
            },
        ]
    }
    return graph


def make_ls_test_graph_scrambled(
    angular_momentum_magnitude, coupled_spin_magnitude
):
    graph = StateTransitionGraph()
    graph.set_graph_element_properties_comparator(
        CompareGraphElementPropertiesFunctor()
    )
    graph.nodes.append(0)
    graph.node_props[0] = {
        "QuantumNumber": [
            {
                "Class": "Spin",
                "Value": str(angular_momentum_magnitude),
                "Type": "L",
                "Projection": "0.0",
            },
            {
                "Projection": "0.0",
                "Class": "Spin",
                "Value": str(coupled_spin_magnitude),
                "Type": "S",
            },
        ]
    }
    return graph


class TestSolutionFilter:  # pylint: disable=no-self-use
    @pytest.mark.parametrize(
        "ls_pairs, result", [([(1, 0), (1, 1)], 2), ([(1, 0), (1, 0)], 1),]
    )
    def test_remove_duplicates(self, ls_pairs, result):
        graphs = {"test": []}
        for ls_pair in ls_pairs:
            graphs["test"].append(
                ([make_ls_test_graph(ls_pair[0], ls_pair[1])], [])
            )

        results = remove_duplicate_solutions(graphs)
        num_solutions = [len(result[0]) for result in results["test"]]
        assert sum(num_solutions) == result

        for ls_pair in ls_pairs:
            graphs["test"].append(
                ([make_ls_test_graph_scrambled(ls_pair[0], ls_pair[1])], [])
            )
        results = remove_duplicate_solutions(graphs)
        num_solutions = [len(result[0]) for result in results["test"]]
        assert sum(num_solutions) == result

    @pytest.mark.parametrize(
        "input_values,filter_parameters,result",
        [
            (
                [("foo", (1, 0)), ("foo", (1, 1))],
                (
                    "foo",
                    InteractionQuantumNumberNames.L,
                    create_spin_domain([1], True),
                ),
                2,
            ),
            (
                [("foo", (1, 0)), ("foo", (2, 1))],
                (
                    "foo",
                    InteractionQuantumNumberNames.L,
                    create_spin_domain([1], True),
                ),
                1,
            ),
            (
                [("foo", (1, 0)), ("foo", (1, 1))],
                (
                    "foobar",
                    InteractionQuantumNumberNames.L,
                    create_spin_domain([1], True),
                ),
                0,
            ),
            (
                [("foo", (0, 0)), ("foo", (1, 1)), ("foo", (2, 1))],
                (
                    "foo",
                    InteractionQuantumNumberNames.L,
                    create_spin_domain([1, 2], True),
                ),
                2,
            ),
            (
                [("foo", (1, 0)), ("foo", (1, 1))],
                (
                    "foo",
                    InteractionQuantumNumberNames.S,
                    create_spin_domain([1], True),
                ),
                1,
            ),
        ],
    )
    def test_filter_graphs_for_interaction_qns(
        self, input_values, filter_parameters, result
    ):
        graphs = []
        name_label = particle.Labels.Name.name
        value_label = particle.Labels.Value.name
        for value in input_values:
            tempgraph = make_ls_test_graph(value[1][0], value[1][1])
            tempgraph.add_edges([0])
            tempgraph.attach_edges_to_node_ingoing([0], 0)
            tempgraph.edge_props[0] = {name_label: {value_label: value[0]}}
            graphs.append(tempgraph)

        my_filter = require_interaction_property(*filter_parameters)
        filtered_graphs = filter_graphs(graphs, [my_filter])
        assert len(filtered_graphs) == result


@pytest.mark.parametrize(
    "initial_state,final_state",
    [
        (
            [("Y", [-1])],
            [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])],
        ),
    ],
)
def test_edge_swap(initial_state, final_state):
    tbd_manager = StateTransitionManager(
        initial_state, final_state, [], formalism_type="helicity"
    )

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    tbd_manager.number_of_threads = 1

    topology_graphs = tbd_manager.build_topologies()
    init_graphs = tbd_manager.create_seed_graphs(topology_graphs)

    for graph in init_graphs:
        ref_mapping = create_edge_id_particle_mapping(
            graph, get_final_state_edges
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
            [("Y", [-1])],
            [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])],
        ),
        (
            [("J/psi", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
        ),
    ],
)
def test_match_external_edges(initial_state, final_state):
    tbd_manager = StateTransitionManager(
        initial_state, final_state, [], formalism_type="helicity"
    )

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    tbd_manager.number_of_threads = 1

    topology_graphs = tbd_manager.build_topologies()
    init_graphs = tbd_manager.create_seed_graphs(topology_graphs)

    match_external_edges(init_graphs)

    ref_mapping_fs = create_edge_id_particle_mapping(
        init_graphs[0], get_final_state_edges
    )
    ref_mapping_is = create_edge_id_particle_mapping(
        init_graphs[0], get_initial_state_edges
    )

    for graph in init_graphs[1:]:
        assert ref_mapping_fs == create_edge_id_particle_mapping(
            graph, get_final_state_edges
        )
        assert ref_mapping_is == create_edge_id_particle_mapping(
            graph, get_initial_state_edges
        )


@pytest.mark.parametrize(
    "initial_state,final_state,final_state_groupings,result_graph_count",
    [
        (
            [("Y", [1])],
            [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])],
            [[["D0", "pi0"], ["D0bar", "pi0"]]],
            2,
        ),
        (
            [("Y", [1])],
            [("D0", [0]), ("D0bar", [0]), ("pi0", [0]), ("pi0", [0])],
            [["D0", "pi0"]],
            6,
        ),
        (
            [("J/psi", [1])],
            [("gamma", [1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "pi0"]],
            1,
        ),
        (
            [("J/psi", [-1, 1])],
            [("gamma", [-1, 1]), ("pi0", [0]), ("pi0", [0])],
            [],
            12,
        ),
        (
            [("J/psi", [1])],
            [("gamma", [1]), ("pi0", [0]), ("pi0", [0])],
            [["pi0", "gamma"]],
            2,
        ),
    ],
)
def test_external_edge_identical_particle_combinatorics(
    initial_state, final_state, final_state_groupings, result_graph_count
):
    tbd_manager = StateTransitionManager(
        initial_state, final_state, [], formalism_type="helicity"
    )

    tbd_manager.set_allowed_interaction_types([InteractionTypes.Strong])
    for group in final_state_groupings:
        tbd_manager.add_final_state_grouping(group)
    tbd_manager.number_of_threads = 1

    topology_graphs = tbd_manager.build_topologies()

    init_graphs = tbd_manager.create_seed_graphs(topology_graphs)
    match_external_edges(init_graphs)

    comb_graphs = []
    for group in init_graphs:
        comb_graphs.extend(
            perform_external_edge_identical_particle_combinatorics(group)
        )
    assert len(comb_graphs) == result_graph_count

    ref_mapping_fs = create_edge_id_particle_mapping(
        comb_graphs[0], get_final_state_edges
    )
    ref_mapping_is = create_edge_id_particle_mapping(
        comb_graphs[0], get_initial_state_edges
    )

    for group in comb_graphs[1:]:
        assert ref_mapping_fs == create_edge_id_particle_mapping(
            group, get_final_state_edges
        )
        assert ref_mapping_is == create_edge_id_particle_mapping(
            group, get_initial_state_edges
        )
