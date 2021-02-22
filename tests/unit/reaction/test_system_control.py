# pylint: disable=protected-access
from copy import deepcopy
from typing import List

import pytest

from expertsystem.particle import Particle
from expertsystem.reaction import (
    InteractionTypes,
    ProblemSet,
    StateTransitionManager,
)
from expertsystem.reaction._system_control import (
    create_edge_properties,
    filter_graphs,
    remove_duplicate_solutions,
    require_interaction_property,
)
from expertsystem.reaction.combinatorics import (
    ParticleWithSpin,
    _create_edge_id_particle_mapping,
    match_external_edges,
    perform_external_edge_identical_particle_combinatorics,
)
from expertsystem.reaction.quantum_numbers import (
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumbers,
)
from expertsystem.reaction.topology import Edge, StateTransitionGraph, Topology


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

    stm.set_allowed_interaction_types([InteractionTypes.STRONG])
    for group in final_state_groupings:
        stm.add_final_state_grouping(group)

    problem_sets = stm.create_problem_sets()
    if problem_sets.values():
        assert len(list(problem_sets.values())[0]) == result_graph_count


@pytest.mark.parametrize(
    "particle_name, spin_projection, expected_properties",
    [
        (
            "pi0",
            0,
            {
                EdgeQuantumNumbers.pid: 111,
                EdgeQuantumNumbers.mass: 0.1349768,
                EdgeQuantumNumbers.width: 7.73e-09,
                EdgeQuantumNumbers.spin_magnitude: 0.0,
                EdgeQuantumNumbers.spin_projection: 0,
                EdgeQuantumNumbers.charge: 0,
                EdgeQuantumNumbers.isospin_magnitude: 1.0,
                EdgeQuantumNumbers.isospin_projection: 0.0,
                EdgeQuantumNumbers.strangeness: 0,
                EdgeQuantumNumbers.charmness: 0,
                EdgeQuantumNumbers.bottomness: 0,
                EdgeQuantumNumbers.topness: 0,
                EdgeQuantumNumbers.baryon_number: 0,
                EdgeQuantumNumbers.electron_lepton_number: 0,
                EdgeQuantumNumbers.muon_lepton_number: 0,
                EdgeQuantumNumbers.tau_lepton_number: 0,
                EdgeQuantumNumbers.parity: -1,
                EdgeQuantumNumbers.c_parity: 1,
                EdgeQuantumNumbers.g_parity: -1,
            },
        ),
        (
            "D+",  # no g and c parity
            0,
            {
                EdgeQuantumNumbers.pid: 411,
                EdgeQuantumNumbers.mass: 1.86965,
                EdgeQuantumNumbers.width: 6.33e-13,
                EdgeQuantumNumbers.spin_magnitude: 0.0,
                EdgeQuantumNumbers.spin_projection: 0,
                EdgeQuantumNumbers.charge: 1,
                EdgeQuantumNumbers.isospin_magnitude: 0.5,
                EdgeQuantumNumbers.isospin_projection: 0.5,
                EdgeQuantumNumbers.strangeness: 0,
                EdgeQuantumNumbers.charmness: 1,
                EdgeQuantumNumbers.bottomness: 0,
                EdgeQuantumNumbers.topness: 0,
                EdgeQuantumNumbers.baryon_number: 0,
                EdgeQuantumNumbers.electron_lepton_number: 0,
                EdgeQuantumNumbers.muon_lepton_number: 0,
                EdgeQuantumNumbers.tau_lepton_number: 0,
                EdgeQuantumNumbers.parity: -1,
                EdgeQuantumNumbers.c_parity: None,
                EdgeQuantumNumbers.g_parity: None,
            },
        ),
        (
            "f(2)(1270)",  # spin projection 1
            1.0,
            {
                EdgeQuantumNumbers.pid: 225,
                EdgeQuantumNumbers.mass: 1.2755,
                EdgeQuantumNumbers.width: 0.18669999999999998,
                EdgeQuantumNumbers.spin_magnitude: 2.0,
                EdgeQuantumNumbers.spin_projection: 1.0,
                EdgeQuantumNumbers.charge: 0,
                EdgeQuantumNumbers.isospin_magnitude: 0.0,
                EdgeQuantumNumbers.isospin_projection: 0.0,
                EdgeQuantumNumbers.strangeness: 0,
                EdgeQuantumNumbers.charmness: 0,
                EdgeQuantumNumbers.bottomness: 0,
                EdgeQuantumNumbers.topness: 0,
                EdgeQuantumNumbers.baryon_number: 0,
                EdgeQuantumNumbers.electron_lepton_number: 0,
                EdgeQuantumNumbers.muon_lepton_number: 0,
                EdgeQuantumNumbers.tau_lepton_number: 0,
                EdgeQuantumNumbers.parity: 1,
                EdgeQuantumNumbers.c_parity: 1,
                EdgeQuantumNumbers.g_parity: 1,
            },
        ),
    ],
)
def test_create_edge_properties(
    particle_name, spin_projection, expected_properties, particle_database
):
    particle = particle_database[particle_name]

    assert (
        create_edge_properties(particle, spin_projection)
        == expected_properties
    )


def make_ls_test_graph(
    angular_momentum_magnitude, coupled_spin_magnitude, particle
):
    graph = StateTransitionGraph[ParticleWithSpin](
        topology=Topology(
            nodes={0},
            edges={0: Edge(None, 0)},
        ),
        node_props={
            0: InteractionProperties(
                s_magnitude=coupled_spin_magnitude,
                l_magnitude=angular_momentum_magnitude,
            )
        },
        edge_props={0: (particle, 0)},
    )
    return graph


def make_ls_test_graph_scrambled(
    angular_momentum_magnitude, coupled_spin_magnitude, particle
):
    graph = StateTransitionGraph[ParticleWithSpin](
        topology=Topology(
            nodes={0},
            edges={0: Edge(None, 0)},
        ),
        node_props={
            0: InteractionProperties(
                l_magnitude=angular_momentum_magnitude,
                s_magnitude=coupled_spin_magnitude,
            )
        },
        edge_props={0: (particle, 0)},
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
    def test_remove_duplicates(self, ls_pairs, result, particle_database):
        pi0 = particle_database["pi0"]
        graphs = []
        for ls_pair in ls_pairs:
            graphs.append(make_ls_test_graph(ls_pair[0], ls_pair[1], pi0))

        results = remove_duplicate_solutions(graphs)
        assert len(results) == result

        for ls_pair in ls_pairs:
            graphs.append(
                make_ls_test_graph_scrambled(ls_pair[0], ls_pair[1], pi0)
            )
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
        self, input_values, filter_parameters, result, particle_database
    ):
        graphs = []
        pi0 = particle_database["pi0"]

        for value in input_values:
            tempgraph = make_ls_test_graph(value[1][0], value[1][1], pi0)
            tempgraph = tempgraph.evolve(
                edge_props={
                    0: (
                        Particle(name=value[0], pid=0, mass=1.0, spin=1.0),
                        0.0,
                    )
                }
            )
            graphs.append(tempgraph)

        my_filter = require_interaction_property(*filter_parameters)
        filtered_graphs = filter_graphs(graphs, [my_filter])
        assert len(filtered_graphs) == result


def _create_graph(
    problem_set: ProblemSet,
) -> StateTransitionGraph[ParticleWithSpin]:
    return StateTransitionGraph[ParticleWithSpin](
        topology=problem_set.topology,
        node_props=problem_set.initial_facts.node_props,
        edge_props=problem_set.initial_facts.edge_props,
    )


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
    stm.set_allowed_interaction_types([InteractionTypes.STRONG])

    problem_sets = stm.create_problem_sets()
    init_graphs: List[StateTransitionGraph[ParticleWithSpin]] = []
    for _, problem_set_list in problem_sets.items():
        init_graphs.extend([_create_graph(x) for x in problem_set_list])

    for graph in init_graphs:
        ref_mapping = _create_edge_id_particle_mapping(
            graph, graph.topology.outgoing_edge_ids
        )
        edge_keys = list(ref_mapping.keys())
        edge1 = edge_keys[0]
        edge1_val = graph.topology.edges[edge1]
        edge1_props = deepcopy(graph.get_edge_props(edge1))
        edge2 = edge_keys[1]
        edge2_val = graph.topology.edges[edge2]
        edge2_props = deepcopy(graph.get_edge_props(edge2))
        graph.swap_edges(edge1, edge2)
        assert graph.topology.edges[edge1] == edge2_val
        assert graph.topology.edges[edge2] == edge1_val
        assert graph.get_edge_props(edge1) == edge2_props
        assert graph.get_edge_props(edge2) == edge1_props


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

    stm.set_allowed_interaction_types([InteractionTypes.STRONG])

    problem_sets = stm.create_problem_sets()
    init_graphs: List[StateTransitionGraph[ParticleWithSpin]] = []
    for _, problem_set_list in problem_sets.items():
        init_graphs.extend([_create_graph(x) for x in problem_set_list])

    match_external_edges(init_graphs)

    iter_graphs = iter(init_graphs)
    first_graph = next(iter_graphs)
    ref_mapping_fs = _create_edge_id_particle_mapping(
        first_graph, first_graph.topology.outgoing_edge_ids
    )
    ref_mapping_is = _create_edge_id_particle_mapping(
        first_graph, first_graph.topology.incoming_edge_ids
    )

    for graph in iter_graphs:
        assert ref_mapping_fs == _create_edge_id_particle_mapping(
            graph, first_graph.topology.outgoing_edge_ids
        )
        assert ref_mapping_is == _create_edge_id_particle_mapping(
            graph, first_graph.topology.incoming_edge_ids
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
    stm.set_allowed_interaction_types([InteractionTypes.STRONG])
    for group in final_state_groupings:
        stm.add_final_state_grouping(group)

    problem_sets = stm.create_problem_sets()

    init_graphs = []
    for _, problem_set_list in problem_sets.items():
        init_graphs.extend([_create_graph(x) for x in problem_set_list])

    match_external_edges(init_graphs)

    comb_graphs: List[StateTransitionGraph[ParticleWithSpin]] = []
    for group in init_graphs:
        comb_graphs.extend(
            perform_external_edge_identical_particle_combinatorics(group)
        )
    assert len(comb_graphs) == result_graph_count

    ref_mapping_fs = _create_edge_id_particle_mapping(
        comb_graphs[0], comb_graphs[0].topology.outgoing_edge_ids
    )
    ref_mapping_is = _create_edge_id_particle_mapping(
        comb_graphs[0], comb_graphs[0].topology.incoming_edge_ids
    )

    for group in comb_graphs[1:]:
        assert ref_mapping_fs == _create_edge_id_particle_mapping(
            group, group.topology.outgoing_edge_ids
        )
        assert ref_mapping_is == _create_edge_id_particle_mapping(
            group, group.topology.incoming_edge_ids
        )
