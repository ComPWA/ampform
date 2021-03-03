# pylint: disable=redefined-outer-name

from math import factorial

import pytest

from expertsystem.particle import ParticleCollection
from expertsystem.reaction.combinatorics import (
    _generate_kinematic_permutations,
    _generate_outer_edge_permutations,
    _generate_spin_permutations,
    _get_kinematic_representation,
    _KinematicRepresentation,
    _safe_set_spin_projections,
    create_initial_facts,
)
from expertsystem.reaction.topology import Topology, create_isobar_topologies


@pytest.fixture(scope="session")
def three_body_decay() -> Topology:
    topologies = create_isobar_topologies(3)
    topology = next(iter(topologies))
    return topology


@pytest.mark.parametrize(
    "final_state_groupings",
    [
        ["pi0", "pi0"],
        [["pi0", "pi0"]],
        [[["pi0", "pi0"]]],
        ["gamma", "pi0"],
        [["gamma", "pi0"]],
        [[["gamma", "pi0"]]],
    ],
)
def test_initialize_graph(
    final_state_groupings,
    three_body_decay: Topology,
    particle_database: ParticleCollection,
):
    initial_facts = create_initial_facts(
        three_body_decay,
        initial_state=[("J/psi(1S)", [-1, +1])],
        final_state=["gamma", "pi0", "pi0"],
        particles=particle_database,
        final_state_groupings=final_state_groupings,
    )
    assert len(initial_facts) == 4


@pytest.mark.parametrize(
    "initial_state, final_state",
    [
        (["J/psi(1S)"], ["gamma", "pi0", "pi0"]),
        (["J/psi(1S)"], ["K+", "K-", "pi+", "pi-"]),
        (["e+", "e-"], ["gamma", "pi-", "pi+"]),
        (["e+", "e-"], ["K+", "K-", "pi+", "pi-"]),
    ],
)
def test_generate_outer_edge_permutations(
    initial_state,
    final_state,
    three_body_decay: Topology,
    particle_database: ParticleCollection,
):
    initial_state_with_spins = _safe_set_spin_projections(
        initial_state, particle_database
    )
    final_state_with_spins = _safe_set_spin_projections(
        final_state, particle_database
    )
    list_of_permutations = list(
        _generate_outer_edge_permutations(
            three_body_decay,
            initial_state_with_spins,
            final_state_with_spins,
        )
    )
    n_permutations_final_state = factorial(len(final_state))
    n_permutations_initial_state = factorial(len(initial_state))
    n_permutations = n_permutations_final_state * n_permutations_initial_state
    assert len(list_of_permutations) == n_permutations


class TestKinematicRepresentation:
    @staticmethod
    def test_constructor():
        representation = _KinematicRepresentation(
            initial_state=["J/psi"],
            final_state=["gamma", "pi0"],  # type: ignore
        )
        assert representation.initial_state == [["J/psi"]]
        assert representation.final_state == [["gamma", "pi0"]]
        representation = _KinematicRepresentation([["gamma", "pi0"]])
        assert representation.initial_state is None
        assert representation.final_state == [["gamma", "pi0"]]

    @staticmethod
    def test_from_topology(three_body_decay: Topology):
        pi0 = ("pi0", [0])
        gamma = ("gamma", [-1, 1])
        edge_props = {
            -1: ("J/psi", [-1, +1]),
            0: pi0,
            1: pi0,
            2: gamma,
        }
        kinematic_representation1 = _get_kinematic_representation(
            three_body_decay, edge_props
        )
        assert kinematic_representation1.initial_state == [
            ["J/psi"],
            ["J/psi"],
        ]
        assert kinematic_representation1.final_state == [
            ["gamma", "pi0"],
            ["gamma", "pi0", "pi0"],
        ]

        kinematic_representation2 = _get_kinematic_representation(
            topology=three_body_decay,
            initial_facts={
                -1: ("J/psi", [-1, +1]),
                0: pi0,
                1: gamma,
                2: pi0,
            },
        )
        assert kinematic_representation1 == kinematic_representation2

        kinematic_representation3 = _get_kinematic_representation(
            topology=three_body_decay,
            initial_facts={
                -1: ("J/psi", [-1, +1]),
                0: pi0,
                1: gamma,
                2: gamma,
            },
        )
        assert kinematic_representation2 != kinematic_representation3

    @staticmethod
    def test_repr_and_equality():
        kinematic_representation = _KinematicRepresentation(
            initial_state=[["J/psi"]],
            final_state=[["gamma", "pi0"], ["gamma", "pi0", "pi0"]],
        )
        constructed_from_repr = eval(  # pylint: disable=eval-used
            str(kinematic_representation)
        )
        assert constructed_from_repr == kinematic_representation

    @staticmethod
    def test_in_operator():
        kinematic_representation = _KinematicRepresentation(
            [["gamma", "pi0"], ["gamma", "pi0", "pi0"]],
        )
        subset_representation = _KinematicRepresentation(
            [["gamma", "pi0", "pi0"]],
        )
        assert subset_representation in kinematic_representation
        assert [["J/psi"]] not in kinematic_representation
        assert [["gamma", "pi0"]] in kinematic_representation
        with pytest.raises(ValueError):
            assert float() in kinematic_representation
        with pytest.raises(ValueError):
            assert ["should be nested list"] in kinematic_representation


def test_generate_permutations(
    three_body_decay: Topology, particle_database: ParticleCollection
):
    permutations = _generate_kinematic_permutations(
        three_body_decay,
        initial_state=[("J/psi(1S)", [-1, +1])],
        final_state=["gamma", "pi0", "pi0"],
        particles=particle_database,
        allowed_kinematic_groupings=[_KinematicRepresentation(["pi0", "pi0"])],
    )
    assert len(permutations) == 1

    permutations = _generate_kinematic_permutations(
        three_body_decay,
        initial_state=[("J/psi(1S)", [-1, +1])],
        final_state=["gamma", "pi0", "pi0"],
        particles=particle_database,
    )
    assert len(permutations) == 2
    graph0_final_state_node1 = [
        permutations[0][edge_id]
        for edge_id in three_body_decay.get_originating_final_state_edge_ids(1)
    ]
    graph1_final_state_node1 = [
        permutations[1][edge_id]
        for edge_id in three_body_decay.get_originating_final_state_edge_ids(1)
    ]
    assert graph0_final_state_node1 == [
        ("pi0", [0]),
        ("pi0", [0]),
    ]
    assert graph1_final_state_node1 == [
        ("gamma", [-1, 1]),
        ("pi0", [0]),
    ]

    permutation0 = permutations[0]
    spin_permutations = _generate_spin_permutations(
        permutation0, particle_database
    )
    assert len(spin_permutations) == 4
    assert spin_permutations[0][-1][1] == -1
    assert spin_permutations[0][0][1] == -1
    assert spin_permutations[1][-1][1] == -1
    assert spin_permutations[1][0][1] == +1
    assert spin_permutations[2][-1][1] == +1
    assert spin_permutations[3][-1][1] == +1
