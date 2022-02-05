# pylint: disable=no-member, no-self-use
import itertools
from typing import Iterable, Set

import pytest
from qrules.topology import Topology, create_isobar_topologies

from ampform.helicity.decay import (
    determine_attached_final_state,
    get_sibling_state_id,
    is_opposite_helicity_state,
)


def test_determine_attached_final_state():
    topologies = create_isobar_topologies(4)
    # outer states
    for topology in topologies:
        for i in topology.outgoing_edge_ids:
            assert determine_attached_final_state(topology, state_id=i) == [i]
        for i in topology.incoming_edge_ids:
            assert determine_attached_final_state(
                topology, state_id=i
            ) == list(topology.outgoing_edge_ids)
    # intermediate states
    topology = topologies[0]
    assert determine_attached_final_state(topology, state_id=4) == [0, 1]
    assert determine_attached_final_state(topology, state_id=5) == [2, 3]
    topology = topologies[1]
    assert determine_attached_final_state(topology, state_id=4) == [1, 2, 3]
    assert determine_attached_final_state(topology, state_id=5) == [2, 3]


@pytest.mark.parametrize("n_final_states", [2, 3, 4, 5, 6])
def test_is_opposite_helicity_state_0_is_never_opposite(n_final_states):
    topologies = create_isobar_topologies(n_final_states)
    permutated_topologies = __permutate_topologies(topologies)
    for topology in permutated_topologies:
        assert is_opposite_helicity_state(topology, state_id=0) is False


@pytest.mark.parametrize("n_final_states", [2, 3, 4, 5, 6])
def test_is_opposite_helicity_state_state_sibling_is_opposite(n_final_states):
    topologies = create_isobar_topologies(n_final_states)
    permutated_topologies = __permutate_topologies(topologies)
    for topology in permutated_topologies:
        for state_id in topology.edges:
            if state_id in topology.incoming_edge_ids:
                continue
            sibling_id = get_sibling_state_id(topology, state_id)
            assert is_opposite_helicity_state(
                topology,
                state_id,
            ) != is_opposite_helicity_state(
                topology,
                sibling_id,
            )


def __permutate_topologies(topologies: Iterable[Topology]) -> Set[Topology]:
    permutated_topologies = set()
    for topology in topologies:
        permutated_topologies.update(__permutate_final_state_ids(topology))
    return permutated_topologies


def __permutate_final_state_ids(topology: Topology) -> Set[Topology]:
    permutated_topologies = set()
    final_state_ids = sorted(topology.outgoing_edge_ids)
    for permutation in itertools.permutations(final_state_ids):
        renames = dict(zip(permutation, final_state_ids))
        permutated_topologies.add(topology.relabel_edges(renames))
    return permutated_topologies
