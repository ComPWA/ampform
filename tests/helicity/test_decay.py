# pylint: disable=no-member, no-self-use
import itertools
from typing import Iterable, Optional, Set

import pytest
from qrules.particle import Particle
from qrules.quantum_numbers import InteractionProperties
from qrules.topology import Topology, create_isobar_topologies

from ampform.helicity.decay import (
    StateWithID,
    TwoBodyDecay,
    determine_attached_final_state,
    get_sibling_state_id,
    is_opposite_helicity_state,
)


def _create_dummy_decay(
    l_magnitude: Optional[int], spin_magnitude: float
) -> TwoBodyDecay:
    dummy = Particle(name="dummy", pid=123, spin=spin_magnitude, mass=1.0)
    return TwoBodyDecay(
        parent=StateWithID(
            id=0, particle=dummy, spin_projection=spin_magnitude
        ),
        children=(
            StateWithID(id=1, particle=dummy, spin_projection=0.0),
            StateWithID(id=2, particle=dummy, spin_projection=0.0),
        ),
        interaction=InteractionProperties(l_magnitude=l_magnitude),
    )


class TestTwoBodyDecay:
    @pytest.mark.parametrize(
        ("decay", "expected_l"),
        [
            (_create_dummy_decay(1, 0.5), 1),
            (_create_dummy_decay(0, 1.0), 0),
            (_create_dummy_decay(2, 1.0), 2),
            (_create_dummy_decay(None, 0.0), 0),
            (_create_dummy_decay(None, 1.0), 1),
        ],
    )
    def test_extract_angular_momentum(
        self, decay: TwoBodyDecay, expected_l: int
    ):
        assert expected_l == decay.extract_angular_momentum()

    @pytest.mark.parametrize(
        "decay",
        [
            _create_dummy_decay(None, 0.5),
            _create_dummy_decay(None, 1.5),
        ],
    )
    def test_invalid_angular_momentum(self, decay: TwoBodyDecay):
        with pytest.raises(ValueError, match="not integral"):
            decay.extract_angular_momentum()


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
