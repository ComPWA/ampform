# pylint: disable=no-member, no-self-use
from typing import Optional

import pytest
from qrules.particle import Particle
from qrules.quantum_numbers import InteractionProperties
from qrules.topology import create_isobar_topologies

from ampform.helicity.decay import (
    StateWithID,
    TwoBodyDecay,
    determine_attached_final_state,
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
