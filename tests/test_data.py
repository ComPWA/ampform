# pylint: disable=no-member,no-self-use
import numpy as np
import pytest

from ampform.data import (
    EventCollection,
    FourMomentumSequence,
    MatrixSequence,
    ThreeMomentum,
)


class TestFourMomentumSequence:
    def test_properties(self):
        sample = FourMomentumSequence(
            [
                [0, 1, 2, 3],
            ]
        )
        assert len(sample) == 1
        assert sample.energy == 0
        assert sample.p_x == 1
        assert sample.p_y == 2
        assert sample.p_z == 3
        assert np.all(sample.three_momentum == ThreeMomentum([[1, 2, 3]]))
        assert sample.p_norm()[0] == [np.sqrt(1 ** 2 + 2 ** 2 + 3 ** 2)]

    @pytest.mark.parametrize(
        ("state_id", "expected_mass"),
        [
            (0, 0.13498),
            (1, 0.00048 + 0.00032j),
            (2, 0.13498),
            (3, 0.13498),
        ],
    )
    def test_mass(
        self,
        data_sample: EventCollection,
        state_id: int,
        expected_mass: float,
    ):
        four_momenta = data_sample[state_id]
        inv_mass = four_momenta.mass()
        assert len(inv_mass) == 10
        average_mass = np.average(inv_mass)
        assert pytest.approx(average_mass, abs=1e-5) == expected_mass

    def test_phi(self):
        vector = FourMomentumSequence(np.array([[0, 1, 1, 0]]))
        assert pytest.approx(vector.phi()) == np.pi / 4

    def test_theta(self):
        vector = FourMomentumSequence(np.array([[0, 0, 1, 1]]))
        assert pytest.approx(vector.theta()) == np.pi / 4
        vector = FourMomentumSequence(np.array([[0, 1, 0, 1]]))
        assert pytest.approx(vector.theta()) == np.pi / 4


class TestMatrixSequence:
    def test_init(self):
        n_events = 10
        zeros = np.zeros(n_events)
        ones = np.ones(n_events)
        twos = 2 * ones
        threes = 3 * ones
        matrices = MatrixSequence(
            np.array(
                [
                    [twos, zeros, zeros, twos],
                    [zeros, ones, zeros, zeros],
                    [zeros, threes, ones, zeros],
                    [zeros, zeros, zeros, threes],
                ]
            ).transpose(2, 0, 1)
        )
        assert pytest.approx(matrices[0]) == np.array(
            [
                [2, 0, 0, 2],
                [0, 1, 0, 0],
                [0, 3, 1, 0],
                [0, 0, 0, 3],
            ]
        )
