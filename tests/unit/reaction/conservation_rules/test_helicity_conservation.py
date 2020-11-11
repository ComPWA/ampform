from itertools import product

import pytest

from expertsystem.reaction.conservation_rules import helicity_conservation
from expertsystem.reaction.quantum_numbers import EdgeQuantumNumbers


@pytest.mark.parametrize(
    "in_edge_qns, out_edge_qns, expected",
    [
        (
            [
                EdgeQuantumNumbers.spin_magnitude(s_mag),
            ],
            [
                EdgeQuantumNumbers.spin_projection(lambda1),
                EdgeQuantumNumbers.spin_projection(lambda2),
            ],
            abs(lambda1 - lambda2) <= s_mag,
        )
        for s_mag, lambda1, lambda2 in product(
            [0, 0.5, 1, 1.5, 2],
            [-2, -1.5, -1.0, -0.5, 0, 0.5, 1, 1.5, 2],
            [-1, 0, 1],
        )
    ],
)
def test_helicity_conservation(in_edge_qns, out_edge_qns, expected):
    assert helicity_conservation(in_edge_qns, out_edge_qns) is expected
