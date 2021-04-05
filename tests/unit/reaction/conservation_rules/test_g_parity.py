from itertools import product

import pytest

from expertsystem.reaction.conservation_rules import (
    GParityEdgeInput,
    GParityNodeInput,
    g_parity_conservation,
)
from expertsystem.reaction.quantum_numbers import Parity

# Currently need to cast to the proper Edge/NodeQuantumNumber type, see
# https://github.com/ComPWA/expertsystem/issues/255


@pytest.mark.parametrize(
    "rule_input, expected",
    [
        (
            (
                [
                    GParityEdgeInput(
                        isospin_magnitude=0,
                        spin_magnitude=0,
                        pid=123,
                        g_parity=Parity(g_parity_in[0]),
                    )
                ],
                [
                    GParityEdgeInput(
                        isospin_magnitude=0,
                        spin_magnitude=0,
                        pid=0,
                        g_parity=Parity(g_parity_out[0][0]),
                    ),
                    GParityEdgeInput(
                        isospin_magnitude=0,
                        spin_magnitude=0,
                        pid=0,
                        g_parity=Parity(g_parity_out[0][1]),
                    ),
                ],
                GParityNodeInput(l_magnitude=0, s_magnitude=0),
            ),
            g_parity_in[1] is g_parity_out[1],
        )
        for g_parity_in, g_parity_out in product(
            [
                (1, True),
                (-1, False),
            ],
            [
                ((1, 1), True),
                ((-1, -1), True),
                ((-1, 1), False),
                ((1, -1), False),
            ],
        )
    ],
)
def test_g_parity_all_defined(rule_input, expected):
    assert g_parity_conservation(*rule_input) is expected


@pytest.mark.parametrize(
    "rule_input, expected",
    [
        (
            (
                [
                    GParityEdgeInput(
                        isospin_magnitude=isospin,
                        spin_magnitude=0,
                        pid=123,
                        g_parity=Parity(g_parity),
                    )
                ],
                [
                    GParityEdgeInput(
                        isospin_magnitude=0,
                        spin_magnitude=0,
                        pid=100,
                    ),
                    GParityEdgeInput(
                        isospin_magnitude=0,
                        spin_magnitude=0,
                        pid=-100,
                    ),
                ],
                GParityNodeInput(l_magnitude=l_magnitude, s_magnitude=0),
            ),
            (-1) ** (l_magnitude + isospin) == g_parity,
        )
        for g_parity, isospin, l_magnitude in product(
            [-1, 1], [0, 1], range(0, 5)
        )
    ],
)
def test_g_parity_multiparticle_boson(rule_input, expected):
    assert g_parity_conservation(*rule_input) is expected
