from itertools import product

import pytest

from expertsystem.particle import Parity
from expertsystem.reaction.conservation_rules import (
    HelicityParityEdgeInput,
    parity_conservation,
    parity_conservation_helicity,
)
from expertsystem.reaction.quantum_numbers import (
    EdgeQuantumNumbers,
    NodeQuantumNumbers,
)

# Currently need to cast to the proper Edge/NodeQuantumNumber type, see
# https://github.com/ComPWA/expertsystem/issues/255


@pytest.mark.parametrize(
    "in_parities, out_parities, l_mag, expected",
    [
        (
            [
                EdgeQuantumNumbers.parity(Parity(parity_in)),
            ],
            [
                EdgeQuantumNumbers.parity(Parity(parity_out1)),
                EdgeQuantumNumbers.parity(Parity(1)),
            ],
            NodeQuantumNumbers.l_magnitude(l_mag),
            parity_in
            == parity_out1
            * (-1) ** (l_mag),  # pylint: disable=undefined-variable
        )
        for parity_in, parity_out1, l_mag in product(
            [-1, 1], [-1, 1], range(0, 5)
        )
    ],
)
def test_parity_conservation(in_parities, out_parities, l_mag, expected):
    assert parity_conservation(in_parities, out_parities, l_mag) is expected


@pytest.mark.parametrize(
    "in_parities, out_parities, l_mag, expected",
    [
        (
            [
                HelicityParityEdgeInput(
                    parity=EdgeQuantumNumbers.parity(Parity(in_parity)),
                    spin_mag=EdgeQuantumNumbers.spin_magnitude(in_spin_mag),
                    spin_proj=EdgeQuantumNumbers.spin_projection(0),
                )
            ],
            [
                HelicityParityEdgeInput(
                    parity=EdgeQuantumNumbers.parity(Parity(out_parity1)),
                    spin_mag=EdgeQuantumNumbers.spin_magnitude(1),
                    spin_proj=EdgeQuantumNumbers.spin_projection(-1),
                ),
                HelicityParityEdgeInput(
                    parity=EdgeQuantumNumbers.parity(Parity(out_parity2)),
                    spin_mag=EdgeQuantumNumbers.spin_magnitude(1),
                    spin_proj=EdgeQuantumNumbers.spin_projection(-1),
                ),
            ],
            NodeQuantumNumbers.parity_prefactor(1),
            in_parity * out_parity1 * out_parity2 * (-1) ** (in_spin_mag % 2)
            == 1,
        )
        for in_spin_mag, in_parity, out_parity1, out_parity2 in product(
            range(0, 4), [-1, 1], [-1, 1], [-1, 1]
        )
    ],
)
def test_parity_conservation_helicity_prefactor(
    in_parities, out_parities, l_mag, expected
):
    assert (
        parity_conservation_helicity(in_parities, out_parities, l_mag)
        is expected
    )


@pytest.mark.parametrize(
    "in_parities, out_parities, l_mag, expected",
    [
        (
            [
                HelicityParityEdgeInput(
                    parity=EdgeQuantumNumbers.parity(Parity(in_parity)),
                    spin_mag=EdgeQuantumNumbers.spin_magnitude(in_spin_mag),
                    spin_proj=EdgeQuantumNumbers.spin_projection(0),
                )
            ],
            [
                HelicityParityEdgeInput(
                    parity=EdgeQuantumNumbers.parity(Parity(1)),
                    spin_mag=EdgeQuantumNumbers.spin_magnitude(1),
                    spin_proj=EdgeQuantumNumbers.spin_projection(0),
                ),
                HelicityParityEdgeInput(
                    parity=EdgeQuantumNumbers.parity(Parity(1)),
                    spin_mag=EdgeQuantumNumbers.spin_magnitude(1),
                    spin_proj=EdgeQuantumNumbers.spin_projection(0),
                ),
            ],
            NodeQuantumNumbers.parity_prefactor(parity_prefactor),
            in_parity * (-1) ** (in_spin_mag % 2) == 1
            and parity_prefactor == 1,
        )
        for in_spin_mag, in_parity, parity_prefactor in product(
            range(0, 4), [-1, 1], [-1, 1]
        )
    ],
)
def test_parity_conservation_helicity(
    in_parities, out_parities, l_mag, expected
):
    assert (
        parity_conservation_helicity(in_parities, out_parities, l_mag)
        is expected
    )
