from itertools import product

import pytest

from expertsystem.reaction.conservation_rules import (
    GellMannNishijimaInput,
    gellmann_nishijima,
)
from expertsystem.reaction.quantum_numbers import EdgeQuantumNumbers

# Currently need to cast to the proper Edge/NodeQuantumNumber type, see
# https://github.com/ComPWA/expertsystem/issues/255


@pytest.mark.parametrize(
    "particle, expected",
    [
        (
            GellMannNishijimaInput(
                charge=EdgeQuantumNumbers.charge(charge),
                isospin_proj=EdgeQuantumNumbers.isospin_projection(isospin_z),
                baryon_number=EdgeQuantumNumbers.baryon_number(1),
            ),
            charge == isospin_z + 0.5,
        )
        for charge, isospin_z in product(range(-1, 1), [-1, 0.5, 0, 0.5, 1])
    ]
    + [
        (
            GellMannNishijimaInput(
                charge=EdgeQuantumNumbers.charge(charge),
                isospin_proj=EdgeQuantumNumbers.isospin_projection(isospin_z),
                baryon_number=EdgeQuantumNumbers.baryon_number(1),
                strangeness=EdgeQuantumNumbers.strangeness(strangeness),
            ),
            charge == isospin_z + 0.5 * (1 + strangeness),
        )
        for charge, isospin_z, strangeness in product(
            range(-1, 1), [-1, 0.5, 0, 0.5, 1], [-1, 0, 1]
        )
    ],
)
def test_gellmann_nishijima(particle, expected):
    assert gellmann_nishijima(particle) is expected
