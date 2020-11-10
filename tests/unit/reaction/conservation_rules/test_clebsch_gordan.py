from typing import List, Tuple

import pytest

from expertsystem.particle import Spin
from expertsystem.reaction.conservation_rules import (
    IsoSpinEdgeInput,
    SpinEdgeInput,
    SpinNodeInput,
    clebsch_gordan_helicity_to_canonical,
    isospin_conservation,
    spin_conservation,
)

from .test_spin import __create_two_body_decay_spin_data

_SpinRuleInputType = Tuple[
    List[SpinEdgeInput], List[SpinEdgeInput], SpinNodeInput
]


@pytest.mark.parametrize(
    "rule_input, expected",
    [
        (
            __create_two_body_decay_spin_data(
                Spin(1, 0),
                Spin(1, 0),
                Spin(1, 0),
                Spin(1, 0),
                Spin(0, 0),
            ),
            True,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1, 0),
                Spin(1, 0),
                Spin(1, 0),
                Spin(1, 0),
                Spin(1, 0),
            ),
            False,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1, 1),
                Spin(1, 1),
                Spin(1, 0),
                Spin(1, 0),
                Spin(1, 0),
            ),
            False,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1, 1),
                Spin(1, 1),
                Spin(1, 0),
                Spin(1, 0),
                Spin(1, 1),
            ),
            True,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1, 0),
                Spin(1, 1),
                Spin(1, -1),
                Spin(1, 0),
                Spin(1, 0),
            ),
            False,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1, 1),
                Spin(1, 1),
                Spin(1, -1),
                Spin(1, 1),
                Spin(1, 0),
            ),
            True,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(2, 0),
                Spin(0, 0),
                Spin(1, 1),
                Spin(2, 0),
                Spin(1, 0),
            ),
            False,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(3, 0),
                Spin(1, 1),
                Spin(1, -1),
                Spin(2, 0),
                Spin(1, 0),
            ),
            True,
        ),
    ],
)
def test_clebsch_gordan_ls_coupling(
    rule_input: _SpinRuleInputType,
    expected: bool,
):
    assert spin_conservation(*rule_input) is expected


@pytest.mark.parametrize(
    "rule_input, expected",
    [
        (
            __create_two_body_decay_spin_data(
                Spin(1, 0),
                Spin(1, 1),
                Spin(1, 1),
                Spin(0, 0),
                Spin(1, 0),
            ),
            True,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1, 0),
                Spin(1, 1),
                Spin(1, -1),
                Spin(0, 0),
                Spin(1, 0),
            ),
            False,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1, 0),
                Spin(1, 1),
                Spin(1, 1),
                Spin(1, 0),
                Spin(1, 0),
            ),
            False,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1, -1),
                Spin(1, 0),
                Spin(1, 1),
                Spin(1, 0),
                Spin(1, -1),
            ),
            True,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1.5, -0.5),
                Spin(0.5, 0.5),
                Spin(1, 1),
                Spin(1, 0),
                Spin(0.5, -0.5),
            ),
            True,
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1.5, -0.5),
                Spin(0.5, 0.5),
                Spin(1, 1),
                Spin(0, 0),
                Spin(0.5, -0.5),
            ),
            True,
            # This case does actually not work for spin magnitude reasons.
            # However the spin magnitude rule is already filtering out such,
            # cases that the Clebsch Gordan rule does not have to implement,
            # this functionality.
        ),
        (
            __create_two_body_decay_spin_data(
                Spin(1.5, -0.5),
                Spin(0.5, 0.5),
                Spin(1, 1),
                Spin(0, 0),
                Spin(1.5, 0.5),
            ),
            False,
        ),
    ],
)
def test_clebsch_gordan_helicity_canonical(
    rule_input: _SpinRuleInputType,
    expected: bool,
):
    assert clebsch_gordan_helicity_to_canonical(*rule_input) is expected


@pytest.mark.parametrize(
    "coupled_isospin_mag, isospin_mag1, isospin_mag2, expected",
    [
        (1, 1, 1, False),
        (2, 1, 1, True),
        (2, 1, 2, False),
        (3, 1, 2, True),
        (0, 2, 2, True),
        (1, 2, 2, False),
        (2, 2, 2, True),
        (3, 2, 2, False),
    ],
)
def test_isospin_clebsch_gordan_zeros(
    coupled_isospin_mag: int,
    isospin_mag1: int,
    isospin_mag2: int,
    expected: bool,
) -> None:
    assert (
        isospin_conservation(
            [IsoSpinEdgeInput(coupled_isospin_mag, 0)],  # type: ignore
            [
                IsoSpinEdgeInput(isospin_mag1, 0),  # type: ignore
                IsoSpinEdgeInput(isospin_mag2, 0),  # type: ignore
            ],
        )
        is expected
    )
