from typing import List, Optional, Tuple

import pytest

from expertsystem.reaction.conservation_rules import (
    SpinEdgeInput,
    SpinNodeInput,
    spin_conservation,
    spin_magnitude_conservation,
)
from expertsystem.reaction.particle import Spin

_SpinRuleInputType = Tuple[
    List[SpinEdgeInput], List[SpinEdgeInput], SpinNodeInput
]


def __create_two_body_decay_spin_data(
    in_spin: Optional[Spin] = None,
    out_spin1: Optional[Spin] = None,
    out_spin2: Optional[Spin] = None,
    angular_momentum: Optional[Spin] = None,
    coupled_spin: Optional[Spin] = None,
) -> _SpinRuleInputType:
    spin_zero = Spin(0, 0)
    if in_spin is None:
        in_spin = spin_zero
    if out_spin1 is None:
        out_spin1 = spin_zero
    if out_spin2 is None:
        out_spin2 = spin_zero
    if angular_momentum is None:
        angular_momentum = spin_zero
    if coupled_spin is None:
        coupled_spin = spin_zero
    return (
        [SpinEdgeInput(in_spin.magnitude, in_spin.projection)],
        [
            SpinEdgeInput(out_spin1.magnitude, out_spin1.projection),
            SpinEdgeInput(out_spin2.magnitude, out_spin2.projection),
        ],
        SpinNodeInput(
            angular_momentum.magnitude,
            angular_momentum.projection,
            coupled_spin.magnitude,
            coupled_spin.projection,
        ),
    )


@pytest.mark.parametrize(
    "rule_input, expected",
    [
        (
            __create_two_body_decay_spin_data(
                angular_momentum=Spin(ang_mom_mag, 0)
            ),
            expected,
        )
        for ang_mom_mag, expected in [
            (0, True),
            (1, False),
            (2, False),
            (3, False),
        ]
    ]
    + [
        (
            __create_two_body_decay_spin_data(
                in_spin=Spin(spin_magnitude, 0),
                angular_momentum=Spin(spin_magnitude, 0),
            ),
            expected,
        )
        for spin_magnitude, expected in zip([0, 1, 2], [True] * 3)
    ]
    + [
        (
            __create_two_body_decay_spin_data(
                in_spin=Spin(spin_magnitude, 0),
                out_spin1=Spin(1, -1),
                out_spin2=Spin(1, 1),
                angular_momentum=Spin(1, 0),
                coupled_spin=Spin(spin_magnitude, 0),
            ),
            expected,
        )
        for spin_magnitude, expected in [
            (0, False),
            (1, False),
            (2, False),
            (3, False),
        ]
    ]
    + [
        (
            __create_two_body_decay_spin_data(
                in_spin=Spin(1, -1),
                out_spin2=Spin(1, -1),
                coupled_spin=Spin(1, -1),
            ),
            True,
        ),
        (
            __create_two_body_decay_spin_data(
                in_spin=Spin(1, 0),
                out_spin1=Spin(1, 1),
                out_spin2=Spin(1, -1),
                angular_momentum=Spin(1, 0),
                coupled_spin=Spin(2, 0),
            ),
            True,
        ),
    ],
)
def test_spin_all_defined(
    rule_input: _SpinRuleInputType, expected: bool
) -> None:
    assert spin_conservation(*rule_input) is expected


@pytest.mark.parametrize(
    "rule_input, expected",
    [
        (
            __create_two_body_decay_spin_data(
                in_spin=Spin(1, 1),
                out_spin1=Spin(spin2_mag, 0),
                out_spin2=Spin(1, -1),
                angular_momentum=Spin(ang_mom_mag, 0),
                coupled_spin=Spin(coupled_spin_mag, -1),
            ),
            True,
        )
        for spin2_mag, ang_mom_mag, coupled_spin_mag in zip(
            (0, 0, 1), (2, 1, 2), (1, 1, 2)
        )
    ]
    + [
        (
            __create_two_body_decay_spin_data(
                in_spin=Spin(1, 1),
                out_spin1=Spin(spin2_mag, 0),
                out_spin2=Spin(1, -1),
                angular_momentum=Spin(ang_mom_mag, 0),
                coupled_spin=Spin(coupled_spin_mag, 0),
            ),
            False,
        )
        for spin2_mag, ang_mom_mag, coupled_spin_mag in zip(
            (1, 0, 1), (0, 1, 2), (0, 2, 0)
        )
    ],
)
def test_spin_ignore_z_component(
    rule_input: _SpinRuleInputType, expected: bool
) -> None:
    assert spin_magnitude_conservation(*rule_input) is expected  # type: ignore
