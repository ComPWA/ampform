from typing import List, Tuple

import pytest

from expertsystem.particle import Spin
from expertsystem.reaction.conservation_rules import (
    IsoSpinConservation,
    IsoSpinEdgeInput,
    SpinConservation,
    SpinConservationMagnitude,
    SpinEdgeInput,
    SpinNodeInput,
)
from expertsystem.reaction.quantum_numbers import (
    EdgeQuantumNumbers,
    NodeQuantumNumbers,
)

_SpinRuleInputType = Tuple[
    List[SpinEdgeInput], List[SpinEdgeInput], SpinNodeInput
]


def __create_two_body_decay_spin_data(
    in_spin: Spin = Spin(0, 0),
    out_spin1: Spin = Spin(0, 0),
    out_spin2: Spin = Spin(0, 0),
    angular_momentum: Spin = Spin(0, 0),
    coupled_spin: Spin = Spin(0, 0),
) -> _SpinRuleInputType:
    return (
        [
            SpinEdgeInput(
                EdgeQuantumNumbers.spin_magnitude(in_spin.magnitude),
                EdgeQuantumNumbers.spin_projection(in_spin.projection),
            )
        ],
        [
            SpinEdgeInput(
                EdgeQuantumNumbers.spin_magnitude(out_spin1.magnitude),
                EdgeQuantumNumbers.spin_projection(out_spin1.projection),
            ),
            SpinEdgeInput(
                EdgeQuantumNumbers.spin_magnitude(out_spin2.magnitude),
                EdgeQuantumNumbers.spin_projection(out_spin2.projection),
            ),
        ],
        SpinNodeInput(
            NodeQuantumNumbers.l_magnitude(angular_momentum.magnitude),
            NodeQuantumNumbers.l_projection(angular_momentum.projection),
            NodeQuantumNumbers.s_magnitude(coupled_spin.magnitude),
            NodeQuantumNumbers.s_projection(coupled_spin.projection),
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
                in_spin=Spin(spin_mag, 0), angular_momentum=Spin(spin_mag, 0)
            ),
            expected,
        )
        for spin_mag, expected in zip([0, 1, 2], [True] * 3)
    ]
    + [
        (
            __create_two_body_decay_spin_data(
                in_spin=Spin(spin_mag, 0),
                out_spin1=Spin(1, -1),
                out_spin2=Spin(1, 1),
                angular_momentum=Spin(1, 0),
                coupled_spin=Spin(spin_mag, 0),
            ),
            expected,
        )
        for spin_mag, expected in [
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
    spin_rule = SpinConservation()

    assert spin_rule(*rule_input) is expected


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
    spin_rule = SpinConservation()

    assert spin_rule(*rule_input) is expected


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
    spin_rule = SpinConservationMagnitude()

    assert spin_rule(*rule_input) is expected  # type: ignore


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
    isospin_rule = IsoSpinConservation()

    assert (
        isospin_rule(
            [IsoSpinEdgeInput(coupled_isospin_mag, 0)],  # type: ignore
            [
                IsoSpinEdgeInput(isospin_mag1, 0),  # type: ignore
                IsoSpinEdgeInput(isospin_mag2, 0),  # type: ignore
            ],
        )
        is expected
    )
