from typing import Any

from expertsystem.data import Spin
from expertsystem.state.conservation_rules import SpinConservation
from expertsystem.state.particle import (
    InteractionQuantumNumberNames,
    StateQuantumNumberNames,
)


class TestSpin:  # pylint: disable=no-self-use
    def test_spin_all_defined(self):
        spin_label = StateQuantumNumberNames.Spin
        ang_mom_label = InteractionQuantumNumberNames.L
        intspin_label = InteractionQuantumNumberNames.S
        spin_rule = SpinConservation(spin_label)
        cases: Any = []
        case: Any = None

        for case in [([0], True), ([1, 2, 3], False)]:
            for spin_mag in case[0]:
                temp_case = (
                    [{spin_label: Spin(0, 0)}],
                    [{spin_label: Spin(0, 0)}, {spin_label: Spin(0, 0)}],
                    {
                        ang_mom_label: Spin(spin_mag, 0),
                        intspin_label: Spin(0, 0),
                    },
                    case[1],
                )
                cases.append(temp_case)

        for case in [([0, 1, 2], True)]:
            for spin_mag in case[0]:
                temp_case = (
                    [{spin_label: Spin(spin_mag, 0)}],
                    [{spin_label: Spin(0, 0)}, {spin_label: Spin(0, 0)}],
                    {
                        ang_mom_label: Spin(spin_mag, 0),
                        intspin_label: Spin(0, 0),
                    },
                    case[1],
                )
                cases.append(temp_case)

        for case in [([0, 1, 2], True), ([3], False)]:
            for spin_mag in case[0]:
                temp_case = (
                    [{spin_label: Spin(spin_mag, 0)}],
                    [{spin_label: Spin(1, -1)}, {spin_label: Spin(1, 1)}],
                    {
                        ang_mom_label: Spin(0, 0),
                        intspin_label: Spin(spin_mag, 0),
                    },
                    case[1],
                )
                cases.append(temp_case)

        for case in [
            (
                Spin(1, -1),
                Spin(0, 0),
                Spin(1, -1),
                Spin(0, 0),
                Spin(1, -1),
                True,
            )
        ]:
            temp_case = (
                [{spin_label: case[0]}],
                [{spin_label: case[1]}, {spin_label: case[2]}],
                {ang_mom_label: case[3], intspin_label: case[4]},
                case[5],
            )
            cases.append(temp_case)

        for case in cases:
            assert spin_rule.check(case[0], case[1], case[2]) is case[3]

    def test_spin_ignore_z_component(self):
        spin_label = StateQuantumNumberNames.Spin
        ang_mom_label = InteractionQuantumNumberNames.L
        intspin_label = InteractionQuantumNumberNames.S
        spin_rule = SpinConservation(spin_label, False)
        cases = []
        case: Any = None

        for case in [([(0, 0, 1), (2, 1, 2), (2, 1, 1)], True)]:
            for spin_mag in case[0]:
                temp_case = (
                    [{spin_label: Spin(1, 1)}],
                    [
                        {spin_label: Spin(spin_mag[0], 0)},
                        {spin_label: Spin(1, -1)},
                    ],
                    {
                        ang_mom_label: Spin(spin_mag[1], 0),
                        intspin_label: Spin(spin_mag[2], -1),
                    },
                    case[1],
                )
                cases.append(temp_case)

        for case in cases:
            assert spin_rule.check(case[0], case[1], case[2]) is case[3]

    def test_isospin_clebsch_gordan_zeros(self):
        spin_label = StateQuantumNumberNames.IsoSpin
        spin_rule = SpinConservation(spin_label)
        cases = []

        for case in [
            (1, 1, 1, False),
            (2, 1, 1, True),
            (2, 1, 2, False),
            (3, 1, 2, True),
            (0, 2, 2, True),
            (1, 2, 2, False),
            (2, 2, 2, True),
            (3, 2, 2, False),
        ]:
            temp_case: Any = (
                [{spin_label: Spin(case[0], 0)}],
                [
                    {spin_label: Spin(case[1], 0)},
                    {spin_label: Spin(case[2], 0)},
                ],
                {},
                case[3],
            )
            cases.append(temp_case)

        for case in cases:
            assert spin_rule.check(case[0], case[1], case[2]) is case[3]
