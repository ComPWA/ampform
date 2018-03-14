from itertools import (product)

from core.state.conservationrules import SpinConservation
from core.state.particle import (StateQuantumNumberNames,
                                 XMLLabelConstants,
                                 InteractionQuantumNumberNames,
                                 Spin,
                                 create_spin_domain)


class TestSpin(object):
    def test_spin_all_defined(self):
        spin_label = StateQuantumNumberNames.Spin
        angmom_label = InteractionQuantumNumberNames.L
        intspin_label = InteractionQuantumNumberNames.S
        spin_rule = SpinConservation(spin_label)
        cases = []

        for case in [([0, 1, 2], True)]:
            for spin_mag in case[0]:
                temp_case = ([{spin_label: Spin(spin_mag, 0)}],
                             [{spin_label: Spin(0, 0)}, {
                                 spin_label: Spin(0, 0)}],
                             {angmom_label: Spin(spin_mag, 0),
                              intspin_label: Spin(0, 0)},
                             case[1])
                cases.append(temp_case)

        for case in cases:
            assert spin_rule.check(
                case[0], case[1], case[2]) is case[3]
