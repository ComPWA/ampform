from core.state.conservationrules import CParityConservation
from core.state.particle import (ParticleQuantumNumberNames,
                                 XMLLabelConstants,
                                 InteractionQuantumNumberNames,
                                 Spin)


class TestCParity(object):
    def test_cparity_all_defined(self):
        cpar_rule = CParityConservation()
        cparity_label = ParticleQuantumNumberNames.Cparity
        in_part_qns = [
            ([{cparity_label: 1}], True),
            ([{cparity_label: -1}], False)
        ]
        out_part_qns = [
            ([{cparity_label: 1}, {cparity_label: 1}], True),
            ([{cparity_label: -1}, {cparity_label: -1}], True),
            ([{cparity_label: -1}, {cparity_label: 1}], False),
            ([{cparity_label: 1}, {cparity_label: -1}], False)
        ]
        for in_case in in_part_qns:
            for out_case in out_part_qns:
                if in_case[1]:
                    assert cpar_rule.check(
                        in_case[0], out_case[0], []) is out_case[1]
                else:
                    assert cpar_rule.check(
                        in_case[0], out_case[0], []) is not out_case[1]

    def test_cparity_mother_defined(self):
        cpar_rule = CParityConservation()
        cparity_label = ParticleQuantumNumberNames.Cparity
        spin_label = ParticleQuantumNumberNames.Spin
        pid_label = XMLLabelConstants.Pid
        angmom_label = InteractionQuantumNumberNames.L
        in_part_qns = [
            ([{cparity_label: -1}], True),
            ([{cparity_label: 1}], False),
        ]
        out_part_qns = [
            ([{spin_label: 0.0, pid_label: 100}, {
             spin_label: 0.0, pid_label: -100}], True),
        ]
        int_qns = [
            {angmom_label: Spin(1, 0)}
        ]
        for in_case in in_part_qns:
            for out_case in out_part_qns:
                for int_qn in int_qns:
                    if in_case[1]:
                        assert cpar_rule.check(
                            in_case[0], out_case[0], int_qn) is out_case[1]
                    else:
                        assert cpar_rule.check(
                            in_case[0], out_case[0], int_qn) is not out_case[1]
