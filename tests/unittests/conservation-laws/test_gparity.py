from itertools import (product)

from pycompwa.expertsystem.state.conservationrules import GParityConservation
from pycompwa.expertsystem.state.particle import (StateQuantumNumberNames,
                                         ParticlePropertyNames,
                                         InteractionQuantumNumberNames,
                                         Spin)


class TestGParity(object):
    def test_gparity_all_defined(self):
        gpar_rule = GParityConservation()
        gparity_label = StateQuantumNumberNames.Gparity
        in_part_qns = [
            ([{gparity_label: 1}], True),
            ([{gparity_label: -1}], False)
        ]
        out_part_qns = [
            ([{gparity_label: 1}, {gparity_label: 1}], True),
            ([{gparity_label: -1}, {gparity_label: -1}], True),
            ([{gparity_label: -1}, {gparity_label: 1}], False),
            ([{gparity_label: 1}, {gparity_label: -1}], False)
        ]
        for in_case in in_part_qns:
            for out_case in out_part_qns:
                if in_case[1]:
                    assert gpar_rule.check(
                        in_case[0], out_case[0], []) is out_case[1]
                else:
                    assert gpar_rule.check(
                        in_case[0], out_case[0], []) is not out_case[1]

    def test_gparity_multiparticle_boson(self):
        gpar_rule = GParityConservation()
        gparity_label = StateQuantumNumberNames.Gparity
        spin_label = StateQuantumNumberNames.Spin
        pid_label = ParticlePropertyNames.Pid
        isospin_label = StateQuantumNumberNames.IsoSpin
        angmom_label = InteractionQuantumNumberNames.L
        cases = []

        for ang_mom_case in [([0, 2, 4], True), ([1, 3], False)]:
            for ang_mom in ang_mom_case[0]:
                temp_case = ([{gparity_label: 1, isospin_label: Spin(0, 0)}],
                             [{spin_label: Spin(0, 0), pid_label: 100}, {
                                 spin_label: Spin(0, 0), pid_label: -100}],
                             {angmom_label: Spin(ang_mom, 0)},
                             ang_mom_case[1])

                cases.append(temp_case)
                cases.append(
                    ([{gparity_label: -1, isospin_label: Spin(0, 0)}],
                     temp_case[1], temp_case[2], not temp_case[3]))

        for ang_mom_case in [([0, 2, 4], False), ([1, 3], True)]:
            for ang_mom in ang_mom_case[0]:
                temp_case = ([{gparity_label: 1, isospin_label: Spin(1, 0)}],
                             [{spin_label: Spin(0, 0), pid_label: 100}, {
                                 spin_label: Spin(0, 0), pid_label: -100}],
                             {angmom_label: Spin(ang_mom, 0)},
                             ang_mom_case[1])

                cases.append(temp_case)
                cases.append(
                    ([{gparity_label: -1, isospin_label: Spin(1, 0)}],
                     temp_case[1], temp_case[2], not temp_case[3]))

        for ang_mom in [0, 1, 2, 3]:
            temp_case = ([{gparity_label: 1}],
                         [{spin_label: Spin(0, 0), pid_label: 100}, {
                             spin_label: Spin(0, 0), pid_label: 100}],
                         {angmom_label: Spin(ang_mom, 0)},
                         True)

            cases.append(temp_case)
            cases.append(
                ([{gparity_label: None}], temp_case[1],
                    temp_case[2], True))

        for case in cases:
            assert gpar_rule.check(
                case[0], case[1], case[2]) is case[3]

    '''def test_cparity_multiparticle_fermion(self):
        cpar_rule = CParityConservation()
        cparity_label = StateQuantumNumberNames.Cparity
        spin_label = StateQuantumNumberNames.Spin
        pid_label = ParticlePropertyNames.Pid
        angmom_label = InteractionQuantumNumberNames.L
        intspin_label = InteractionQuantumNumberNames.S
        cases = []

        all_spin_cases = list(product(range(0, 5), range(0, 5)))
        even_sum_cases = [x for x in all_spin_cases if (x[0] + x[1]) % 2 == 0]
        odd_sum_cases = [x for x in all_spin_cases if (x[0] + x[1]) % 2 != 0]
        for spin_case in [(even_sum_cases, True), (odd_sum_cases, False)]:
            for spin_ang_mom in spin_case[0]:
                temp_case = ([{cparity_label: 1}],
                             [{spin_label: Spin(0.5, 0), pid_label: 100}, {
                                 spin_label: Spin(0.5, 0), pid_label: -100}],
                             {angmom_label: Spin(spin_ang_mom[1], 0),
                              intspin_label: Spin(spin_ang_mom[0], 0)},
                             spin_case[1])

                cases.append(temp_case)
                cases.append(
                    ([{cparity_label: -1}], temp_case[1],
                     temp_case[2], not temp_case[3]))

        for ang_mom in [0, 1, 2, 3]:
            temp_case = ([{cparity_label: None}],
                         [{spin_label: 0.0, pid_label: 100}, {
                             spin_label: 0.0, pid_label: 100}],
                         {angmom_label: Spin(0, 0),
                             intspin_label: Spin(ang_mom, 0)},
                         True)

            cases.append(temp_case)
            cases.append(
                ([{cparity_label: None}], temp_case[1],
                    temp_case[2], True))

        for case in cases:
            assert cpar_rule.check(
                case[0], case[1], case[2]) is case[3]'''
