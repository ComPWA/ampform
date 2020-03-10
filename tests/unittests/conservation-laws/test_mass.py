from pycompwa.expertsystem.state.conservationrules import MassConservation
from pycompwa.expertsystem.state.particle import (
    ParticlePropertyNames, ParticleDecayPropertyNames)


class TestMass(object):
    def test_mass_two_body_decay_stable_outgoing(self):
        mass_label = ParticlePropertyNames.Mass
        width_label = ParticleDecayPropertyNames.Width
        mass_rule = MassConservation(5)
        cases = []

        # we assume a two charged pion final state here
        # units are always in GeV
        for case in [([(0.280, 0.0), (0.260, 0.010), (0.300, 0.05)], True),
                     ([(0.270, 0.0), (0.250, 0.005), (0.200, 0.01)], False)]:
            for in_mass_case in case[0]:
                temp_case = ([{mass_label: in_mass_case[0],
                               width_label: in_mass_case[1]}],
                             [{mass_label: 0.139},
                              {mass_label: 0.139}],
                             {},
                             case[1])
                cases.append(temp_case)

        for case in cases:
            assert mass_rule.check(
                case[0], case[1], case[2]) is case[3]
