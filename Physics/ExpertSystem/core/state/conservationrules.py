from abc import ABC, abstractmethod
from functools import reduce

from numpy import arange

from core.state.particle import (ParticleQuantumNumberNames,
                                 InteractionQuantumNumberNames,
                                 QNNameClassMapping,
                                 QuantumNumberClasses)


class AbstractRule(ABC):
    def __init__(self):
        self.required_qn_names = []
        self.specify_required_qns()

    @abstractmethod
    def specify_required_qns(self):
        pass

    def add_required_qn(self, qn_name):
        if not (isinstance(qn_name, ParticleQuantumNumberNames) or
                isinstance(qn_name, InteractionQuantumNumberNames)):
            raise TypeError('qn_name has to be of type '
                            + 'ParticleQuantumNumberNames or '
                            + 'InteractionQuantumNumberNames')
        self.required_qn_names.append(qn_name)

    def get_required_qn_names(self):
        return self.required_qn_names

    @abstractmethod
    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        pass


class ChargeConservation(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(ParticleQuantumNumberNames.Charge)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        charge_name = ParticleQuantumNumberNames.Charge

        in_charge = sum([part[charge_name]
                         for part in ingoing_part_qns if charge_name in part])
        out_charge = sum([part[charge_name]
                          for part in outgoing_part_qns if (
                              charge_name in part)])
        return in_charge == out_charge


class ParityConservation(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(ParticleQuantumNumberNames.Parity)
        self.add_required_qn(InteractionQuantumNumberNames.L)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """ implements P_in = P_out * (-1)^L+1 """
        # is this valid for two outgoing particles only?
        parity_label = ParticleQuantumNumberNames.Parity
        parity_in = reduce(
            lambda x, y: x * y[parity_label], ingoing_part_qns, 1)
        parity_out = reduce(
            lambda x, y: x * y[parity_label], outgoing_part_qns, 1)
        ang_mom = interaction_qns[InteractionQuantumNumberNames.L].magnitude()
        if parity_in == parity_out * (-1)**ang_mom:
            return True
        return False


class CParityConservation(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(ParticleQuantumNumberNames.Cparity)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """ implements C_in = C_out """
        # is this valid for two outgoing particles only?
        cparity_label = ParticleQuantumNumberNames.Cparity
        cparity_in = reduce(
            lambda x, y: x * y[cparity_label], ingoing_part_qns, 1)
        cparity_out = reduce(
            lambda x, y: x * y[cparity_label], outgoing_part_qns, 1)
        if cparity_in == cparity_out:
            return True
        return False


class IdenticalParticleSymmetrization(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(ParticleQuantumNumberNames.All)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        if self.check_particles_identical(outgoing_part_qns):
            spin_label = ParticleQuantumNumberNames.Spin
            parity_label = ParticleQuantumNumberNames.Parity

            spin_value = outgoing_part_qns[0][spin_label].magnitude()
            if abs(spin_value % 1) < 0.01:
                # we have a boson, check if parity of mother is even
                parity = ingoing_part_qns[0][parity_label]
                if parity == -1:
                    # if its odd then return False
                    return False
            else:
                # its fermion
                parity = ingoing_part_qns[0][parity_label]
                if parity == 1:
                    return False

        return True

    def check_particles_identical(self, particles):
        reference = particles[0]
        for p in particles[1:]:
            if p != reference:
                return False
        return True


class SpinConservation(AbstractRule):
    """
    Implements conservation of a spin-like quantum number for a two body decay
    (coupling of two particle states). See ::meth::`.check` for details.
    """

    def __init__(self, spinlike_qn, use_projection=True):
        if not isinstance(spinlike_qn, ParticleQuantumNumberNames):
            raise TypeError('Expecting Emum of the type \
                ParticleQuantumNumberNames for spinlike_qn')
        if spinlike_qn not in QNNameClassMapping:
            raise ValueError('spinlike_qn is not associacted with a QN class')
        if QNNameClassMapping[spinlike_qn] is not QuantumNumberClasses.Spin:
            raise ValueError('spinlike_qn is not of class Spin')
        self.spinlike_qn = spinlike_qn
        self.use_projection = use_projection
        super().__init__()

    def specify_required_qns(self):
        self.add_required_qn(self.spinlike_qn)
        # for actual spins we include the angular momentum
        if self.spinlike_qn is ParticleQuantumNumberNames.Spin:
            self.add_required_qn(InteractionQuantumNumberNames.L)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """
        implement  |S1 - S2| <= S <= |S1 + S2|
        and |L - S| <= J <= |L + S| (optionally)
        also checks M1 + M2 == M
        and if clebsch gordan coefficients are 0
        """
        # only valid for two particle decays?
        if len(ingoing_part_qns) == 1 and len(outgoing_part_qns) == 2:
            spin_label = self.spinlike_qn

            in_spins = [x[spin_label] for x in ingoing_part_qns]
            out_spins = [x[spin_label] for x in outgoing_part_qns]
            if (self.use_projection and
                    not self.check_projections(in_spins, out_spins)):
                return False
            if not self.check_magnitude(in_spins, out_spins, interaction_qns):
                return False
            return True
        return False

    def check_projections(self, in_part, out_part):
        in_proj = [x.projection() for x in in_part]
        out_proj = [x.projection() for x in out_part]
        return sum(in_proj) == sum(out_proj)

    def check_magnitude(self, in_part, out_part, interaction_qns):
        spin_mother = in_part[0].magnitude()
        spins_daughters_coupled = self.spin_couplings(
            out_part[0].magnitude(),
            out_part[1].magnitude())
        if InteractionQuantumNumberNames.L in interaction_qns:
            L = interaction_qns[InteractionQuantumNumberNames.L].magnitude()
            for s in spins_daughters_coupled:
                possible_total_spins = self.spin_couplings(
                    s, L)
                if spin_mother in possible_total_spins:
                    return True
        else:
            if spin_mother in spins_daughters_coupled:
                return True
        return False

    def clebsch_gordan_coefficient(self, spin1, spin2, spin_coupled):
        '''
        implement clebsch gordan check
        if spin mother is same as one of daughters 
        and their projections are equal but opposite sign 
        then check if the other daugther has (-1)^(S-M) == -1
        if so then return False
        '''
        pass

    def spin_couplings(self, spin1, spin2):
        """
        implements the coupling of two spins
        |S1 - S2| <= S <= |S1 + S2| and M1 + M2 == M
        """
        return arange(abs(spin1 - spin2), spin1 + spin2 + 1, 1).tolist()


class HelicityConservation(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(ParticleQuantumNumberNames.Spin)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """
        implement |lambda2-lambda3| <= S1
        """
        if len(ingoing_part_qns) == 1 and len(outgoing_part_qns) == 2:
            spin_label = ParticleQuantumNumberNames.Spin

            mother_spin = ingoing_part_qns[0][spin_label].magnitude()
            daughter_hel = [x[spin_label].projection()
                            for x in outgoing_part_qns]
            if mother_spin >= abs(daughter_hel[0] - daughter_hel[1]):
                return True
        return False
