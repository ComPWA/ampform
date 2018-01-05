from abc import ABC, abstractmethod
from functools import reduce

from numpy import arange

from core.state.particle import (ParticleQuantumNumberNames,
                                 InteractionQuantumNumberNames,
                                 QuantumNumberClasses,
                                 QNNameClassMapping,
                                 XMLLabelConstants,
                                 get_attributes_for_qn)


class AbstractRule(ABC):
    def __init__(self):
        self.required_qn_names = {}
        self.specify_required_qns()

    @abstractmethod
    def specify_required_qns(self):
        pass

    def add_required_qn(self, qn_name, qn_attr=[XMLLabelConstants.Value]):
        if not (isinstance(qn_name, ParticleQuantumNumberNames) or
                isinstance(qn_name, InteractionQuantumNumberNames)):
            raise TypeError('qn_name has to be of type '
                            + 'ParticleQuantumNumberNames or '
                            + 'InteractionQuantumNumberNames')
        if not qn_attr:
            raise ValueError('qn_attr has to be a non-empty list')

        for qn_att in qn_attr:
            if (qn_att is not XMLLabelConstants.Value and
                    qn_att not in [
                        x[0] for x in get_attributes_for_qn(qn_name)]):
                raise TypeError('quantum number attribute ' + str(qn_att)
                                + ' is not valid for the qn ' + str(qn_name))
            if qn_name not in self.required_qn_names:
                self.required_qn_names[qn_name] = []
            self.required_qn_names[qn_name].append(qn_att)

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
        value_label = XMLLabelConstants.Value
        in_charge = sum([part[charge_name][value_label]
                         for part in ingoing_part_qns if charge_name in part])
        out_charge = sum([part[charge_name][value_label]
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
        value_label = XMLLabelConstants.Value
        parity_in = reduce(
            lambda x, y: x * y[parity_label][value_label], ingoing_part_qns, 1)
        parity_out = reduce(
            lambda x, y: x * y[parity_label][value_label], outgoing_part_qns, 1)
        ang_mom = interaction_qns[InteractionQuantumNumberNames.L][value_label]
        if parity_in == parity_out * (-1)**ang_mom:
            return True
        return False


class IdenticalParticleSymmetrization(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(ParticleQuantumNumberNames.All)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        if self.check_particles_identical(outgoing_part_qns):
            spin_label = ParticleQuantumNumberNames.Spin
            parity_label = ParticleQuantumNumberNames.Parity
            value_label = XMLLabelConstants.Value
            spin_value = str(outgoing_part_qns[0][spin_label])
            if ('.' and '/') not in spin_value:
                # we have a boson, check if parity of mother is even
                parity = ingoing_part_qns[0][parity_label][value_label]
                if parity == -1:
                    # if its odd then return False
                    return False
            else:
                # its fermion
                parity = ingoing_part_qns[0][parity_label][value_label]
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
        qn_attr = [XMLLabelConstants.Value]
        if self.use_projection:
            qn_attr.append(XMLLabelConstants.Projection)
        self.add_required_qn(self.spinlike_qn, qn_attr)
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
        proj_label = XMLLabelConstants.Projection
        in_proj = [x[proj_label] for x in in_part]
        out_proj = [x[proj_label] for x in out_part]
        return sum(in_proj) == sum(out_proj)

    def check_magnitude(self, in_part, out_part, interaction_qns):
        value_label = XMLLabelConstants.Value
        spin_mother = in_part[0][value_label]
        spins_daughters_coupled = self.spin_couplings(
            out_part[0][value_label],
            out_part[1][value_label])
        if InteractionQuantumNumberNames.L in interaction_qns:
            L = interaction_qns[InteractionQuantumNumberNames.L][XMLLabelConstants.Value]
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

    def spin_couplings(self, spin1, spin2):
        """
        implements the coupling of two spins
        |S1 - S2| <= S <= |S1 + S2| and M1 + M2 == M
        """
        return arange(abs(spin1 - spin2), spin1 + spin2 + 1, 1).tolist()


class HelicityConservation(AbstractRule):
    def specify_required_qns(self):
        qn_attr = [XMLLabelConstants.Value,
                   XMLLabelConstants.Projection]
        self.add_required_qn(ParticleQuantumNumberNames.Spin, qn_attr)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """
        implement |lambda2-lambda3| <= S1
        """
        # only valid for two particles?
        return True
