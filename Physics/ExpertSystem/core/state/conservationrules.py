from abc import ABC, abstractmethod
from functools import reduce

from numpy import arange

from core.state.particle import (StateQuantumNumberNames,
                                 InteractionQuantumNumberNames,
                                 ParticlePropertyNames,
                                 QNNameClassMapping,
                                 QuantumNumberClasses,
                                 get_xml_label,
                                 XMLLabelConstants,
                                 is_boson)


''' Functors for quantum number condition checks '''


class AbstractConditionFunctor(ABC):
    @abstractmethod
    def check(self, qn_names, in_edges, out_edges, int_node):
        pass


class DefinedForAllEdges(AbstractConditionFunctor):
    def check(self, qn_names, in_edges, out_edges, int_node):
        for qn_name in qn_names:
            for edge in (in_edges + out_edges):
                if qn_name not in edge:
                    return False
        return True


class DefinedForAllOutgoingEdges(AbstractConditionFunctor):
    def check(self, qn_names, in_edges, out_edges, int_node):
        for qn_name in qn_names:
            for edge in out_edges:
                if qn_name not in edge:
                    return False
        return True


class DefinedForInteractionNode(AbstractConditionFunctor):
    def check(self, qn_names, in_edges, out_edges, int_node):
        for qn_name in qn_names:
            if qn_name not in int_node:
                return False
        return True


class DefinedIfOtherQnNotDefinedInOutSeperate(AbstractConditionFunctor):
    '''
    Implements logic for...
    '''

    def __init__(self, other_qn_names):
        self.other_qn_names = other_qn_names

    def check(self, qn_names, in_edges, out_edges, int_node):
        return (self.check_edge_set(qn_names, in_edges, int_node) and
                self.check_edge_set(qn_names, out_edges, int_node))

    def check_edge_set(self, qn_names, edges, int_node):
        found_for_all = True
        for qn_name in self.other_qn_names:
            if isinstance(qn_name, (StateQuantumNumberNames,
                                    ParticlePropertyNames)):
                found_for_all = True
                for edge_props in edges:
                    if not self.find_in_dict(qn_name, edge_props):
                        found_for_all = False
                        break
                if not found_for_all:
                    break
            else:
                if not self.find_in_dict(qn_name, int_node):
                    found_for_all = False
                    break

        if not found_for_all:
            for qn_name in qn_names:
                if isinstance(qn_name, (StateQuantumNumberNames,
                                        ParticlePropertyNames)):
                    for edge_props in edges:
                        if not self.find_in_dict(qn_name, edge_props):
                            return False
                else:
                    if not self.find_in_dict(qn_name, int_node):
                        return False
        return True

    def find_in_dict(self, name, props):
        found = False
        for ele in props:
            if name == ele:
                found = True
                break
        return found


class AbstractRule(ABC):
    def __init__(self):
        self.required_qn_names = []
        self.qn_conditions = []
        self.specify_required_qns()

    def get_qn_conditions(self):
        return self.qn_conditions

    @abstractmethod
    def specify_required_qns(self):
        pass

    def add_required_qn(self, qn_name, qn_condition_functions=[]):
        if not (isinstance(qn_name, StateQuantumNumberNames) or
                isinstance(qn_name, InteractionQuantumNumberNames) or
                isinstance(qn_name, ParticlePropertyNames)):
            raise TypeError('qn_name has to be of type '
                            + 'ParticleQuantumNumberNames or '
                            + 'InteractionQuantumNumberNames or '
                            + 'ParticlePropertyNames')
        self.required_qn_names.append(qn_name)
        for cond in qn_condition_functions:
            self.qn_conditions.append(([qn_name], cond))

    def get_required_qn_names(self):
        return self.required_qn_names

    @abstractmethod
    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        pass


class AdditiveQuantumNumberConservation(AbstractRule):
    """
    implements sum(Q_in) == sum(Q_out), which is used for etc
    electric charge, baryon number, lepton number conservation
    """

    def __init__(self, qn_name):
        self.qn_name = qn_name
        super().__init__()

    def specify_required_qns(self):
        self.add_required_qn(self.qn_name, [DefinedForAllEdges()])

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        in_qn_sum = sum([part[self.qn_name]
                         for part in ingoing_part_qns if self.qn_name in part])
        out_qn_sum = sum([part[self.qn_name]
                          for part in outgoing_part_qns if (
            self.qn_name in part)])
        return in_qn_sum == out_qn_sum


class ParityConservation(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(
            StateQuantumNumberNames.Parity, [DefinedForAllEdges()])
        self.add_required_qn(InteractionQuantumNumberNames.L)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """ implements P_in = P_out * (-1)^L+1 """
        # is this valid for two outgoing particles only?
        parity_label = StateQuantumNumberNames.Parity
        parity_in = reduce(
            lambda x, y: x * y[parity_label], ingoing_part_qns, 1)
        parity_out = reduce(
            lambda x, y: x * y[parity_label], outgoing_part_qns, 1)
        ang_mom = interaction_qns[InteractionQuantumNumberNames.L].magnitude()
        if parity_in == (parity_out * (-1)**ang_mom):
            return True
        return False


class CParityConservation(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(StateQuantumNumberNames.Cparity)
        # the spin quantum number is required to check if the daughter
        # particles are fermions or bosons
        self.add_required_qn(StateQuantumNumberNames.Spin,
                             [DefinedIfOtherQnNotDefinedInOutSeperate(
                                 [StateQuantumNumberNames.Cparity])])
        self.add_required_qn(InteractionQuantumNumberNames.L,
                             [DefinedIfOtherQnNotDefinedInOutSeperate(
                                 [StateQuantumNumberNames.Cparity])])
        self.add_required_qn(InteractionQuantumNumberNames.S,
                             [DefinedIfOtherQnNotDefinedInOutSeperate(
                                 [StateQuantumNumberNames.Cparity])])
        self.add_required_qn(ParticlePropertyNames.Pid,
                             [DefinedIfOtherQnNotDefinedInOutSeperate(
                                 [StateQuantumNumberNames.Cparity])])

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """ implements C_in = C_out """
        cparity_in = self.get_cparity_multiparticle(
            ingoing_part_qns, interaction_qns)
        # if cparity_in is None:
        #    return True

        cparity_out = self.get_cparity_multiparticle(
            outgoing_part_qns, interaction_qns)
        # if cparity_out is None:
        #    return True

        return cparity_in == cparity_out

    def get_cparity_multiparticle(self, part_qns, interaction_qns):
        cparity_label = StateQuantumNumberNames.Cparity
        pid_label = ParticlePropertyNames.Pid
        ang_mom_label = InteractionQuantumNumberNames.L
        int_spin_label = InteractionQuantumNumberNames.S

        no_cpar_part = [part_qns.index(x) for x in part_qns
                        if cparity_label not in x or x[cparity_label] is None]
        # if all states have c parity defined, then just multiply them
        if not no_cpar_part:
            return reduce(lambda x, y: x * y[cparity_label], part_qns, 1)

        # is this valid for two outgoing particles only?
        if len(part_qns) == 2:
            if (self.is_particle_antiparticle_pair(part_qns[0][pid_label],
                                                   part_qns[1][pid_label])):
                ang_mom = interaction_qns[ang_mom_label].magnitude()
                # if boson
                if is_boson(part_qns[0]):
                    return (-1)**ang_mom
                else:
                    coupled_spin = interaction_qns[int_spin_label].magnitude()
                    return (-1)**(ang_mom + coupled_spin)

        return None

    def is_particle_antiparticle_pair(self, pid1, pid2):
        # we just check if the pid is opposite in sign
        # this is a requirement of the pid numbers of course
        return pid1 == -pid2


class GParityConservation(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(StateQuantumNumberNames.Gparity)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """ implements G_in = G_out """
        gparity_label = StateQuantumNumberNames.Gparity
        in_part_no_gpar = [1 for x in ingoing_part_qns
                           if gparity_label not in x]
        if in_part_no_gpar:
            return True
        out_part_no_gpar = [1 for x in outgoing_part_qns
                            if gparity_label not in x]
        if out_part_no_gpar:
            return True
        gparity_in = reduce(
            lambda x, y: x * y[gparity_label], ingoing_part_qns, 1)
        gparity_out = reduce(
            lambda x, y: x * y[gparity_label], outgoing_part_qns, 1)
        if gparity_in == gparity_out:
            return True
        return False


class IdenticalParticleSymmetrization(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(StateQuantumNumberNames.Parity)
        self.add_required_qn(ParticlePropertyNames.Pid,
                             [DefinedForAllOutgoingEdges()])
        self.add_required_qn(StateQuantumNumberNames.Spin,
                             [DefinedForAllOutgoingEdges()])

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        if self.check_particles_identical(outgoing_part_qns):
            parity_label = StateQuantumNumberNames.Parity

            if is_boson(outgoing_part_qns[0]):
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
        # check if pids match
        pid_label = ParticlePropertyNames.Pid
        reference_pid = particles[0][pid_label]
        for p in particles[1:]:
            if p[pid_label] != reference_pid:
                return False
        return True


class SpinConservation(AbstractRule):
    """
    Implements conservation of a spin-like quantum number for a two body decay
    (coupling of two particle states). See ::meth::`.check` for details.
    """

    def __init__(self, spinlike_qn, use_projection=True):
        if not isinstance(spinlike_qn, StateQuantumNumberNames):
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
        self.add_required_qn(self.spinlike_qn, [DefinedForAllEdges()])
        # for actual spins we include the angular momentum
        if self.spinlike_qn is StateQuantumNumberNames.Spin:
            self.add_required_qn(InteractionQuantumNumberNames.L, [
                                 DefinedForInteractionNode()])
            self.add_required_qn(InteractionQuantumNumberNames.S, [
                                 DefinedForInteractionNode()])

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
            return self.check_magnitude(in_spins, out_spins, interaction_qns)
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
            S = interaction_qns[InteractionQuantumNumberNames.S].magnitude()
            if S not in spins_daughters_coupled:
                return False
            else:
                possible_total_spins = self.spin_couplings(S, L)
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
        self.add_required_qn(
            StateQuantumNumberNames.Spin, [DefinedForAllEdges()])

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """
        implement |lambda2-lambda3| <= S1
        """
        if len(ingoing_part_qns) == 1 and len(outgoing_part_qns) == 2:
            spin_label = StateQuantumNumberNames.Spin

            mother_spin = ingoing_part_qns[0][spin_label].magnitude()
            daughter_hel = [x[spin_label].projection()
                            for x in outgoing_part_qns]
            if mother_spin >= abs(daughter_hel[0] - daughter_hel[1]):
                return True
        return False


class GellMannNishijimaRule(AbstractRule):
    def specify_required_qns(self):
        self.add_required_qn(
            StateQuantumNumberNames.Charge, [DefinedForAllEdges()])
        self.add_required_qn(
            StateQuantumNumberNames.IsoSpin, [DefinedForAllEdges()])
        self.add_required_qn(StateQuantumNumberNames.Strangeness)
        self.add_required_qn(StateQuantumNumberNames.Charm)
        self.add_required_qn(StateQuantumNumberNames.Bottomness)
        self.add_required_qn(StateQuantumNumberNames.Topness)
        self.add_required_qn(StateQuantumNumberNames.BaryonNumber)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """
        implement hypercharge  (Y=S+C+B+T+B, last B is baryon number)
        and the Gell-Mann–Nishijima formula for each particle: Q=I_3+Y/2
        """
        charge_label = StateQuantumNumberNames.Charge
        isospin_label = StateQuantumNumberNames.IsoSpin

        for particle in ingoing_part_qns + outgoing_part_qns:
            isospin_3 = 0
            if isospin_label in particle:
                isospin_3 = particle[isospin_label].projection()
            if (float(particle[charge_label]) !=
                    (isospin_3 + 0.5 * self.calculate_hypercharge(particle))):
                return False
        return True

    def calculate_hypercharge(self, particle):
        qn_labels = [
            StateQuantumNumberNames.Strangeness,
            StateQuantumNumberNames.Charm,
            StateQuantumNumberNames.Bottomness,
            StateQuantumNumberNames.Topness,
            StateQuantumNumberNames.BaryonNumber
        ]
        qn_values = [particle[x] for x in qn_labels if x in particle]
        return sum(qn_values)


class MassConservation(AbstractRule):
    def __init__(self, width_factor=3):
        self.width_factor = width_factor

    def specify_required_qns(self):
        self.add_required_qn(
            ParticlePropertyNames.Mass, [DefinedForAllEdges()])
        self.add_required_qn(
            ParticlePropertyNames.Width)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """
        implement mass check, which makes sure that 
        the outgoing state has a net mass that lies "well" below 
        the net mass of the ingoing state plus the width times a width factor
        M_in + factor * W_in >= M_out 
        """
        mass_label = ParticlePropertyNames.Mass
        width_label = ParticlePropertyNames.Width

        mass_in = sum([x[mass_label] for x in ingoing_part_qns])
        width_in = sum([x[width_label]
                        for x in ingoing_part_qns if width_label in x])
        mass_out = sum([x[mass_label] for x in outgoing_part_qns])
        width_out = sum([x[width_label]
                         for x in outgoing_part_qns if width_label in x])

        return ((mass_in + self.width_factor * width_in) >=
                (mass_out - self.width_factor * width_out))
