"""Collection of quantum number conservation rules for particle reactions.

Contains:
- Functors for quantum number condition checks.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce

from numpy import arange

from .particle import (
    InteractionQuantumNumberNames,
    ParticleDecayPropertyNames,
    ParticlePropertyNames,
    QNNameClassMapping,
    QuantumNumberClasses,
    Spin,
    StateQuantumNumberNames,
    is_boson,
)


class AbstractConditionFunctor(ABC):
    """Abstract interface of a condition functor."""

    @abstractmethod
    def check(self, qn_names, in_edges, out_edges, int_node):
        pass


class DefinedForAllEdges(AbstractConditionFunctor):
    """Check if a graph has all edges defined."""

    def check(self, qn_names, in_edges, out_edges, int_node):
        for qn_name in qn_names:
            for edge in in_edges + out_edges:
                if qn_name not in edge:
                    return False
        return True


class DefinedForAllOutgoingEdges(AbstractConditionFunctor):
    """Check if all outgoing edges are defined."""

    def check(self, qn_names, in_edges, out_edges, int_node):
        for qn_name in qn_names:
            for edge in out_edges:
                if qn_name not in edge:
                    return False
        return True


class DefinedForInteractionNode(AbstractConditionFunctor):
    """Check if all interaction nodes are defined."""

    def check(self, qn_names, in_edges, out_edges, int_node):
        for qn_name in qn_names:
            if qn_name not in int_node:
                return False
        return True


class DefinedIfOtherQnNotDefinedInOutSeparate(AbstractConditionFunctor):
    """Implements logic for..."""

    def __init__(self, other_qn_names):
        self.other_qn_names = other_qn_names

    def check(self, qn_names, in_edges, out_edges, int_node):
        return self.check_edge_set(
            qn_names, in_edges, int_node
        ) and self.check_edge_set(qn_names, out_edges, int_node)

    def check_edge_set(self, qn_names, edges, int_node):
        found_for_all = True
        for qn_name in self.other_qn_names:
            if isinstance(
                qn_name, (StateQuantumNumberNames, ParticlePropertyNames)
            ):
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
                if isinstance(
                    qn_name, (StateQuantumNumberNames, ParticlePropertyNames)
                ):
                    for edge_props in edges:
                        if not self.find_in_dict(qn_name, edge_props):
                            return False
                else:
                    if not self.find_in_dict(qn_name, int_node):
                        return False
        return True

    @staticmethod
    def find_in_dict(name, props):
        found = False
        for key, val in props.items():
            if name == key and val is not None:
                found = True
                break
        return found


def is_particle_antiparticle_pair(pid1, pid2):
    # we just check if the pid is opposite in sign
    # this is a requirement of the pid numbers of course
    return pid1 == -pid2


class AbstractRule(ABC):
    """Abstract interface for a conservation rule."""

    def __init__(self, rule_name):
        self.rule_name = str(rule_name)
        self.required_qn_names = []
        self.qn_conditions = []
        self.specify_required_qns()

    def __repr__(self):
        return str(self.rule_name)

    def __str__(self):
        return str(self.rule_name)

    def get_qn_conditions(self):
        return self.qn_conditions

    @abstractmethod
    def specify_required_qns(self):
        pass

    def add_required_qn(self, qn_name, qn_condition_functions=None):
        if not (
            isinstance(
                qn_name,
                (
                    StateQuantumNumberNames,
                    InteractionQuantumNumberNames,
                    ParticlePropertyNames,
                    ParticleDecayPropertyNames,
                ),
            )
        ):
            raise TypeError(
                "qn_name has to be of type "
                + "ParticleQuantumNumberNames or "
                + "InteractionQuantumNumberNames or "
                + "ParticlePropertyNames"
            )
        self.required_qn_names.append(qn_name)
        if qn_condition_functions:
            for condition in qn_condition_functions:
                self.qn_conditions.append(([qn_name], condition))

    def get_required_qn_names(self):
        return self.required_qn_names

    @abstractmethod
    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        pass

    def check_requirements(self, in_edges, out_edges, int_node):
        for (qn_name_list, condition_functor) in self.get_qn_conditions():
            # part_props = [x for x in qn_name_list if isinstance(
            #    x, ParticlePropertyNames)]
            # if part_props:
            #    return False

            if not condition_functor.check(
                qn_name_list, in_edges, out_edges, int_node
            ):
                logging.debug(
                    "condition %s for quantum numbers %s for rule %s not satisfied",
                    condition_functor.__class__,
                    qn_name_list,
                    self.__class__,
                )
                return False
        return True


class AdditiveQuantumNumberConservation(AbstractRule):
    r"""Check for conservation of an additive quantum numbers.

    :math:`\sum q_{in} = \sum q_{out}`

    Additive quantum numbers are, for example:
     - electric charge
     - baryon number
     - lepton number
    """

    def __init__(self, qn_name):
        self.qn_name = qn_name
        super().__init__(qn_name.name + "Conservation")

    def specify_required_qns(self):
        self.add_required_qn(self.qn_name, [DefinedForAllEdges()])

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        in_qn_sum = sum(
            [
                part[self.qn_name]
                for part in ingoing_part_qns
                if self.qn_name in part
            ]
        )
        out_qn_sum = sum(
            [
                part[self.qn_name]
                for part in outgoing_part_qns
                if (self.qn_name in part)
            ]
        )
        return in_qn_sum == out_qn_sum


class ParityConservation(AbstractRule):
    """Check parity conservation."""

    def __init__(self):
        super().__init__("ParityConservation")

    def specify_required_qns(self):
        self.add_required_qn(
            StateQuantumNumberNames.Parity, [DefinedForAllEdges()]
        )
        self.add_required_qn(
            InteractionQuantumNumberNames.L, [DefinedForInteractionNode()]
        )

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        r"""Implement :math:`P_{in} = P_{out} \cdot (-1)^L`."""
        # is this valid for two outgoing particles only?
        parity_label = StateQuantumNumberNames.Parity
        parity_in = reduce(
            lambda x, y: x * y[parity_label], ingoing_part_qns, 1
        )
        parity_out = reduce(
            lambda x, y: x * y[parity_label], outgoing_part_qns, 1
        )
        ang_mom = interaction_qns[InteractionQuantumNumberNames.L].magnitude()
        if parity_in == (parity_out * (-1) ** ang_mom):
            return True
        return False


class ParityConservationHelicity(AbstractRule):
    """Check parity conservation for the helicity formalism."""

    def __init__(self):
        super().__init__("ParityConservationHelicity")

    def specify_required_qns(self):
        self.add_required_qn(
            StateQuantumNumberNames.Parity, [DefinedForAllEdges()]
        )
        self.add_required_qn(
            StateQuantumNumberNames.Spin, [DefinedForAllEdges()]
        )
        self.add_required_qn(
            InteractionQuantumNumberNames.ParityPrefactor,
            [DefinedForInteractionNode()],
        )

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        r"""Implements the check parity conservation check.

        Check the following:

        .. math:: A_{-\lambda_1-\lambda_2} = P_1 P_2 P_3 (-1)^{S_2+S_3-S_1}
           A_{\lambda_1\lambda_2}

        Notice that only the special case :math:`\lambda_1=\lambda_2=0` may
        return False.
        """
        if len(ingoing_part_qns) == 1 and len(outgoing_part_qns) == 2:
            spin_label = StateQuantumNumberNames.Spin
            parity_label = StateQuantumNumberNames.Parity

            spins = [
                x[spin_label].magnitude()
                for x in ingoing_part_qns + outgoing_part_qns
            ]
            parity_product = reduce(
                lambda x, y: x * y[parity_label],
                ingoing_part_qns + outgoing_part_qns,
                1,
            )

            prefactor = parity_product * (-1.0) ** (
                spins[1] + spins[2] - spins[0]
            )

            daughter_hel = [
                0
                for x in outgoing_part_qns
                if x[spin_label].projection() == 0.0
            ]
            if len(daughter_hel) == 2:
                if prefactor == -1:
                    return False

            pf_label = InteractionQuantumNumberNames.ParityPrefactor
            return prefactor == interaction_qns[pf_label]
        return True


class CParityConservation(AbstractRule):
    """Check for :math:`C`-parity conservation."""

    def __init__(self):
        super().__init__("CParityConservation")

    def specify_required_qns(self):
        self.add_required_qn(StateQuantumNumberNames.CParity)
        # the spin quantum number is required to check if the daughter
        # particles are fermions or bosons
        self.add_required_qn(
            StateQuantumNumberNames.Spin,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.CParity]
                )
            ],
        )
        self.add_required_qn(
            InteractionQuantumNumberNames.L,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.CParity]
                )
            ],
        )
        self.add_required_qn(
            InteractionQuantumNumberNames.S,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.CParity]
                )
            ],
        )
        self.add_required_qn(
            ParticlePropertyNames.Pid,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.CParity]
                )
            ],
        )

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """Check for :math:`C_{in} = C_{out}`."""
        c_parity_in = self.get_c_parity_multiparticle(
            ingoing_part_qns, interaction_qns
        )
        if c_parity_in is None:
            return True

        c_parity_out = self.get_c_parity_multiparticle(
            outgoing_part_qns, interaction_qns
        )
        if c_parity_out is None:
            return True

        return c_parity_in == c_parity_out

    @staticmethod
    def get_c_parity_multiparticle(part_qns, interaction_qns):
        c_parity_label = StateQuantumNumberNames.CParity
        pid_label = ParticlePropertyNames.Pid
        ang_mom_label = InteractionQuantumNumberNames.L
        int_spin_label = InteractionQuantumNumberNames.S

        no_c_parity_part = [
            part_qns.index(x)
            for x in part_qns
            if c_parity_label not in x or x[c_parity_label] is None
        ]
        # if all states have C parity defined, then just multiply them
        if not no_c_parity_part:
            return reduce(lambda x, y: x * y[c_parity_label], part_qns, 1)

        # two particle case
        if len(part_qns) == 2:
            if is_particle_antiparticle_pair(
                part_qns[0][pid_label], part_qns[1][pid_label]
            ):
                ang_mom = interaction_qns[ang_mom_label].magnitude()
                # if boson
                if is_boson(part_qns[0]):
                    return (-1) ** ang_mom
                coupled_spin = interaction_qns[int_spin_label].magnitude()
                return (-1) ** (ang_mom + coupled_spin)
        return None


class GParityConservation(AbstractRule):
    """Check for :math:`G`-parity conservation."""

    def __init__(self):
        super().__init__("GParityConservation")

    def specify_required_qns(self):
        self.add_required_qn(StateQuantumNumberNames.GParity)
        # the spin quantum number is required to check if the daughter
        # particles are fermions or bosons
        self.add_required_qn(
            StateQuantumNumberNames.Spin,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        )
        self.add_required_qn(
            InteractionQuantumNumberNames.L,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        )
        self.add_required_qn(
            InteractionQuantumNumberNames.S,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        )
        self.add_required_qn(
            StateQuantumNumberNames.IsoSpin,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        )
        self.add_required_qn(
            ParticlePropertyNames.Pid,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        )

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """Check for :math:`G_{in} = G_{out}`."""
        g_parity_label = StateQuantumNumberNames.GParity
        no_g_parity_in_part = [
            ingoing_part_qns.index(x)
            for x in ingoing_part_qns
            if g_parity_label not in x or x[g_parity_label] is None
        ]
        no_g_parity_out_part = [
            outgoing_part_qns.index(x)
            for x in outgoing_part_qns
            if g_parity_label not in x or x[g_parity_label] is None
        ]
        # if all states have G parity defined, then just multiply them
        if not no_g_parity_in_part + no_g_parity_out_part:
            in_g_parity = reduce(
                lambda x, y: x * y[g_parity_label], ingoing_part_qns, 1
            )
            out_g_parity = reduce(
                lambda x, y: x * y[g_parity_label], outgoing_part_qns, 1
            )
            return in_g_parity == out_g_parity

        # two particle case
        particle_counts = (len(ingoing_part_qns), len(outgoing_part_qns))
        if particle_counts == (1, 2):
            if g_parity_label in ingoing_part_qns[0]:
                out_g_parity = self.check_multistate_g_parity(
                    ingoing_part_qns, outgoing_part_qns, interaction_qns
                )
                in_g_parity = ingoing_part_qns[0][g_parity_label]
                if out_g_parity is not None and in_g_parity is not None:
                    return out_g_parity == in_g_parity

        if particle_counts == (2, 1):
            if g_parity_label in outgoing_part_qns[0]:
                in_g_parity = self.check_multistate_g_parity(
                    outgoing_part_qns, ingoing_part_qns, interaction_qns
                )
                out_g_parity = outgoing_part_qns[0][g_parity_label]
                if out_g_parity is not None and in_g_parity is not None:
                    return out_g_parity == in_g_parity
        return True

    @staticmethod
    def check_multistate_g_parity(
        single_state_qns, double_state_qns, interaction_qns
    ):
        isospin_label = StateQuantumNumberNames.IsoSpin
        pid_label = ParticlePropertyNames.Pid
        ang_mom_label = InteractionQuantumNumberNames.L
        int_spin_label = InteractionQuantumNumberNames.S
        if is_particle_antiparticle_pair(
            double_state_qns[0][pid_label], double_state_qns[1][pid_label]
        ):
            ang_mom = interaction_qns[ang_mom_label].magnitude()
            isospin = single_state_qns[0][isospin_label].magnitude()
            # if boson
            if is_boson(double_state_qns[0]):
                return (-1) ** (ang_mom + isospin)
            coupled_spin = interaction_qns[int_spin_label].magnitude()
            return (-1) ** (ang_mom + coupled_spin + isospin)
        return None


class IdenticalParticleSymmetrization(AbstractRule):
    """Implementation of particle symmetrization."""

    def __init__(self):
        super().__init__("IdenticalParticleSymmetrization")

    def specify_required_qns(self):
        self.add_required_qn(StateQuantumNumberNames.Parity)
        self.add_required_qn(
            ParticlePropertyNames.Pid, [DefinedForAllOutgoingEdges()]
        )
        self.add_required_qn(
            StateQuantumNumberNames.Spin, [DefinedForAllOutgoingEdges()]
        )

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

    @staticmethod
    def check_particles_identical(particles):
        """Check if pids and spins match."""
        pid_label = ParticlePropertyNames.Pid
        spin_label = StateQuantumNumberNames.Spin
        reference_pid = particles[0][pid_label]
        reference_spin = particles[0][spin_label]
        for particle in particles[1:]:
            if particle[pid_label] != reference_pid:
                return False
            if particle[spin_label] != reference_spin:
                return False
        return True


class SpinConservation(AbstractRule):
    """Implementation of conservation of a spin-like quantum number.

    That is, for a two body decay (coupling of two particle states). See
    :meth:`~.SpinConservation.check` for details.
    """

    def __init__(self, spinlike_qn, use_projection=True):
        if not isinstance(spinlike_qn, StateQuantumNumberNames):
            raise TypeError(
                "Expecting Enum of the type "
                "ParticleQuantumNumberNames for spinlike_qn"
            )
        if spinlike_qn not in QNNameClassMapping:
            raise ValueError("spinlike_qn is not associated with a QN class")
        if QNNameClassMapping[spinlike_qn] is not QuantumNumberClasses.Spin:
            raise ValueError("spinlike_qn is not of class Spin")
        self.spinlike_qn = spinlike_qn
        self.use_projection = use_projection
        super().__init__(spinlike_qn.name + "Conservation")

    def specify_required_qns(self):
        self.add_required_qn(self.spinlike_qn, [DefinedForAllEdges()])
        # for actual spins we include the angular momentum
        if self.spinlike_qn is StateQuantumNumberNames.Spin:
            self.add_required_qn(
                InteractionQuantumNumberNames.L, [DefinedForInteractionNode()]
            )
            self.add_required_qn(
                InteractionQuantumNumberNames.S, [DefinedForInteractionNode()]
            )

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        r"""Check for spin conservation.

        Implements

        .. math::
            |S_1 - S_2| \leq S \leq |S_1 + S_2|

        and optionally

        .. math::
            |L - S| \leq J \leq |L + S|

        Also checks :math:`M_1 + M_2 = M` and if Clebsch-Gordan coefficients
        are all 0.
        """
        spin_label = self.spinlike_qn

        in_spins = [x[spin_label] for x in ingoing_part_qns]
        out_spins = [x[spin_label] for x in outgoing_part_qns]
        if self.use_projection and not self.check_projections(
            in_spins, out_spins
        ):
            return False
        return self.check_magnitude(in_spins, out_spins, interaction_qns)

    @staticmethod
    def check_projections(in_part, out_part):
        in_proj = [x.projection() for x in in_part]
        out_proj = [x.projection() for x in out_part]
        return sum(in_proj) == sum(out_proj)

    def check_magnitude(self, in_part, out_part, interaction_qns):
        in_tot_spins = self.calculate_total_spins(in_part, interaction_qns)
        out_tot_spins = self.calculate_total_spins(out_part, interaction_qns)
        matching_spins = in_tot_spins.intersection(out_tot_spins)
        if len(matching_spins) > 0:
            return True
        return False

    def calculate_total_spins(self, part_list, interaction_qns):
        # pylint: disable=too-many-branches
        total_spins = set()
        if len(part_list) == 1:
            if self.use_projection:
                total_spins.add(part_list[0])
            else:
                total_spins.add(Spin(part_list[0].magnitude(), 0))
        else:
            # first couple all spins together
            spins_daughters_coupled = set()
            spin_list = deepcopy(part_list)
            while spin_list:
                if spins_daughters_coupled:
                    temp_coupled_spins = set()
                    tempspin = spin_list.pop()
                    for spin in spins_daughters_coupled:
                        coupled_spins = self.spin_couplings(spin, tempspin)
                        temp_coupled_spins.update(coupled_spins)
                    spins_daughters_coupled = temp_coupled_spins
                else:
                    spins_daughters_coupled.add(spin_list.pop())
            if InteractionQuantumNumberNames.L in interaction_qns:
                ang_mom = interaction_qns[InteractionQuantumNumberNames.L]
                spin = interaction_qns[InteractionQuantumNumberNames.S]
                if self.use_projection:
                    if spin in spins_daughters_coupled:
                        total_spins.update(self.spin_couplings(spin, ang_mom))
                else:
                    if spin.magnitude() in [
                        x.magnitude() for x in spins_daughters_coupled
                    ]:
                        total_spins.update(self.spin_couplings(spin, ang_mom))
            else:
                if self.use_projection:
                    total_spins = spins_daughters_coupled
                else:
                    total_spins = [
                        Spin(x.magnitude(), 0.0)
                        for x in spins_daughters_coupled
                    ]
        return total_spins

    def spin_couplings(self, spin1, spin2):
        r"""Implement the coupling of two spins.

        :math:`|S_1 - S_2| \leq S \leq |S_1 + S_2|` and :math:`M_1 + M_2 = M`
        """
        j_1 = spin1.magnitude()
        j_2 = spin2.magnitude()
        if self.use_projection:
            sum_proj = spin1.projection() + spin2.projection()
            possible_spins = [
                Spin(x, sum_proj)
                for x in arange(abs(j_1 - j_2), j_1 + j_2 + 1, 1).tolist()
                if x >= abs(sum_proj)
            ]

            return [
                x
                for x in possible_spins
                if not is_clebsch_gordan_coefficient_zero(spin1, spin2, x)
            ]
        return [
            Spin(x, 0)
            for x in arange(abs(j_1 - j_2), j_1 + j_2 + 1, 1).tolist()
        ]


def is_clebsch_gordan_coefficient_zero(spin1, spin2, spin_coupled):
    m_1 = spin1.projection()
    j_1 = spin1.magnitude()
    m_2 = spin2.projection()
    j_2 = spin2.magnitude()
    proj = spin_coupled.projection()
    mag = spin_coupled.magnitude()
    is_zero = False
    if (j_1 == j_2 and m_1 == m_2) or (m_1 == 0.0 and m_2 == 0.0):
        if abs(mag - j_1 - j_2) % 2 == 1:
            is_zero = True
    elif j_1 == mag and m_1 == -proj:
        if abs(j_2 - j_1 - mag) % 2 == 1:
            is_zero = True
    elif j_2 == mag and m_2 == -proj:
        if abs(j_1 - j_2 - mag) % 2 == 1:
            is_zero = True
    return is_zero


class ClebschGordanCheckHelicityToCanonical(AbstractRule):
    """Implement Clebsch-Gordan checks.

    For :math:`S_1, S_2` to :math:`S` and the :math:`L,S` to :math:`J` coupling
    based on the conversion of helicity to canonical amplitude sums.
    """

    def __init__(self):
        super().__init__("ClebschGordanCheckHelicityToCanonical")

    def specify_required_qns(self):
        self.add_required_qn(
            StateQuantumNumberNames.Spin, [DefinedForAllEdges()]
        )
        self.add_required_qn(
            InteractionQuantumNumberNames.L, [DefinedForInteractionNode()]
        )
        self.add_required_qn(
            InteractionQuantumNumberNames.S, [DefinedForInteractionNode()]
        )

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        if len(ingoing_part_qns) == 1 and len(outgoing_part_qns) == 2:
            spin_label = StateQuantumNumberNames.Spin
            in_spins = [x[spin_label] for x in ingoing_part_qns]
            out_spins = [x[spin_label] for x in outgoing_part_qns]
            out_spins[1] = Spin(
                out_spins[1].magnitude(), -out_spins[1].projection()
            )
            helicity_diff = sum([x.projection() for x in out_spins])
            ang_mom = interaction_qns[InteractionQuantumNumberNames.L]
            spin = interaction_qns[InteractionQuantumNumberNames.S]
            if spin.magnitude() < abs(helicity_diff) or in_spins[
                0
            ].magnitude() < abs(helicity_diff):
                return False
            spin = Spin(spin.magnitude(), helicity_diff)
            if is_clebsch_gordan_coefficient_zero(
                out_spins[0], out_spins[1], spin
            ):
                return False
            in_spins[0] = Spin(in_spins[0].magnitude(), helicity_diff)
            return not is_clebsch_gordan_coefficient_zero(
                ang_mom, spin, in_spins[0]
            )
        return False


class HelicityConservation(AbstractRule):
    """Implementation of helicity conservation."""

    def __init__(self):
        super().__init__("HelicityConservation")

    def specify_required_qns(self):
        self.add_required_qn(
            StateQuantumNumberNames.Spin, [DefinedForAllEdges()]
        )

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        r"""Check for :math:`|\lambda_2-\lambda_3| \leq S_1`."""
        if len(ingoing_part_qns) == 1 and len(outgoing_part_qns) == 2:
            spin_label = StateQuantumNumberNames.Spin

            mother_spin = ingoing_part_qns[0][spin_label].magnitude()
            daughter_hel = [
                x[spin_label].projection() for x in outgoing_part_qns
            ]
            if mother_spin >= abs(daughter_hel[0] - daughter_hel[1]):
                return True
        return False


class GellMannNishijimaRule(AbstractRule):
    """Conservation rule for Gell-Mann-Nishijima."""

    def __init__(self):
        super().__init__("GellMannNishijimaRule")

    def specify_required_qns(self):
        self.add_required_qn(
            StateQuantumNumberNames.Charge, [DefinedForAllEdges()]
        )
        self.add_required_qn(
            StateQuantumNumberNames.IsoSpin, [DefinedForAllEdges()]
        )
        self.add_required_qn(StateQuantumNumberNames.Strangeness)
        self.add_required_qn(StateQuantumNumberNames.Charm)
        self.add_required_qn(StateQuantumNumberNames.Bottomness)
        self.add_required_qn(StateQuantumNumberNames.Topness)
        self.add_required_qn(StateQuantumNumberNames.BaryonNumber)
        self.add_required_qn(StateQuantumNumberNames.ElectronLN)
        self.add_required_qn(StateQuantumNumberNames.MuonLN)
        self.add_required_qn(StateQuantumNumberNames.TauLN)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        r"""Check the Gell-Mann–Nishijima formula.

        :math:`Q=I_3+\frac{Y}{2}` for each particle.
        """
        charge_label = StateQuantumNumberNames.Charge
        isospin_label = StateQuantumNumberNames.IsoSpin

        eln = StateQuantumNumberNames.ElectronLN
        mln = StateQuantumNumberNames.MuonLN
        tln = StateQuantumNumberNames.TauLN

        for particle in ingoing_part_qns + outgoing_part_qns:
            if (
                sum(
                    [
                        abs(particle[x])
                        for x in [eln, mln, tln]
                        if x in particle
                    ]
                )
                > 0.0
            ):
                # if particle is a lepton then skip the check
                continue
            isospin_3 = 0
            if isospin_label in particle:
                isospin_3 = particle[isospin_label].projection()
            if float(particle[charge_label]) != (
                isospin_3 + 0.5 * self.calculate_hypercharge(particle)
            ):
                return False
        return True

    @staticmethod
    def calculate_hypercharge(particle):
        """Calculate the hypercharge :math:`Y=S+C+B+T+B`."""
        qn_labels = [
            StateQuantumNumberNames.Strangeness,
            StateQuantumNumberNames.Charm,
            StateQuantumNumberNames.Bottomness,
            StateQuantumNumberNames.Topness,
            StateQuantumNumberNames.BaryonNumber,
        ]
        qn_values = [particle[x] for x in qn_labels if x in particle]
        return sum(qn_values)


class MassConservation(AbstractRule):
    """Mass conservation rule."""

    def __init__(self, width_factor=3):
        self.width_factor = width_factor
        super().__init__("MassConservation")

    def specify_required_qns(self):
        self.add_required_qn(
            ParticlePropertyNames.Mass, [DefinedForAllEdges()]
        )
        self.add_required_qn(ParticleDecayPropertyNames.Width)

    def check(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        r"""Implement the mass check.

        :math:`M_{out} - N \cdot W_{out} < M_{in} + N \cdot W_{in}`

        It makes sure that the net mass outgoing state :math:`M_{out}` is
        smaller than the net mass of the ingoing state :math:`M_{in}`. Also the
        width :math:`W` of the states is taken into account.
        """
        mass_label = ParticlePropertyNames.Mass
        width_label = ParticleDecayPropertyNames.Width

        mass_in = sum([x[mass_label] for x in ingoing_part_qns])
        width_in = sum(
            [x[width_label] for x in ingoing_part_qns if width_label in x]
        )
        mass_out = sum([x[mass_label] for x in outgoing_part_qns])
        width_out = sum(
            [x[width_label] for x in outgoing_part_qns if width_label in x]
        )

        return (mass_out - self.width_factor * width_out) < (
            mass_in + self.width_factor * width_in
        )
