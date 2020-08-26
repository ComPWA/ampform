"""Collection of quantum number conservation rules for particle reactions.

Contains:
- Functors for quantum number condition checks.
"""

# pylint: disable=abstract-method

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce

from numpy import arange

from expertsystem.data import Spin

from .particle import (
    InteractionQuantumNumberNames,
    ParticleDecayPropertyNames,
    ParticlePropertyNames,
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


class Rule:
    """Interface for rules.

    A `Rule` performs checks on an `.InteractionNode` and its attached `.Edge` s.
    The `__call__` method contains actual rule logic and has to be overwritten.

    For additive quantum numbers the decorator `additive_quantum_number_rule`
    can simplify the constrution of the appropriate `Rule`.

    Besides the rule logic itself, a `Rule` also has the responsibility of
    stating its run conditions. These can be separated into two categories:

    * variable conditions
    * toplogical conditions

    Note: currently only variable conditions are being used.

    Variable conditions can easily be added to rules via the `rule_conditions`
    decorator.
    """

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    def get_required_qn_names(self):
        raise NotImplementedError

    def check_requirements(self, in_edges, out_edges, int_node):
        raise NotImplementedError

    def __call__(
        self, ingoing_part_qns, outgoing_part_qns, interaction_qns
    ) -> bool:
        raise NotImplementedError


def rule_conditions(variable_conditions):
    quantum_number_types = (
        StateQuantumNumberNames,
        InteractionQuantumNumberNames,
        ParticlePropertyNames,
        ParticleDecayPropertyNames,
    )

    def decorator(rule_class):
        all_conditions = []
        required_qns = []
        for var_condition in variable_conditions:
            conditions = None
            if isinstance(var_condition, tuple):
                qn_name, conditions = var_condition
            else:
                qn_name = var_condition
            if not isinstance(qn_name, quantum_number_types):
                raise TypeError(
                    "qn_name has to be of one of the following types:\n"
                    f"  {quantum_number_types}"
                )
            required_qns.append(qn_name)
            if conditions:
                if isinstance(conditions, list):
                    for condition in conditions:
                        all_conditions.append(([qn_name], condition))
                else:
                    all_conditions.append(([qn_name], conditions))

        def check_requirements(self, in_edges, out_edges, int_node):
            for (qn_name_list, condition_functor) in all_conditions:
                if not condition_functor.check(
                    qn_name_list, in_edges, out_edges, int_node
                ):
                    logging.debug(
                        "condition %s for quantum numbers %s for rule %s not satisfied",
                        condition_functor.__class__,
                        qn_name_list,
                        str(self),
                    )
                    return False
            return True

        def get_required_qn_names(self):  # pylint: disable=unused-argument
            return required_qns

        rule_class.check_requirements = check_requirements
        rule_class.get_required_qn_names = get_required_qn_names
        if rule_class.__doc__ is None:
            rule_class.__doc__ = ""
        else:
            rule_class.__doc__ += "\n\n"
        rule_class.__doc__ += "Required quantum numbers:\n\n"
        for required_qn in required_qns:
            rule_class.__doc__ += f"  - `.{required_qn}`\n"

        return rule_class

    return decorator


def additive_quantum_number_rule(quantum_number: StateQuantumNumberNames):
    r"""Class decorator for creating an additive conservation `Rule`.

    Use this decorator to create a conservation `Rule` for a quantum number
    to which an additive conservation rule applies:

    .. math:: \sum q_{in} = \sum q_{out}

    Args:
        quantum_number (StateQuantumNumberNames): Quantum number to which you
            want to apply the additive conservation check.
    """

    def decorator(rule_class):
        def new_call(
            self, ingoing_part_qns, outgoing_part_qns, _
        ):  # pylint: disable=unused-argument
            charge = quantum_number
            in_qn_sum = sum(
                [part[charge] for part in ingoing_part_qns if charge in part]
            )
            out_qn_sum = sum(
                [part[charge] for part in outgoing_part_qns if charge in part]
            )
            return in_qn_sum == out_qn_sum

        rule_class.__call__ = new_call
        rule_class.__doc__ = (
            f"""Decorated via `{additive_quantum_number_rule.__name__}`.\n\n"""
            f"""Check for {quantum_number.name} conservation."""
        )
        rule_class = rule_conditions(
            variable_conditions=[(quantum_number, [DefinedForAllEdges()])]
        )(rule_class)
        return rule_class

    return decorator


@additive_quantum_number_rule(StateQuantumNumberNames.Charge)
class ChargeConservation(Rule):
    pass


@additive_quantum_number_rule(StateQuantumNumberNames.BaryonNumber)
class BaryonNumberConservation(Rule):
    pass


@additive_quantum_number_rule(StateQuantumNumberNames.ElectronLN)
class ElectronLNConservation(Rule):
    pass


@additive_quantum_number_rule(StateQuantumNumberNames.MuonLN)
class MuonLNConservation(Rule):
    pass


@additive_quantum_number_rule(StateQuantumNumberNames.TauLN)
class TauLNConservation(Rule):
    pass


@additive_quantum_number_rule(StateQuantumNumberNames.Strangeness)
class StrangenessConservation(Rule):
    pass


@additive_quantum_number_rule(StateQuantumNumberNames.Charmness)
class CharmConservation(Rule):
    pass


@rule_conditions(
    variable_conditions=[
        (StateQuantumNumberNames.Parity, [DefinedForAllEdges()]),
        (InteractionQuantumNumberNames.L, [DefinedForInteractionNode()]),
    ]
)
class ParityConservation(Rule):
    def __call__(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        r"""Implement :math:`P_{in} = P_{out} \cdot (-1)^L`."""
        # is this valid for two outgoing particles only?
        parity_label = StateQuantumNumberNames.Parity
        parity_in = reduce(
            lambda x, y: x * y[parity_label], ingoing_part_qns, 1
        )
        parity_out = reduce(
            lambda x, y: x * y[parity_label], outgoing_part_qns, 1
        )
        ang_mom = interaction_qns[InteractionQuantumNumberNames.L].magnitude
        if parity_in == (parity_out * (-1) ** ang_mom):
            return True
        return False


@rule_conditions(
    variable_conditions=[
        (StateQuantumNumberNames.Parity, [DefinedForAllEdges()]),
        (StateQuantumNumberNames.Spin, [DefinedForAllEdges()]),
        (
            InteractionQuantumNumberNames.ParityPrefactor,
            [DefinedForInteractionNode()],
        ),
    ]
)
class ParityConservationHelicity(Rule):
    def __call__(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        r"""Implements parity conservation for helicity formalism.

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
                x[spin_label].magnitude
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
                0 for x in outgoing_part_qns if x[spin_label].projection == 0.0
            ]
            if len(daughter_hel) == 2:
                if prefactor == -1:
                    return False

            pf_label = InteractionQuantumNumberNames.ParityPrefactor
            return prefactor == interaction_qns[pf_label]
        return True


@rule_conditions(
    variable_conditions=[
        (StateQuantumNumberNames.CParity),
        (
            StateQuantumNumberNames.Spin,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.CParity]
                )
            ],
        ),
        (
            InteractionQuantumNumberNames.L,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.CParity]
                )
            ],
        ),
        (
            InteractionQuantumNumberNames.S,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.CParity]
                )
            ],
        ),
        (
            ParticlePropertyNames.Pid,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.CParity]
                )
            ],
        ),
    ]
)
class CParityConservation(Rule):
    def __call__(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """Check for :math:`C`-parity conservation.

        Implements :math:`C_{in} = C_{out}`.
        """

        def _get_c_parity_multiparticle(part_qns, interaction_qns):
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
                    ang_mom = interaction_qns[ang_mom_label].magnitude
                    # if boson
                    if is_boson(part_qns[0]):
                        return (-1) ** ang_mom
                    coupled_spin = interaction_qns[int_spin_label].magnitude
                    return (-1) ** (ang_mom + coupled_spin)
            return None

        c_parity_in = _get_c_parity_multiparticle(
            ingoing_part_qns, interaction_qns
        )
        if c_parity_in is None:
            return True

        c_parity_out = _get_c_parity_multiparticle(
            outgoing_part_qns, interaction_qns
        )
        if c_parity_out is None:
            return True

        return c_parity_in == c_parity_out


@rule_conditions(
    variable_conditions=[
        (
            StateQuantumNumberNames.Spin,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        ),
        (
            InteractionQuantumNumberNames.L,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        ),
        (
            InteractionQuantumNumberNames.S,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        ),
        (
            StateQuantumNumberNames.IsoSpin,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        ),
        (
            ParticlePropertyNames.Pid,
            [
                DefinedIfOtherQnNotDefinedInOutSeparate(
                    [StateQuantumNumberNames.GParity]
                )
            ],
        ),
    ]
)
class GParityConservation(Rule):
    def __call__(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """Check for :math:`G`-parity conservation.

        Implements for :math:`G_{in} = G_{out}`.
        """

        def _check_multistate_g_parity(
            single_state_qns, double_state_qns, interaction_qns
        ):
            isospin_label = StateQuantumNumberNames.IsoSpin
            pid_label = ParticlePropertyNames.Pid
            ang_mom_label = InteractionQuantumNumberNames.L
            int_spin_label = InteractionQuantumNumberNames.S
            if is_particle_antiparticle_pair(
                double_state_qns[0][pid_label], double_state_qns[1][pid_label]
            ):
                ang_mom = interaction_qns[ang_mom_label].magnitude
                isospin = single_state_qns[0][isospin_label].magnitude
                # if boson
                if is_boson(double_state_qns[0]):
                    return (-1) ** (ang_mom + isospin)
                coupled_spin = interaction_qns[int_spin_label].magnitude
                return (-1) ** (ang_mom + coupled_spin + isospin)
            return None

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
                out_g_parity = _check_multistate_g_parity(
                    ingoing_part_qns, outgoing_part_qns, interaction_qns
                )
                in_g_parity = ingoing_part_qns[0][g_parity_label]
                if out_g_parity is not None and in_g_parity is not None:
                    return out_g_parity == in_g_parity

        if particle_counts == (2, 1):
            if g_parity_label in outgoing_part_qns[0]:
                in_g_parity = _check_multistate_g_parity(
                    outgoing_part_qns, ingoing_part_qns, interaction_qns
                )
                out_g_parity = outgoing_part_qns[0][g_parity_label]
                if out_g_parity is not None and in_g_parity is not None:
                    return out_g_parity == in_g_parity
        return True


@rule_conditions(
    variable_conditions=[
        (StateQuantumNumberNames.Parity, [DefinedForAllEdges()]),
        (ParticlePropertyNames.Pid, [DefinedForAllOutgoingEdges()]),
        (StateQuantumNumberNames.Spin, [DefinedForAllOutgoingEdges()]),
    ]
)
class IdenticalParticleSymmetrization(Rule):
    def __call__(self, ingoing_part_qns, outgoing_part_qns, _):
        """Implementation of particle symmetrization."""

        def _check_particles_identical(particles):
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

        if _check_particles_identical(outgoing_part_qns):
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


def _is_clebsch_gordan_coefficient_zero(spin1, spin2, spin_coupled):
    m_1 = spin1.projection
    j_1 = spin1.magnitude
    m_2 = spin2.projection
    j_2 = spin2.magnitude
    proj = spin_coupled.projection
    mag = spin_coupled.magnitude
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


def _check_projections(in_part, out_part):
    in_proj = [x.projection for x in in_part]
    out_proj = [x.projection for x in out_part]
    return sum(in_proj) == sum(out_proj)


def _check_magnitude(in_part, out_part, interaction_qns, use_projection=True):
    in_tot_spins = __calculate_total_spins(
        in_part, interaction_qns, use_projection
    )
    out_tot_spins = __calculate_total_spins(
        out_part, interaction_qns, use_projection
    )
    matching_spins = in_tot_spins.intersection(out_tot_spins)
    if len(matching_spins) > 0:
        return True
    return False


def __calculate_total_spins(part_list, interaction_qns, use_projection):
    # pylint: disable=too-many-branches
    total_spins = set()
    if len(part_list) == 1:
        if use_projection:
            total_spins.add(part_list[0])
        else:
            spin_magnitude = part_list[0].magnitude
            total_spins.add(Spin(spin_magnitude, spin_magnitude))
    else:
        # first couple all spins together
        spins_daughters_coupled = set()
        spin_list = deepcopy(part_list)
        while spin_list:
            if spins_daughters_coupled:
                temp_coupled_spins = set()
                tempspin = spin_list.pop()
                for spin in spins_daughters_coupled:
                    coupled_spins = __spin_couplings(
                        spin, tempspin, use_projection
                    )
                    temp_coupled_spins.update(coupled_spins)
                spins_daughters_coupled = temp_coupled_spins
            else:
                spins_daughters_coupled.add(spin_list.pop())
        if InteractionQuantumNumberNames.L in interaction_qns:
            ang_mom = interaction_qns[InteractionQuantumNumberNames.L]
            spin = interaction_qns[InteractionQuantumNumberNames.S]
            if use_projection:
                if spin in spins_daughters_coupled:
                    total_spins.update(
                        __spin_couplings(spin, ang_mom, use_projection)
                    )
            else:
                if spin.magnitude in [
                    x.magnitude for x in spins_daughters_coupled
                ]:
                    total_spins.update(
                        __spin_couplings(spin, ang_mom, use_projection)
                    )
        else:
            if use_projection:
                total_spins = spins_daughters_coupled
            else:
                total_spins = [
                    Spin(x.magnitude, x.magnitude)
                    for x in spins_daughters_coupled
                ]
    return total_spins


def __spin_couplings(spin1, spin2, use_projection):
    r"""Implement the coupling of two spins.

    :math:`|S_1 - S_2| \leq S \leq |S_1 + S_2|` and :math:`M_1 + M_2 = M`
    """
    j_1 = spin1.magnitude
    j_2 = spin2.magnitude
    if use_projection:
        sum_proj = spin1.projection + spin2.projection
        possible_spins = [
            Spin(x, sum_proj)
            for x in arange(abs(j_1 - j_2), j_1 + j_2 + 1, 1).tolist()
            if x >= abs(sum_proj)
        ]

        return [
            x
            for x in possible_spins
            if not _is_clebsch_gordan_coefficient_zero(spin1, spin2, x)
        ]
    return [
        Spin(x, x) for x in arange(abs(j_1 - j_2), j_1 + j_2 + 1, 1).tolist()
    ]


@rule_conditions(
    variable_conditions=[
        (StateQuantumNumberNames.IsoSpin, [DefinedForAllEdges()])
    ]
)
class IsoSpinConservation(Rule):
    r"""Check for isospin conservation.

    Implements

    .. math::
        |I_1 - I_2| \leq I \leq |I_1 + I_2|

    Also checks :math:`I_{1,z} + I_{2,z} = I_z` and if Clebsch-Gordan
    coefficients are all 0.
    """

    def __call__(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):

        isospin_label = StateQuantumNumberNames.IsoSpin

        in_spins = [x[isospin_label] for x in ingoing_part_qns]
        out_spins = [x[isospin_label] for x in outgoing_part_qns]
        if not _check_projections(in_spins, out_spins):
            return False
        return _check_magnitude(
            in_spins, out_spins, interaction_qns, use_projection=True
        )


@rule_conditions(
    variable_conditions=[
        (StateQuantumNumberNames.Spin, [DefinedForAllEdges()]),
        (InteractionQuantumNumberNames.L, [DefinedForInteractionNode()]),
        (InteractionQuantumNumberNames.S, [DefinedForInteractionNode()]),
    ]
)
class SpinConservation(Rule):
    r"""Check for spin conservation.

    Implements

    .. math::
        |S_1 - S_2| \leq S \leq |S_1 + S_2|

    and

    .. math::
        |L - S| \leq J \leq |L + S|

    Also checks :math:`M_1 + M_2 = M` and if Clebsch-Gordan coefficients
    are all 0.
    """

    def __init__(self, use_projection=True):
        self.__use_projection = use_projection
        super().__init__()

    def __call__(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        spin_label = StateQuantumNumberNames.Spin

        in_spins = [x[spin_label] for x in ingoing_part_qns]
        out_spins = [x[spin_label] for x in outgoing_part_qns]
        if self.__use_projection and not _check_projections(
            in_spins, out_spins
        ):
            return False
        return _check_magnitude(
            in_spins,
            out_spins,
            interaction_qns,
            use_projection=self.__use_projection,
        )


@rule_conditions(
    variable_conditions=[
        (StateQuantumNumberNames.Spin, [DefinedForAllEdges()]),
        (InteractionQuantumNumberNames.L, [DefinedForInteractionNode()]),
        (InteractionQuantumNumberNames.S, [DefinedForInteractionNode()]),
    ]
)
class ClebschGordanCheckHelicityToCanonical(Rule):
    def __call__(self, ingoing_part_qns, outgoing_part_qns, interaction_qns):
        """Implement Clebsch-Gordan checks.

        For :math:`S_1, S_2` to :math:`S` and the :math:`L,S` to :math:`J` coupling
        based on the conversion of helicity to canonical amplitude sums.
        """
        if len(ingoing_part_qns) == 1 and len(outgoing_part_qns) == 2:
            spin_label = StateQuantumNumberNames.Spin
            in_spins = [x[spin_label] for x in ingoing_part_qns]
            out_spins = [x[spin_label] for x in outgoing_part_qns]
            out_spins[1] = Spin(
                out_spins[1].magnitude, -out_spins[1].projection
            )
            helicity_diff = sum([x.projection for x in out_spins])
            ang_mom = interaction_qns[InteractionQuantumNumberNames.L]
            spin = interaction_qns[InteractionQuantumNumberNames.S]
            if spin.magnitude < abs(helicity_diff) or in_spins[
                0
            ].magnitude < abs(helicity_diff):
                return False
            spin = Spin(spin.magnitude, helicity_diff)
            if _is_clebsch_gordan_coefficient_zero(
                out_spins[0], out_spins[1], spin
            ):
                return False
            in_spins[0] = Spin(in_spins[0].magnitude, helicity_diff)
            return not _is_clebsch_gordan_coefficient_zero(
                ang_mom, spin, in_spins[0]
            )
        return False


@rule_conditions(
    variable_conditions=[
        (StateQuantumNumberNames.Spin, [DefinedForAllEdges()]),
    ]
)
class HelicityConservation(Rule):
    def __call__(self, ingoing_part_qns, outgoing_part_qns, _):
        r"""Implementation of helicity conservation.

        Check for :math:`|\lambda_2-\lambda_3| \leq S_1`.
        """
        if len(ingoing_part_qns) == 1 and len(outgoing_part_qns) == 2:
            spin_label = StateQuantumNumberNames.Spin

            mother_spin = ingoing_part_qns[0][spin_label].magnitude
            daughter_hel = [
                x[spin_label].projection for x in outgoing_part_qns
            ]
            if mother_spin >= abs(daughter_hel[0] - daughter_hel[1]):
                return True
        return False


@rule_conditions(
    variable_conditions=[
        (StateQuantumNumberNames.Charge, [DefinedForAllEdges()]),
        (StateQuantumNumberNames.IsoSpin, [DefinedForAllEdges()]),
        (StateQuantumNumberNames.Strangeness),
        (StateQuantumNumberNames.Charmness),
        (StateQuantumNumberNames.Bottomness),
        (StateQuantumNumberNames.Topness),
        (StateQuantumNumberNames.BaryonNumber),
        (StateQuantumNumberNames.ElectronLN),
        (StateQuantumNumberNames.MuonLN),
        (StateQuantumNumberNames.TauLN),
    ]
)
class GellMannNishijimaRule(Rule):
    def __call__(self, ingoing_part_qns, outgoing_part_qns, _):
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
                isospin_3 = particle[isospin_label].projection
            if float(particle[charge_label]) != (
                isospin_3 + 0.5 * _calculate_hypercharge(particle)
            ):
                return False
        return True


def _calculate_hypercharge(particle):
    """Calculate the hypercharge :math:`Y=S+C+B+T+B`."""
    qn_labels = [
        StateQuantumNumberNames.Strangeness,
        StateQuantumNumberNames.Charmness,
        StateQuantumNumberNames.Bottomness,
        StateQuantumNumberNames.Topness,
        StateQuantumNumberNames.BaryonNumber,
    ]
    qn_values = [particle[x] for x in qn_labels if x in particle]
    return sum(qn_values)


@rule_conditions(
    variable_conditions=[
        (ParticlePropertyNames.Mass, [DefinedForAllEdges()]),
        (ParticleDecayPropertyNames.Width),
    ]
)
class MassConservation(Rule):
    """Mass conservation rule."""

    def __init__(self, width_factor):
        self.__width_factor = width_factor
        super().__init__()

    def __call__(self, ingoing_part_qns, outgoing_part_qns, _):
        r"""Implements mass conservation.

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

        return (mass_out - self.__width_factor * width_out) < (
            mass_in + self.__width_factor * width_in
        )
