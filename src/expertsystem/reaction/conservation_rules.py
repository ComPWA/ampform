"""Collection of quantum number conservation rules for particle reactions.

This module is the place where the 'expert' defines the rules that verify
quantum numbers of the reaction.

A rule is a function that takes quantum numbers as input and outputs a boolean.
There are three different types of rules:

1. `GraphElementRule` that work on individual graph edges or nodes.
2. `EdgeQNConservationRule` that work on the interaction level, which use
   ingoing edges, outgoing edges as arguments.  E.g.: `.ChargeConservation`.
3. `ConservationRule` that work on the interaction level, which use ingoing
   edges, outgoing edges and a interaction node as arguments. E.g:
   `.parity_conservation`.

The arguments can be any type of quantum number. However a rule argument
resembling edges only accepts `~.quantum_numbers.EdgeQuantumNumbers`. Similarly
arguments that resemble a node only accept
`~.quantum_numbers.NodeQuantumNumbers`. The argument types do not have to be
limited to a single quantum number, but can be a composite (see
`.CParityEdgeInput`).

.. warning::
    Besides the rule logic itself, a rule also has the responsibility of
    stating its run conditions. These run conditions **must** be stated by
    the type annotations of its :code:`__call__` method. The type annotations
    therefore are not just there for *static* type checking: they also
    carry more information about the rule that is extracted *dynamically*
    by the `.reaction` module.

Generally, the conditions can be separated into two categories:

* variable conditions
* toplogical conditions

Currently, only variable conditions are being used. Topological conditions
could be created in the form of `~typing.Tuple` instead of `~typing.List`.

For additive quantum numbers, the decorator `additive_quantum_number_rule`
can be used to automatically generate the appropriate behavior.


The module is therefore strongly typed (both
for the reader of the code and for type checking with :doc:`mypy
<mypy:index>`). An example is `.HelicityParityEdgeInput`, which has been
defined to provide type checks on `.parity_conservation_helicity`.
"""

from copy import deepcopy
from functools import reduce
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import attr

from .combinatorics import arange
from .quantum_numbers import EdgeQuantumNumbers, NodeQuantumNumbers

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore


def _is_boson(spin_magnitude: float) -> bool:
    return abs(spin_magnitude % 1) < 0.01


def _is_particle_antiparticle_pair(pid1: int, pid2: int) -> bool:
    # we just check if the pid is opposite in sign
    # this is a requirement of the pid numbers of course
    return pid1 == -pid2


class GraphElementRule(Protocol):
    def __call__(self, __qns: Any) -> bool:
        ...


class EdgeQNConservationRule(Protocol):
    def __call__(
        self, __ingoing_edge_qns: List[Any], __outgoing_edge_qns: List[Any]
    ) -> bool:
        ...


class ConservationRule(Protocol):
    def __call__(
        self,
        __ingoing_edge_qns: List[Any],
        __outgoing_edge_qns: List[Any],
        __node_qns: Any,
    ) -> bool:
        ...


# Note a generic would be more fitting here. However the type annotations of
# __call__ method in a concrete version of the generic are still containing the
# TypeVar types. See https://github.com/python/typing/issues/762
def additive_quantum_number_rule(
    quantum_number: type,
) -> Callable[[Any], EdgeQNConservationRule]:
    r"""Class decorator for creating an additive conservation rule.

    Use this decorator to create a `EdgeQNConservationRule` for a quantum number
    to which an additive conservation rule applies:

    .. math:: \sum q_{in} = \sum q_{out}

    Args:
        quantum_number: Quantum number to which you want to apply the additive
            conservation check. An example would be
            `.EdgeQuantumNumbers.charge`.
    """

    def decorator(rule_class: Any) -> EdgeQNConservationRule:
        def new_call(  # type: ignore
            self,  # pylint: disable=unused-argument
            ingoing_edge_qns: List[quantum_number],  # type: ignore
            outgoing_edge_qns: List[quantum_number],  # type: ignore
        ) -> bool:
            return sum(ingoing_edge_qns) == sum(outgoing_edge_qns)

        rule_class.__call__ = new_call
        rule_class.__doc__ = (
            f"""Decorated via `{additive_quantum_number_rule.__name__}`.\n\n"""
            f"""Check for `~.EdgeQuantumNumbers.{quantum_number.__name__}` conservation."""
        )
        return rule_class

    return decorator


@additive_quantum_number_rule(EdgeQuantumNumbers.charge)
class ChargeConservation(EdgeQNConservationRule):
    pass


@additive_quantum_number_rule(EdgeQuantumNumbers.baryon_number)
class BaryonNumberConservation(EdgeQNConservationRule):
    pass


@additive_quantum_number_rule(EdgeQuantumNumbers.electron_lepton_number)
class ElectronLNConservation(EdgeQNConservationRule):
    pass


@additive_quantum_number_rule(EdgeQuantumNumbers.muon_lepton_number)
class MuonLNConservation(EdgeQNConservationRule):
    pass


@additive_quantum_number_rule(EdgeQuantumNumbers.tau_lepton_number)
class TauLNConservation(EdgeQNConservationRule):
    pass


@additive_quantum_number_rule(EdgeQuantumNumbers.strangeness)
class StrangenessConservation(EdgeQNConservationRule):
    pass


@additive_quantum_number_rule(EdgeQuantumNumbers.charmness)
class CharmConservation(EdgeQNConservationRule):
    pass


@additive_quantum_number_rule(EdgeQuantumNumbers.bottomness)
class BottomnessConservation(EdgeQNConservationRule):
    pass


def parity_conservation(
    ingoing_edge_qns: List[EdgeQuantumNumbers.parity],
    outgoing_edge_qns: List[EdgeQuantumNumbers.parity],
    l_mag: NodeQuantumNumbers.l_magnitude,
) -> bool:
    r"""Implement :math:`P_{in} = P_{out} \cdot (-1)^L`."""
    if len(ingoing_edge_qns) == 1 and len(outgoing_edge_qns) == 2:
        parity_in = reduce(lambda x, y: x * y.value, ingoing_edge_qns, 1)
        parity_out = reduce(lambda x, y: x * y.value, outgoing_edge_qns, 1)
        return parity_in == (parity_out * (-1) ** l_mag)
    return True


@attr.s(frozen=True)
class HelicityParityEdgeInput:
    parity: EdgeQuantumNumbers.parity = attr.ib()
    spin_mag: EdgeQuantumNumbers.spin_magnitude = attr.ib()
    spin_proj: EdgeQuantumNumbers.spin_projection = attr.ib()


def parity_conservation_helicity(
    ingoing_edge_qns: List[HelicityParityEdgeInput],
    outgoing_edge_qns: List[HelicityParityEdgeInput],
    parity_prefactor: NodeQuantumNumbers.parity_prefactor,
) -> bool:
    r"""Implements parity conservation for helicity formalism.

    Check the following:

    .. math:: A_{-\lambda_1-\lambda_2} = P_1 P_2 P_3 (-1)^{S_2+S_3-S_1}
        A_{\lambda_1\lambda_2}

    .. math:: \mathrm{parity\,prefactor} = P_1 P_2 P_3 (-1)^{S_2+S_3-S_1}

    .. note:: Only the special case :math:`\lambda_1=\lambda_2=0` may
      return False independent on the parity prefactor.
    """
    if len(ingoing_edge_qns) == 1 and len(outgoing_edge_qns) == 2:
        out_spins = [x.spin_mag for x in outgoing_edge_qns]
        parity_product = reduce(
            lambda x, y: x * y.parity.value if y.parity else x,
            ingoing_edge_qns + outgoing_edge_qns,
            1,
        )

        prefactor = parity_product * (-1.0) ** (
            sum(out_spins) - ingoing_edge_qns[0].spin_mag
        )

        if (
            all(x.spin_proj == 0.0 for x in outgoing_edge_qns)
            and prefactor == -1
        ):
            return False

        return prefactor == parity_prefactor
    return True


@attr.s(frozen=True)
class CParityEdgeInput:
    spin_mag: EdgeQuantumNumbers.spin_magnitude = attr.ib()
    pid: EdgeQuantumNumbers.pid = attr.ib()
    c_parity: Optional[EdgeQuantumNumbers.c_parity] = attr.ib(default=None)


@attr.s(frozen=True)
class CParityNodeInput:
    l_mag: NodeQuantumNumbers.l_magnitude = attr.ib()
    s_mag: NodeQuantumNumbers.s_magnitude = attr.ib()


def c_parity_conservation(
    ingoing_edge_qns: List[CParityEdgeInput],
    outgoing_edge_qns: List[CParityEdgeInput],
    interaction_node_qns: CParityNodeInput,
) -> bool:
    """Check for :math:`C`-parity conservation.

    Implements :math:`C_{in} = C_{out}`.
    """

    def _get_c_parity_multiparticle(
        part_qns: List[CParityEdgeInput], interaction_qns: CParityNodeInput
    ) -> Optional[int]:
        c_parities_part = [x.c_parity.value for x in part_qns if x.c_parity]
        # if all states have C parity defined, then just multiply them
        if len(c_parities_part) == len(part_qns):
            return reduce(lambda x, y: x * y, c_parities_part, 1)

        # two particle case
        if len(part_qns) == 2:
            if _is_particle_antiparticle_pair(
                part_qns[0].pid, part_qns[1].pid
            ):
                ang_mom = interaction_qns.l_mag
                # if boson
                if _is_boson(part_qns[0].spin_mag):
                    return (-1) ** int(ang_mom)
                coupled_spin = interaction_qns.s_mag
                if isinstance(coupled_spin, int) or coupled_spin.is_integer():
                    return (-1) ** int(ang_mom + coupled_spin)
        return None

    c_parity_in = _get_c_parity_multiparticle(
        ingoing_edge_qns, interaction_node_qns
    )
    if c_parity_in is None:
        return True

    c_parity_out = _get_c_parity_multiparticle(
        outgoing_edge_qns, interaction_node_qns
    )
    if c_parity_out is None:
        return True

    return c_parity_in == c_parity_out


@attr.s(frozen=True)
class GParityEdgeInput:
    isospin: EdgeQuantumNumbers.isospin_magnitude = attr.ib()
    spin_mag: EdgeQuantumNumbers.spin_magnitude = attr.ib()
    pid: EdgeQuantumNumbers.pid = attr.ib()
    g_parity: Optional[EdgeQuantumNumbers.g_parity] = attr.ib(default=None)


@attr.s(frozen=True)
class GParityNodeInput:
    l_mag: NodeQuantumNumbers.l_magnitude = attr.ib()
    s_mag: NodeQuantumNumbers.s_magnitude = attr.ib()


def g_parity_conservation(
    ingoing_edge_qns: List[GParityEdgeInput],
    outgoing_edge_qns: List[GParityEdgeInput],
    interaction_qns: GParityNodeInput,
) -> bool:
    """Check for :math:`G`-parity conservation.

    Implements for :math:`G_{in} = G_{out}`.
    """

    def check_multistate_g_parity(
        isospin: EdgeQuantumNumbers.isospin_magnitude,
        double_state_qns: Tuple[GParityEdgeInput, GParityEdgeInput],
    ) -> Optional[int]:
        if _is_particle_antiparticle_pair(
            double_state_qns[0].pid, double_state_qns[1].pid
        ):
            ang_mom = interaction_qns.l_mag
            if isinstance(isospin, int) or isospin.is_integer():
                # if boson
                if _is_boson(double_state_qns[0].spin_mag):
                    return (-1) ** int(ang_mom + isospin)
                coupled_spin = interaction_qns.s_mag
                if isinstance(coupled_spin, int) or coupled_spin.is_integer():
                    return (-1) ** int(ang_mom + coupled_spin + isospin)
        return None

    def check_g_parity_isobar(
        single_state: GParityEdgeInput,
        couple_state: Tuple[GParityEdgeInput, GParityEdgeInput],
    ) -> bool:
        couple_state_g_parity = check_multistate_g_parity(
            single_state.isospin,
            (couple_state[0], couple_state[1]),
        )
        single_state_g_parity = (
            ingoing_edge_qns[0].g_parity.value
            if ingoing_edge_qns[0].g_parity
            else None
        )

        if not couple_state_g_parity or not single_state_g_parity:
            return True
        return couple_state_g_parity == single_state_g_parity

    no_g_parity_in_part = [
        True for x in ingoing_edge_qns if x.g_parity is None
    ]
    no_g_parity_out_part = [
        True for x in outgoing_edge_qns if x.g_parity is None
    ]
    # if all states have G parity defined, then just multiply them
    if not any(no_g_parity_in_part + no_g_parity_out_part):
        in_g_parity = reduce(
            lambda x, y: x * y.g_parity.value if y.g_parity else x,
            ingoing_edge_qns,
            1,
        )
        out_g_parity = reduce(
            lambda x, y: x * y.g_parity.value if y.g_parity else x,
            outgoing_edge_qns,
            1,
        )
        return in_g_parity == out_g_parity

    # two particle case
    particle_counts = (len(ingoing_edge_qns), len(outgoing_edge_qns))
    if particle_counts == (1, 2):
        return check_g_parity_isobar(
            ingoing_edge_qns[0],
            (outgoing_edge_qns[0], outgoing_edge_qns[1]),
        )

    if particle_counts == (2, 1):
        return check_g_parity_isobar(
            outgoing_edge_qns[0],
            (ingoing_edge_qns[0], ingoing_edge_qns[1]),
        )
    return True


@attr.s(frozen=True)
class IdenticalParticleSymmetryOutEdgeInput:
    spin_magnitude: EdgeQuantumNumbers.spin_magnitude = attr.ib()
    spin_projection: EdgeQuantumNumbers.spin_projection = attr.ib()
    pid: EdgeQuantumNumbers.pid = attr.ib()


def identical_particle_symmetrization(
    ingoing_parities: List[EdgeQuantumNumbers.parity],
    outgoing_edge_qns: List[IdenticalParticleSymmetryOutEdgeInput],
) -> bool:
    """Verifies multi particle state symmetrization for identical particles.

    In case of a multi particle state with identical particles, their exchange
    symmetry has to follow the spin statistic theorem.

    For bosonic systems the total exchange symmetry (parity) has to be even
    (+1). For fermionic systems the total exchange symmetry (parity) has to be
    odd (-1).

    In case of a particle decaying into N identical particles (N>1), the
    decaying particle has to have the same parity as required by the spin
    statistic theorem of the multi body state.
    """

    def _check_particles_identical(
        particles: List[IdenticalParticleSymmetryOutEdgeInput],
    ) -> bool:
        """Check if pids and spins match."""
        if len(particles) == 1:
            return False

        reference_pid = particles[0].pid
        reference_spin_proj = particles[0].spin_projection
        for particle in particles[1:]:
            if particle.pid != reference_pid:
                return False
            if particle.spin_projection != reference_spin_proj:
                return False
        return True

    if len(ingoing_parities) == 1:
        if _check_particles_identical(outgoing_edge_qns):
            if _is_boson(outgoing_edge_qns[0].spin_magnitude):
                # we have a boson, check if parity of mother is even
                parity = ingoing_parities[0]
                if parity == -1:
                    # if its odd then return False
                    return False
            else:
                # its fermion
                parity = ingoing_parities[0]
                if parity == 1:
                    return False

    return True


@attr.s(frozen=True)
class _Spin:
    magnitude: float = attr.ib()
    projection: float = attr.ib()


def _is_clebsch_gordan_coefficient_zero(
    spin1: _Spin, spin2: _Spin, spin_coupled: _Spin
) -> bool:
    m_1 = spin1.projection
    j_1 = spin1.magnitude
    m_2 = spin2.projection
    j_2 = spin2.magnitude
    proj = spin_coupled.projection
    mag = spin_coupled.magnitude
    if (j_1 == j_2 and m_1 == m_2) or (m_1 == 0.0 and m_2 == 0.0):
        if abs(mag - j_1 - j_2) % 2 == 1:
            return True
    if j_1 == mag and m_1 == -proj:
        if abs(j_2 - j_1 - mag) % 2 == 1:
            return True
    if j_2 == mag and m_2 == -proj:
        if abs(j_1 - j_2 - mag) % 2 == 1:
            return True
    return False


@attr.s(frozen=True)
class SpinNodeInput:
    l_magnitude: NodeQuantumNumbers.l_magnitude = attr.ib()
    l_projection: NodeQuantumNumbers.l_projection = attr.ib()
    s_magnitude: NodeQuantumNumbers.s_magnitude = attr.ib()
    s_projection: NodeQuantumNumbers.s_projection = attr.ib()


@attr.s(frozen=True)
class SpinMagnitudeNodeInput:
    l_magnitude: NodeQuantumNumbers.l_magnitude = attr.ib()
    s_magnitude: NodeQuantumNumbers.s_magnitude = attr.ib()


def ls_spin_validity(spin_input: SpinNodeInput) -> bool:
    r"""Check for valid isospin magnitude and projection."""
    return _check_spin_valid(
        float(spin_input.l_magnitude), float(spin_input.l_projection)
    ) and _check_spin_valid(
        float(spin_input.s_magnitude), float(spin_input.s_projection)
    )


def _check_magnitude(
    in_part: List[float],
    out_part: List[float],
    interaction_qns: Optional[Union[SpinMagnitudeNodeInput, SpinNodeInput]],
) -> bool:
    def couple_mags(j_1: float, j_2: float) -> List[float]:
        return [
            x / 2.0
            for x in range(
                int(2 * abs(j_1 - j_2)), int(2 * (j_1 + j_2 + 1)), 2
            )
        ]

    def couple_magnitudes(
        magnitudes: List[float],
        interaction_qns: Optional[
            Union[SpinMagnitudeNodeInput, SpinNodeInput]
        ],
    ) -> Set[float]:
        if len(magnitudes) == 1:
            return set(magnitudes)

        coupled_magnitudes = set([magnitudes[0]])
        for mag in magnitudes[1:]:
            temp_set = coupled_magnitudes
            coupled_magnitudes = set()
            for ref_mag in temp_set:
                coupled_magnitudes.update(couple_mags(mag, ref_mag))

        if interaction_qns:
            if interaction_qns.s_magnitude in coupled_magnitudes:
                return set(
                    couple_mags(
                        interaction_qns.s_magnitude,
                        interaction_qns.l_magnitude,
                    )
                )
            return set()  # in case there the spin coupling fails
        return coupled_magnitudes

    in_tot_spins = couple_magnitudes(in_part, interaction_qns)
    out_tot_spins = couple_magnitudes(out_part, interaction_qns)
    matching_spins = in_tot_spins.intersection(out_tot_spins)

    if len(matching_spins) > 0:
        return True
    return False


def _check_spin_couplings(
    in_part: List[_Spin],
    out_part: List[_Spin],
    interaction_qns: Optional[SpinNodeInput],
) -> bool:
    in_tot_spins = __calculate_total_spins(in_part, interaction_qns)
    out_tot_spins = __calculate_total_spins(out_part, interaction_qns)
    matching_spins = in_tot_spins & out_tot_spins
    if len(matching_spins) > 0:
        return True
    return False


def __calculate_total_spins(
    spins: List[_Spin],
    interaction_qns: Optional[SpinNodeInput] = None,
) -> Set[_Spin]:
    total_spins = set()
    if len(spins) == 1:
        return set(spins)
    total_spins = __create_coupled_spins(spins)
    if interaction_qns:
        coupled_spin = _Spin(
            interaction_qns.s_magnitude, interaction_qns.s_projection
        )
        if coupled_spin in total_spins:
            return __spin_couplings(
                coupled_spin,
                _Spin(
                    interaction_qns.l_magnitude, interaction_qns.l_projection
                ),
            )
        total_spins = set()

    return total_spins


def __create_coupled_spins(spins: List[_Spin]) -> Set[_Spin]:
    """Creates all combinations of coupled spins."""
    spins_daughters_coupled: Set[_Spin] = set()
    spin_list = deepcopy(spins)
    while spin_list:
        if spins_daughters_coupled:
            temp_coupled_spins = set()
            tempspin = spin_list.pop()
            for spin in spins_daughters_coupled:
                coupled_spins = __spin_couplings(spin, tempspin)
                temp_coupled_spins.update(coupled_spins)
            spins_daughters_coupled = temp_coupled_spins
        else:
            spins_daughters_coupled.add(spin_list.pop())

    return spins_daughters_coupled


def __spin_couplings(spin1: _Spin, spin2: _Spin) -> Set[_Spin]:
    r"""Implement the coupling of two spins.

    :math:`|S_1 - S_2| \leq S \leq |S_1 + S_2|` and :math:`M_1 + M_2 = M`
    """
    s_1 = spin1.magnitude
    s_2 = spin2.magnitude

    sum_proj = spin1.projection + spin2.projection
    return set(
        _Spin(x, sum_proj)
        for x in arange(abs(s_1 - s_2), s_1 + s_2 + 1, 1.0)
        if x >= abs(sum_proj)
        and not _is_clebsch_gordan_coefficient_zero(
            spin1, spin2, _Spin(x, sum_proj)
        )
    )


@attr.s
class IsoSpinEdgeInput:
    isospin_mag: EdgeQuantumNumbers.isospin_magnitude = attr.ib()
    isospin_proj: EdgeQuantumNumbers.isospin_projection = attr.ib()


def _check_spin_valid(magnitude: float, projection: float) -> bool:
    if magnitude % 0.5 != 0.0:
        return False
    if abs(projection) > magnitude:
        return False
    return float(projection - magnitude).is_integer()


def isospin_validity(isospin: IsoSpinEdgeInput) -> bool:
    r"""Check for valid isospin magnitude and projection."""
    return _check_spin_valid(
        float(isospin.isospin_mag), float(isospin.isospin_proj)
    )


def isospin_conservation(
    ingoing_isospins: List[IsoSpinEdgeInput],
    outgoing_isospins: List[IsoSpinEdgeInput],
) -> bool:
    r"""Check for isospin conservation.

    Implements

    .. math::
        |I_1 - I_2| \leq I \leq |I_1 + I_2|

    Also checks :math:`I_{1,z} + I_{2,z} = I_z` and if Clebsch-Gordan
    coefficients are all 0.
    """
    if not sum([x.isospin_proj for x in ingoing_isospins]) == sum(
        [x.isospin_proj for x in outgoing_isospins]
    ):
        return False
    if not all(
        isospin_validity(x) for x in ingoing_isospins + outgoing_isospins
    ):
        return False
    return _check_spin_couplings(
        [_Spin(x.isospin_mag, x.isospin_proj) for x in ingoing_isospins],
        [_Spin(x.isospin_mag, x.isospin_proj) for x in outgoing_isospins],
        None,
    )


@attr.s
class SpinEdgeInput:
    spin_magnitude: EdgeQuantumNumbers.spin_magnitude = attr.ib()
    spin_projection: EdgeQuantumNumbers.spin_projection = attr.ib()


def spin_validity(spin: SpinEdgeInput) -> bool:
    r"""Check for valid spin magnitude and projection."""
    return _check_spin_valid(
        float(spin.spin_magnitude), float(spin.spin_projection)
    )


def spin_conservation(
    ingoing_spins: List[SpinEdgeInput],
    outgoing_spins: List[SpinEdgeInput],
    interaction_qns: SpinNodeInput,
) -> bool:
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
    # L and S can only be used if one side is a single state
    # and the other side contains of two states (isobar)
    # So do a full check if this is the case
    if (len(ingoing_spins) == 1 and len(outgoing_spins) == 2) or (
        len(ingoing_spins) == 2 and len(outgoing_spins) == 1
    ):

        return _check_spin_couplings(
            [
                _Spin(x.spin_magnitude, x.spin_projection)
                for x in ingoing_spins
            ],
            [
                _Spin(x.spin_magnitude, x.spin_projection)
                for x in outgoing_spins
            ],
            interaction_qns,
        )

    # otherwise don't use S and L and just check magnitude
    # are integral or non integral on both sides
    return (
        sum([float(x.spin_magnitude) for x in ingoing_spins]).is_integer()
        == sum([float(x.spin_magnitude) for x in outgoing_spins]).is_integer()
    )


def spin_magnitude_conservation(
    ingoing_spins: List[SpinEdgeInput],
    outgoing_spins: List[SpinEdgeInput],
    interaction_qns: SpinMagnitudeNodeInput,
) -> bool:
    r"""Check for spin conservation.

    Implements

    .. math::
        |S_1 - S_2| \leq S \leq |S_1 + S_2|

    and

    .. math::
        |L - S| \leq J \leq |L + S|
    """
    # L and S can only be used if one side is a single state
    # and the other side contains of two states (isobar)
    # So do a full check if this is the case
    if (len(ingoing_spins) == 1 and len(outgoing_spins) == 2) or (
        len(ingoing_spins) == 2 and len(outgoing_spins) == 1
    ):
        return _check_magnitude(
            [x.spin_magnitude for x in ingoing_spins],
            [x.spin_magnitude for x in outgoing_spins],
            interaction_qns,
        )

    # otherwise don't use S and L and just check magnitude
    # are integral or non integral on both sides
    return (
        sum([float(x.spin_magnitude) for x in ingoing_spins]).is_integer()
        == sum([float(x.spin_magnitude) for x in outgoing_spins]).is_integer()
    )


def clebsch_gordan_helicity_to_canonical(
    ingoing_spins: List[SpinEdgeInput],
    outgoing_spins: List[SpinEdgeInput],
    interaction_qns: SpinNodeInput,
) -> bool:
    """Implement Clebsch-Gordan checks.

    For :math:`S_1, S_2` to :math:`S` and the :math:`L,S` to :math:`J`
    coupling based on the conversion of helicity to canonical amplitude sums.

    .. note:: This rule does not check that the spin magnitudes couple
      correctly to L and S, as this is already performed by
      `~.spin_magnitude_conservation`.
    """
    if len(ingoing_spins) == 1 and len(outgoing_spins) == 2:
        out_spin1 = _Spin(
            outgoing_spins[0].spin_magnitude,
            outgoing_spins[0].spin_projection,
        )
        out_spin2 = _Spin(
            outgoing_spins[1].spin_magnitude,
            -outgoing_spins[1].spin_projection,
        )

        helicity_diff = out_spin1.projection + out_spin2.projection
        if helicity_diff != interaction_qns.s_projection:
            return False

        ang_mom = _Spin(
            interaction_qns.l_magnitude, interaction_qns.l_projection
        )
        coupled_spin = _Spin(
            interaction_qns.s_magnitude, interaction_qns.s_projection
        )
        parent_spin = ingoing_spins[0].spin_magnitude

        coupled_spin = _Spin(coupled_spin.magnitude, helicity_diff)
        if not _check_spin_valid(
            coupled_spin.magnitude, coupled_spin.projection
        ):
            return False
        in_spin = _Spin(parent_spin, helicity_diff)
        if not _check_spin_valid(in_spin.magnitude, in_spin.projection):
            return False

        if _is_clebsch_gordan_coefficient_zero(
            out_spin1, out_spin2, coupled_spin
        ):
            return False

        return not _is_clebsch_gordan_coefficient_zero(
            ang_mom, coupled_spin, in_spin
        )
    return False


def helicity_conservation(
    ingoing_spin_mags: List[EdgeQuantumNumbers.spin_magnitude],
    outgoing_helicities: List[EdgeQuantumNumbers.spin_projection],
) -> bool:
    r"""Implementation of helicity conservation.

    Check for :math:`|\lambda_2-\lambda_3| \leq S_1`.
    """
    if len(ingoing_spin_mags) == 1 and len(outgoing_helicities) == 2:
        mother_spin = ingoing_spin_mags[0]
        if mother_spin >= abs(outgoing_helicities[0] - outgoing_helicities[1]):
            return True
    return False


@attr.s(frozen=True)
class GellMannNishijimaInput:
    # pylint: disable=too-many-instance-attributes
    charge: EdgeQuantumNumbers.charge = attr.ib()
    isospin_proj: Optional[EdgeQuantumNumbers.isospin_projection] = attr.ib(
        None
    )
    strangeness: Optional[EdgeQuantumNumbers.strangeness] = attr.ib(None)
    charmness: Optional[EdgeQuantumNumbers.charmness] = attr.ib(None)
    bottomness: Optional[EdgeQuantumNumbers.bottomness] = attr.ib(None)
    topness: Optional[EdgeQuantumNumbers.topness] = attr.ib(None)
    baryon_number: Optional[EdgeQuantumNumbers.baryon_number] = attr.ib(None)
    electron_ln: Optional[EdgeQuantumNumbers.electron_lepton_number] = attr.ib(
        None
    )
    muon_ln: Optional[EdgeQuantumNumbers.muon_lepton_number] = attr.ib(None)
    tau_ln: Optional[EdgeQuantumNumbers.tau_lepton_number] = attr.ib(None)


def gellmann_nishijima(edge_qns: GellMannNishijimaInput) -> bool:
    r"""Check the Gell-Mann–Nishijima formula.

    :math:`Q=I_3+\frac{Y}{2}` for each particle.
    """

    def calculate_hypercharge(
        particle: GellMannNishijimaInput,
    ) -> float:
        """Calculate the hypercharge :math:`Y=S+C+B+T+B`."""
        return sum(
            [
                0.0 if x is None else x
                for x in [
                    particle.strangeness,
                    particle.charmness,
                    particle.bottomness,
                    particle.topness,
                    particle.baryon_number,
                ]
            ]
        )

    if edge_qns.electron_ln or edge_qns.muon_ln or edge_qns.tau_ln:
        return True
    isospin_3 = 0.0
    if edge_qns.isospin_proj:
        isospin_3 = edge_qns.isospin_proj
    if float(edge_qns.charge) != (
        isospin_3 + 0.5 * calculate_hypercharge(edge_qns)
    ):
        return False
    return True


@attr.s(frozen=True)
class MassEdgeInput:
    mass: EdgeQuantumNumbers.mass = attr.ib()
    width: Optional[EdgeQuantumNumbers.width] = attr.ib(default=None)


class MassConservation:
    """Mass conservation rule."""

    def __init__(self, width_factor: float):
        self.__width_factor = width_factor

    def __call__(
        self,
        ingoing_edge_qns: List[MassEdgeInput],
        outgoing_edge_qns: List[MassEdgeInput],
    ) -> bool:
        r"""Implements mass conservation.

        :math:`M_{out} - N \cdot W_{out} < M_{in} + N \cdot W_{in}`

        It makes sure that the net mass outgoing state :math:`M_{out}` is
        smaller than the net mass of the ingoing state :math:`M_{in}`. Also the
        width :math:`W` of the states is taken into account.
        """
        mass_in = sum([x.mass for x in ingoing_edge_qns])
        width_in = sum([x.width for x in ingoing_edge_qns if x.width])
        mass_out = sum([x.mass for x in outgoing_edge_qns])
        width_out = sum([x.width for x in outgoing_edge_qns if x.width])

        return (mass_out - self.__width_factor * width_out) < (
            mass_in + self.__width_factor * width_in
        )
