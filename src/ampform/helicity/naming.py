"""Generate descriptions used in the `~ampform.helicity` formalism."""

from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cache
from typing import TYPE_CHECKING

import sympy as sp
from qrules.transition import ReactionInfo, State, StateTransition

from ampform.helicity.decay import (
    assert_isobar_topology,
    determine_attached_final_state,
    get_helicity_info,
    get_outer_state_ids,
    get_sorted_states,
    group_by_spin_projection,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
if TYPE_CHECKING:
    from collections.abc import Iterable
    from fractions import Fraction

    from qrules.topology import Topology


class NameGenerator(ABC):
    """Name generator for amplitudes and coefficients in a `.HelicityModel`.

    .. seealso:: :ref:`usage/helicity/formalism:Coefficient names`
    """

    @abstractmethod
    def generate_amplitude_name(
        self, transition: StateTransition, node_id: int | None = None
    ) -> str:
        """Generates a unique name for the amplitude corresponding.

        That is, corresponging to the given `~qrules.topology.Transition`. If
        ``node_id`` is given, it generates a unique name for the partial amplitude
        corresponding to the interaction node of the given
        `~qrules.topology.Transition`.
        """

    @abstractmethod
    def generate_sequential_amplitude_suffix(self, transition: StateTransition) -> str:
        """Generate unique suffix for a sequential amplitude transition."""

    @abstractmethod
    def generate_two_body_decay_suffix(
        self, transition: StateTransition, node_id: int
    ) -> str:
        """Generate partial amplitude coefficient name suffix."""

    @property
    @abstractmethod
    def parity_partner_coefficient_mapping(self) -> dict[str, str]: ...


class HelicityAmplitudeNameGenerator(NameGenerator):
    def __init__(
        self,
        transitions: ReactionInfo | Iterable[StateTransition],
        insert_parent_helicities: bool = False,
        insert_child_helicities: bool = True,
    ) -> None:
        if isinstance(transitions, ReactionInfo):
            transitions = transitions.transitions
        self.__transitions = transitions
        self.__insert_parent_helicities = insert_parent_helicities
        self.__insert_child_helicities = insert_child_helicities
        self._register_amplitude_coefficients()

    @property
    def parity_partner_coefficient_mapping(self) -> dict[str, str]:
        return self.__parity_partner_coefficient_mapping

    @property
    def insert_parent_helicities(self) -> bool:
        """Insert helicities of each parent state in the coefficient names."""
        return self.__insert_parent_helicities

    @insert_parent_helicities.setter
    def insert_parent_helicities(self, value: bool) -> None:
        self.__insert_parent_helicities = value
        self._register_amplitude_coefficients()

    @property
    def insert_child_helicities(self) -> bool:
        """Embed the helicity of each decay product in the coefficients."""
        return self.__insert_child_helicities

    @insert_child_helicities.setter
    def insert_child_helicities(self, value: bool) -> None:
        self.__insert_child_helicities = value
        self._register_amplitude_coefficients()

    def _register_amplitude_coefficients(self) -> None:
        self.__parity_partner_coefficient_mapping: dict[str, str] = {}
        for transition in self.__transitions:
            self.__register_amplitude_coefficient_name(transition)

    def __register_amplitude_coefficient_name(
        self, transition: StateTransition
    ) -> None:
        for node_id in transition.topology.nodes:
            (
                coefficient_suffix,
                parity_partner_coefficient_suffix,
                priority_partner_coefficient_suffix,
            ) = self.__generate_amplitude_coefficient_couple(transition, node_id)

            if transition.interactions[node_id].parity_prefactor is None:
                continue

            if coefficient_suffix not in self.__parity_partner_coefficient_mapping:
                if (
                    parity_partner_coefficient_suffix
                    in self.__parity_partner_coefficient_mapping
                ):
                    if (
                        parity_partner_coefficient_suffix
                        == priority_partner_coefficient_suffix
                    ):
                        self.__parity_partner_coefficient_mapping[
                            coefficient_suffix
                        ] = parity_partner_coefficient_suffix
                    else:
                        self.__parity_partner_coefficient_mapping[
                            parity_partner_coefficient_suffix
                        ] = coefficient_suffix
                        self.__parity_partner_coefficient_mapping[
                            coefficient_suffix
                        ] = coefficient_suffix

                else:
                    # if neither this coefficient nor its partner are registered just add it
                    self.__parity_partner_coefficient_mapping[coefficient_suffix] = (
                        coefficient_suffix
                    )

    def __generate_amplitude_coefficient_couple(
        self, transition: StateTransition, node_id: int
    ) -> tuple[str, str, str]:
        incoming_state, outgoing_states = get_helicity_info(transition, node_id)
        par_name_suffix = self.generate_two_body_decay_suffix(transition, node_id)

        pp_par_name_suffix = (
            _state_to_str(incoming_state, use_helicity=False)
            + R" \to "
            + " ".join(
                _state_to_str(s, make_parity_partner=True) for s in outgoing_states
            )
        )

        priority_name_suffix = par_name_suffix
        if outgoing_states[0].spin_projection < 0 or (
            outgoing_states[0].spin_projection == 0
            and outgoing_states[1].spin_projection < 0
        ):
            priority_name_suffix = pp_par_name_suffix

        return (par_name_suffix, pp_par_name_suffix, priority_name_suffix)

    def generate_amplitude_name(  # noqa: PLR6301
        self,
        transition: StateTransition,
        node_id: int | None = None,
    ) -> str:
        name = ""
        if node_id is None:
            node_ids = transition.topology.nodes
        else:
            node_ids = frozenset({node_id})
        names: list[str] = []
        for i in node_ids:
            incoming_state, outgoing_states = get_helicity_info(transition, i)
            name = (
                _state_to_str(incoming_state)
                + R" \to "
                + " ".join(_state_to_str(s) for s in outgoing_states)
            )
            names.append(name)
        return "; ".join(names)

    def generate_two_body_decay_suffix(
        self, transition: StateTransition, node_id: int
    ) -> str:
        components = self._get_coefficient_components(transition, node_id)
        return "".join(components)

    def _get_coefficient_components(
        self, transition: StateTransition, node_id: int
    ) -> tuple[str, str, str]:
        in_hel_info, out_hel_info = get_helicity_info(transition, node_id)
        return (
            _state_to_str(
                in_hel_info,
                use_helicity=self.insert_parent_helicities,
            ),
            R" \to ",
            " ".join(
                _state_to_str(
                    state,
                    use_helicity=self.insert_child_helicities,
                )
                for state in out_hel_info
            ),
        )

    def generate_sequential_amplitude_suffix(self, transition: StateTransition) -> str:
        coefficient_names: list[str] = []
        for node_id in transition.topology.nodes:
            suffix = self.generate_two_body_decay_suffix(transition, node_id)
            if suffix in self.parity_partner_coefficient_mapping:
                suffix = self.parity_partner_coefficient_mapping[suffix]
            coefficient_names.append(suffix)
        return "; ".join(coefficient_names)


class CanonicalAmplitudeNameGenerator(HelicityAmplitudeNameGenerator):
    @override
    def __init__(
        self,
        transitions: ReactionInfo | Iterable[StateTransition],
        insert_parent_helicities: bool = False,
        insert_child_helicities: bool = False,
        insert_ls_combinations: bool = True,
    ) -> None:
        self.__insert_ls_combinations = insert_ls_combinations
        super().__init__(
            transitions,
            insert_parent_helicities=insert_parent_helicities,
            insert_child_helicities=insert_child_helicities,
        )

    @property
    def insert_ls_combinations(self) -> bool:
        """Embed each :math:`LS`-combination in the coefficient names."""
        return self.__insert_ls_combinations

    @insert_ls_combinations.setter
    def insert_ls_combinations(self, value: bool) -> None:
        self.__insert_ls_combinations = value
        self._register_amplitude_coefficients()

    @override
    def generate_amplitude_name(
        self,
        transition: StateTransition,
        node_id: int | None = None,
    ) -> str:
        if isinstance(node_id, int):
            node_ids = frozenset({node_id})
        else:
            node_ids = transition.topology.nodes
        names: list[str] = []
        for node in node_ids:
            helicity_name = super().generate_amplitude_name(transition, node)
            canonical_name = helicity_name.replace(
                R" \to ",
                self.__generate_ls_arrow(transition, node),
            )
            names.append(canonical_name)
        return "; ".join(names)

    @override
    def _get_coefficient_components(
        self, transition: StateTransition, node_id: int
    ) -> tuple[str, str, str]:
        components = super()._get_coefficient_components(transition, node_id)
        if not self.insert_ls_combinations:
            return components
        return (
            components[0],
            self.__generate_ls_arrow(transition, node_id),
            components[2],
        )

    @staticmethod
    def __generate_ls_arrow(transition: StateTransition, node_id: int) -> str:
        interaction = transition.interactions[node_id]
        angular_momentum = sp.Rational(interaction.l_magnitude)
        coupled_spin = sp.Rational(interaction.s_magnitude)
        return Rf" \xrightarrow[S={coupled_spin}]{{L={angular_momentum}}} "


def create_amplitude_symbol(transition: StateTransition) -> sp.Indexed:
    outer_state_ids = get_outer_state_ids(transition)
    helicities = tuple(
        sp.Rational(transition.states[i].spin_projection) for i in outer_state_ids
    )
    base = create_amplitude_base(transition.topology)
    return base[helicities]


def create_amplitude_base(topology: Topology) -> sp.IndexedBase:
    superscript = get_topology_identifier(topology)
    return sp.IndexedBase(f"A^{superscript}", complex=True)


def generate_transition_label(transition: StateTransition) -> str:
    r"""Generate a label for a coherent intensity, including spin projection.

    >>> import qrules
    >>> reaction = qrules.generate_transitions(
    ...     initial_state="J/psi(1S)",
    ...     final_state=["gamma", "pi0", "pi0"],
    ...     allowed_intermediate_particles=["f(0)(980)"],
    ... )
    >>> print(generate_transition_label(reaction.transitions[0]))
    J/\psi(1S)_{-1} \to \gamma_{-1} \pi^{0}_{0} \pi^{0}_{0}
    >>> print(generate_transition_label(reaction.transitions[-1]))
    J/\psi(1S)_{+1} \to \gamma_{+1} \pi^{0}_{0} \pi^{0}_{0}
    """
    initial_state_ids = transition.topology.incoming_edge_ids
    final_state_ids = transition.topology.outgoing_edge_ids
    initial_states = get_sorted_states(transition, initial_state_ids)
    final_states = get_sorted_states(transition, final_state_ids)
    return (
        _state_to_str(initial_states[0])
        + R" \to "
        + " ".join(_state_to_str(s) for s in final_states)
    )


def get_helicity_angle_symbols(
    topology: Topology, state_id: int
) -> tuple[sp.Symbol, sp.Symbol]:
    r"""Generate a nested helicity angle label for :math:`\phi,\theta`.

    See :func:`get_boost_chain_suffix` for the meaning of the suffix.
    """
    suffix = get_boost_chain_suffix(topology, state_id)
    return (
        sp.Symbol(f"phi{suffix}", real=True),
        sp.Symbol(f"theta{suffix}", real=True),
    )


@cache
def get_boost_chain_suffix(topology: Topology, state_id: int) -> str:
    """Generate a subscript-superscript to identify a chain of Lorentz boosts.

    The generated subscripts describe the decay sequence from the right to the left,
    separated by commas. Resonance edge IDs are expressed as a sum of the final state
    IDs that lie below them (see :func:`.determine_attached_final_state`). The generated
    label does not state the top-most edge (the initial state).

    Example
    -------
    The following two allowed isobar topologies for a **1-to-5-body** decay illustrates
    how the naming scheme results in a unique label for each of the **eight edges** in
    the decay topology. Note that label only uses final state IDs, but still reflects
    the internal decay topology.

    >>> from qrules.topology import create_isobar_topologies
    >>> from ampform._qrules import get_qrules_version
    >>> topologies = create_isobar_topologies(5)
    >>> topology = topologies[0 if get_qrules_version() < (0, 10) else 3]
    >>> for i in topology.intermediate_edge_ids | topology.outgoing_edge_ids:
    ...     suffix = get_boost_chain_suffix(topology, i)
    ...     print(f"{i}: 'phi{suffix}'")
    0: 'phi_0^034'
    1: 'phi_1^12'
    2: 'phi_2^12'
    3: 'phi_3^34,034'
    4: 'phi_4^34,034'
    5: 'phi_034'
    6: 'phi_12'
    7: 'phi_34^034'
    >>> topology = topologies[1 if get_qrules_version() < (0, 10) else 2]
    >>> for i in topology.intermediate_edge_ids | topology.outgoing_edge_ids:
    ...     suffix = get_boost_chain_suffix(topology, i)
    ...     print(f"{i}: 'phi{suffix}'")
    0: 'phi_0^01'
    1: 'phi_1^01'
    2: 'phi_2^234'
    3: 'phi_3^34,234'
    4: 'phi_4^34,234'
    5: 'phi_01'
    6: 'phi_234'
    7: 'phi_34^234'

    Some labels explained:

    - :code:`phi_12`: **edge 6** on the *left* topology, because for this
      topology, we have :math:`p_6=p_1+p_2`.
    - :code:`phi_234`: **edge 6** *right*, because for this topology,
      :math:`p_6=p_2+p_3+p_4`.
    - :code:`phi_1^12`: **edge 1** *left*, because 1 decays from
      :math:`p_6=p_1+p_2`.
    - :code:`phi_1^01`: **edge 1** *right*, because it decays from
      :math:`p_5=p_0+p_1`.
    - :code:`phi_4^34,234`: **edge 4** *right*, because it decays from edge 7
      (:math:`p_7=p_3+p_4`), which comes from edge 6 (:math:`p_7=p_2+p_3+p_4`).

    As noted, the top-most parent (initial state) is not listed in the label.
    """
    assert_isobar_topology(topology)

    def recursive_label(topology: Topology, state_id: int) -> str:
        edge = topology.edges[state_id]
        if edge.ending_node_id is None:
            label = f"{state_id}"
        else:
            attached_final_state_ids = determine_attached_final_state(
                topology, state_id
            )
            label = "".join(map(str, attached_final_state_ids))
        if edge.originating_node_id is not None:
            incoming_state_ids = topology.get_edge_ids_ingoing_to_node(
                edge.originating_node_id
            )
            state_id = next(iter(incoming_state_ids))
            if state_id not in topology.incoming_edge_ids:
                label += f",{recursive_label(topology, state_id)}"
        return label

    label = recursive_label(topology, state_id)

    index_groups = label.split(",")
    subscript = index_groups[0]
    suffix = f"_{subscript}"
    if len(index_groups) > 1:
        superscript = ",".join(index_groups[1:])
        suffix += f"^{superscript}"
    return suffix


def get_helicity_suffix(topology: Topology, state_id: int) -> str:
    """Create an identifier suffix for a topology.

    Used in :doc:`/usage/helicity/spin-alignment`. Comparable to
    :func:`get_boost_chain_suffix`.
    """
    superscript = get_topology_identifier(topology)
    return f"_{state_id}^{superscript}"


def get_topology_identifier(topology: Topology) -> str:
    """Create an identifier `str` for a `~qrules.topology.Topology`."""
    resonance_names = [
        "".join(__get_resonance_identifier(topology, i))
        for i in topology.intermediate_edge_ids
    ]
    return ",".join(sorted(resonance_names, key=natural_sorting))


def __get_resonance_identifier(topology: Topology, state_id: int) -> str:
    attached_final_state_ids = determine_attached_final_state(topology, state_id)
    return "".join(map(str, attached_final_state_ids))


def natural_sorting(text: str) -> list[float | str]:
    """Function that can be used for natural sort order in :func:`sorted`.

    See `natural sort order
    <https://en.wikipedia.org/wiki/Natural_sort_order>`_.

    >>> sorted(["z11", "z2"], key=natural_sorting)
    ['z2', 'z11']
    """
    # https://stackoverflow.com/a/5967539/13219025
    return [
        __attempt_number_cast(c)
        for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)
    ]


def __attempt_number_cast(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


def _state_to_str(
    state: State,
    use_helicity: bool = True,
    make_parity_partner: bool = False,
) -> str:
    if state.particle.latex is not None:
        output_string = state.particle.latex
    else:
        output_string = state.particle.name
    if use_helicity:
        if make_parity_partner:
            helicity = -1 * state.spin_projection
        else:
            helicity = state.spin_projection
        helicity_str = _render_float(helicity)
        if "_" in output_string:
            output_string = f"{{{output_string}}}"
        output_string += f"_{{{helicity_str}}}"
    return output_string


def _render_float(value: float | Fraction) -> str:
    """Render a `float` nicely as a string.

    >>> _render_float(-0.5)
    '-1/2'
    >>> _render_float(1)
    '+1'
    """
    rational = sp.Rational(value)
    if value > 0:
        return f"+{rational}"
    return str(rational)


def create_helicity_symbol(
    topology: Topology, state_id: int, root: str = "lambda"
) -> sp.Symbol:
    if state_id == -1:  # initial state
        name = "m_A"
    else:
        suffix = get_helicity_suffix(topology, state_id)
        name = f"{root}{suffix}"
    return sp.Symbol(name, rational=True)


def create_spin_projection_symbol(state_id: int) -> sp.Symbol:
    if state_id == -1:  # initial state
        suffix = "_A"
    else:
        suffix = str(state_id)
    return sp.Symbol(f"m{suffix}", rational=True)


def collect_spin_projections(
    reaction: ReactionInfo,
) -> dict[sp.Symbol, set[sp.Rational]]:
    outer_state_ids = get_outer_state_ids(reaction)
    spin_projections: defaultdict[sp.Symbol, set[sp.Rational]] = defaultdict(set)
    spin_groups = group_by_spin_projection(reaction.transitions)
    for group in spin_groups:
        for transition in group:
            for i in outer_state_ids:
                state = transition.states[i]
                symbol = create_spin_projection_symbol(i)
                value = sp.Rational(state.spin_projection)
                spin_projections[symbol].add(value)
    return dict(spin_projections)
