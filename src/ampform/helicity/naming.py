"""Generate descriptions used in the `~ampform.helicity` formalism."""

from typing import Dict, List, Optional, Tuple

import sympy as sp
from qrules.transition import State, StateTransition

from .decay import get_helicity_info, get_sorted_states


class HelicityAmplitudeNameGenerator:
    def __init__(self) -> None:
        self.parity_partner_coefficient_mapping: Dict[str, str] = {}

    def register_amplitude_coefficient_name(
        self, transition: StateTransition
    ) -> None:
        for node_id in transition.topology.nodes:
            (
                coefficient_suffix,
                parity_partner_coefficient_suffix,
                priority_partner_coefficient_suffix,
            ) = self.__generate_amplitude_coefficient_couple(
                transition, node_id
            )

            if transition.interactions[node_id].parity_prefactor is None:
                continue

            if (
                coefficient_suffix
                not in self.parity_partner_coefficient_mapping
            ):
                if (
                    parity_partner_coefficient_suffix
                    in self.parity_partner_coefficient_mapping
                ):
                    if (
                        parity_partner_coefficient_suffix
                        == priority_partner_coefficient_suffix
                    ):
                        self.parity_partner_coefficient_mapping[
                            coefficient_suffix
                        ] = parity_partner_coefficient_suffix
                    else:
                        self.parity_partner_coefficient_mapping[
                            parity_partner_coefficient_suffix
                        ] = coefficient_suffix
                        self.parity_partner_coefficient_mapping[
                            coefficient_suffix
                        ] = coefficient_suffix

                else:
                    # if neither this coefficient nor its partner are registered just add it
                    self.parity_partner_coefficient_mapping[
                        coefficient_suffix
                    ] = coefficient_suffix

    def __generate_amplitude_coefficient_couple(
        self, transition: StateTransition, node_id: int
    ) -> Tuple[str, str, str]:
        incoming_state, outgoing_states = get_helicity_info(
            transition, node_id
        )
        par_name_suffix = self.generate_coefficient_name(transition, node_id)

        pp_par_name_suffix = (
            _state_to_str(incoming_state, use_helicity=False)
            + R" \to "
            + " ".join(
                _state_to_str(s, make_parity_partner=True)
                for s in outgoing_states
            )
        )

        priority_name_suffix = par_name_suffix
        if outgoing_states[0].spin_projection < 0 or (
            outgoing_states[0].spin_projection == 0
            and outgoing_states[1].spin_projection < 0
        ):
            priority_name_suffix = pp_par_name_suffix

        return (par_name_suffix, pp_par_name_suffix, priority_name_suffix)

    def generate_amplitude_name(  # pylint: disable=no-self-use
        self,
        transition: StateTransition,
        node_id: Optional[int] = None,
    ) -> str:
        """Generates a unique name for the amplitude corresponding.

        That is, corresponging to the given
        `~qrules.transition.StateTransition`. If ``node_id`` is given, it
        generates a unique name for the partial amplitude corresponding to the
        interaction node of the given `~qrules.transition.StateTransition`.
        """
        name = ""
        if node_id is None:
            node_ids = transition.topology.nodes
        else:
            node_ids = frozenset({node_id})
        names: List[str] = []
        for i in node_ids:
            incoming_state, outgoing_states = get_helicity_info(transition, i)
            name = (
                _state_to_str(incoming_state)
                + R" \to "
                + " ".join(_state_to_str(s) for s in outgoing_states)
            )
            names.append(name)
        return "; ".join(names)

    def generate_coefficient_name(  # pylint: disable=no-self-use
        self, transition: StateTransition, node_id: int
    ) -> str:
        """Generate partial amplitude coefficient name suffix."""
        in_hel_info, out_hel_info = get_helicity_info(transition, node_id)
        return (
            _state_to_str(in_hel_info, use_helicity=False)
            + R" \to "
            + " ".join(_state_to_str(s) for s in out_hel_info)
        )

    def generate_sequential_amplitude_suffix(
        self, transition: StateTransition
    ) -> str:
        """Generate unique suffix for a sequential amplitude transition."""
        coefficient_names: List[str] = []
        for node_id in transition.topology.nodes:
            suffix = self.generate_coefficient_name(transition, node_id)
            if suffix in self.parity_partner_coefficient_mapping:
                suffix = self.parity_partner_coefficient_mapping[suffix]
            coefficient_names.append(suffix)
        return "; ".join(coefficient_names)


class CanonicalAmplitudeNameGenerator(HelicityAmplitudeNameGenerator):
    def generate_amplitude_name(
        self,
        transition: StateTransition,
        node_id: Optional[int] = None,
    ) -> str:
        if isinstance(node_id, int):
            node_ids = frozenset({node_id})
        else:
            node_ids = transition.topology.nodes
        names: List[str] = []
        for node in node_ids:
            helicity_name = super().generate_amplitude_name(transition, node)
            canonical_name = helicity_name.replace(
                R" \to ",
                self.__generate_ls_arrow(transition, node),
            )
            names.append(canonical_name)
        return "; ".join(names)

    def generate_coefficient_name(
        self, transition: StateTransition, node_id: int
    ) -> str:
        incoming_state, outgoing_states = get_helicity_info(
            transition, node_id
        )
        return (
            _state_to_str(incoming_state, use_helicity=False)
            + self.__generate_ls_arrow(transition, node_id)
            + " ".join(
                _state_to_str(s, use_helicity=False) for s in outgoing_states
            )
        )

    @staticmethod
    def __generate_ls_arrow(transition: StateTransition, node_id: int) -> str:
        interaction = transition.interactions[node_id]
        angular_momentum = sp.Rational(interaction.l_magnitude)
        coupled_spin = sp.Rational(interaction.s_magnitude)
        return fR" \xrightarrow[S={coupled_spin}]{{L={angular_momentum}}} "


def generate_transition_label(transition: StateTransition) -> str:
    initial_state_ids = transition.topology.incoming_edge_ids
    final_state_ids = transition.topology.outgoing_edge_ids
    initial_states = get_sorted_states(transition, initial_state_ids)
    final_states = get_sorted_states(transition, final_state_ids)
    return (
        _state_to_str(initial_states[0])
        + R" \to "
        + " ".join(_state_to_str(s) for s in final_states)
    )


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
        output_string += f"_{{{helicity_str}}}"
    return output_string


def _render_float(value: float) -> str:
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
