"""Generate an amplitude model with the helicity formalism."""

import logging
import operator
from collections import defaultdict
from functools import reduce
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import attr
import sympy as sp
from attr.validators import instance_of
from qrules.combinatorics import (
    perform_external_edge_identical_particle_combinatorics,
)
from qrules.particle import ParticleCollection, Spin
from qrules.quantum_numbers import InteractionProperties
from qrules.transition import ReactionInfo, State, StateTransition
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation as Wigner
from sympy.printing.latex import LatexPrinter

from .dynamics.builder import (
    ResonanceDynamicsBuilder,
    TwoBodyKinematicVariableSet,
    verify_signature,
)
from .kinematics import (
    HelicityAdapter,
    get_helicity_angle_label,
    get_invariant_mass_label,
)

ParameterValue = Union[float, complex, int]


@attr.s(auto_attribs=True, frozen=True)
class StateWithID(State):
    id: int  # noqa: A003

    @classmethod
    def from_transition(
        cls, transition: StateTransition, state_id: int
    ) -> "StateWithID":
        state = transition.states[state_id]
        return cls(
            id=state_id,
            particle=state.particle,
            spin_projection=state.spin_projection,
        )


@attr.s(auto_attribs=True, frozen=True)
class TwoBodyDecay:
    parent: StateWithID
    children: Tuple[StateWithID, StateWithID]
    interaction: InteractionProperties

    @classmethod
    def from_transition(
        cls, transition: StateTransition, node_id: int
    ) -> "TwoBodyDecay":
        topology = transition.topology
        in_state_ids = topology.get_edge_ids_ingoing_to_node(node_id)
        out_state_ids = topology.get_edge_ids_outgoing_from_node(node_id)
        if len(in_state_ids) != 1 or len(out_state_ids) != 2:
            raise ValueError(
                f"Node {node_id} does not represent a 1-to-2 body decay!"
            )

        sorted_by_id = sorted(out_state_ids)
        final_state_ids = [
            i for i in sorted_by_id if i in topology.outgoing_edge_ids
        ]
        intermediate_state_ids = [
            i for i in sorted_by_id if i in topology.intermediate_edge_ids
        ]
        sorted_by_ending = tuple(intermediate_state_ids + final_state_ids)

        ingoing_state_id = next(iter(in_state_ids))
        out_state_id1, out_state_id2, *_ = sorted_by_ending
        return cls(
            parent=StateWithID.from_transition(transition, ingoing_state_id),
            children=(
                StateWithID.from_transition(transition, out_state_id1),
                StateWithID.from_transition(transition, out_state_id2),
            ),
            interaction=transition.interactions[node_id],
        )

    def extract_angular_momentum(self) -> int:
        angular_momentum = self.interaction.l_magnitude
        if angular_momentum is not None:
            return angular_momentum
        spin_magnitude = self.parent.particle.spin
        if spin_magnitude.is_integer():
            return int(spin_magnitude)
        raise ValueError(
            f"Spin magnitude ({spin_magnitude}) of single particle state cannot be"
            f" used as the angular momentum as it is not integral!"
        )


@attr.s(frozen=True)
class HelicityModel:
    _expression: sp.Expr = attr.ib(
        validator=attr.validators.instance_of(sp.Expr)
    )
    _parameter_defaults: Dict[sp.Symbol, ParameterValue] = attr.ib(
        validator=attr.validators.instance_of(dict)
    )
    _components: Dict[str, sp.Expr] = attr.ib(
        validator=attr.validators.instance_of(dict)
    )
    _adapter: HelicityAdapter = attr.ib(
        validator=attr.validators.instance_of(HelicityAdapter)
    )
    particles: ParticleCollection = attr.ib(
        validator=instance_of(ParticleCollection)
    )

    @property
    def expression(self) -> sp.Expr:
        return self._expression

    @property
    def components(self) -> Dict[str, sp.Expr]:
        return self._components

    @property
    def parameter_defaults(self) -> Dict[sp.Symbol, ParameterValue]:
        return self._parameter_defaults

    @property
    def adapter(self) -> HelicityAdapter:
        return self._adapter


class _HelicityAmplitudeNameGenerator:
    def __init__(self) -> None:
        self.parity_partner_coefficient_mapping: Dict[str, str] = {}

    def _generate_amplitude_coefficient_couple(
        self, transition: StateTransition, node_id: int
    ) -> Tuple[str, str, str]:
        incoming_state, outgoing_states = self._retrieve_helicity_info(
            transition, node_id
        )
        par_name_suffix = self.generate_amplitude_coefficient_name(
            transition, node_id
        )

        pp_par_name_suffix = (
            _generate_particles_string([incoming_state], use_helicity=False)
            + R" \to "
            + _generate_particles_string(
                outgoing_states, make_parity_partner=True
            )
        )

        priority_name_suffix = par_name_suffix
        if outgoing_states[0].spin_projection < 0 or (
            outgoing_states[0].spin_projection == 0
            and outgoing_states[1].spin_projection < 0
        ):
            priority_name_suffix = pp_par_name_suffix

        return (par_name_suffix, pp_par_name_suffix, priority_name_suffix)

    def register_amplitude_coefficient_name(
        self, transition: StateTransition
    ) -> None:
        for node_id in transition.topology.nodes:
            (
                coefficient_suffix,
                parity_partner_coefficient_suffix,
                priority_partner_coefficient_suffix,
            ) = self._generate_amplitude_coefficient_couple(
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

    def generate_unique_amplitude_name(
        self,
        transition: StateTransition,
        node_id: Optional[int] = None,
    ) -> str:
        """Generates a unique name for the amplitude corresponding.

        That is, corresponging to the given :class:`StateTransition`. If
        ``node_id`` is given, it generates a unique name for the partial
        amplitude corresponding to the interaction node of the given
        :class:`StateTransition`.
        """
        name = ""
        if node_id is None:
            node_ids = transition.topology.nodes
        else:
            node_ids = frozenset({node_id})
        names: List[str] = []
        for i in node_ids:
            incoming_state, outgoing_states = self._retrieve_helicity_info(
                transition, i
            )
            name = (
                _generate_particles_string([incoming_state])
                + R" \to "
                + _generate_particles_string(outgoing_states)
            )
            names.append(name)
        return "; ".join(names)

    @staticmethod
    def _retrieve_helicity_info(
        transition: StateTransition, node_id: int
    ) -> Tuple[State, Tuple[State, State]]:
        in_edge_ids = transition.topology.get_edge_ids_ingoing_to_node(node_id)
        out_edge_ids = transition.topology.get_edge_ids_outgoing_from_node(
            node_id
        )
        in_helicity_list = _get_helicity_particles(transition, in_edge_ids)
        out_helicity_list = _get_helicity_particles(transition, out_edge_ids)
        if len(in_helicity_list) != 1 or len(out_helicity_list) != 2:
            raise ValueError(f"Node {node_id} it not a 1-to-2 decay")
        return (
            in_helicity_list[0],
            (out_helicity_list[0], out_helicity_list[1]),
        )

    def generate_amplitude_coefficient_name(
        self, transition: StateTransition, node_id: int
    ) -> str:
        """Generate partial amplitude coefficient name suffix."""
        in_hel_info, out_hel_info = self._retrieve_helicity_info(
            transition, node_id
        )
        return (
            _generate_particles_string([in_hel_info], use_helicity=False)
            + R" \to "
            + _generate_particles_string(out_hel_info)
        )

    def generate_sequential_amplitude_suffix(
        self, transition: StateTransition
    ) -> str:
        """Generate unique suffix for a sequential amplitude transition."""
        coefficient_names: List[str] = []
        for node_id in transition.topology.nodes:
            suffix = self.generate_amplitude_coefficient_name(
                transition, node_id
            )
            if suffix in self.parity_partner_coefficient_mapping:
                suffix = self.parity_partner_coefficient_mapping[suffix]
            coefficient_names.append(suffix)
        return "; ".join(coefficient_names)


class _CanonicalAmplitudeNameGenerator(_HelicityAmplitudeNameGenerator):
    def generate_amplitude_coefficient_name(
        self, transition: StateTransition, node_id: int
    ) -> str:
        incoming_state, outgoing_states = self._retrieve_helicity_info(
            transition, node_id
        )
        return (
            _generate_particles_string([incoming_state], use_helicity=False)
            + self.__generate_ls_arrow(transition, node_id)
            + _generate_particles_string(outgoing_states, use_helicity=False)
        )

    def generate_unique_amplitude_name(
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
            helicity_name = super().generate_unique_amplitude_name(
                transition, node
            )
            canonical_name = helicity_name.replace(
                R" \to ",
                self.__generate_ls_arrow(transition, node),
            )
            names.append(canonical_name)
        return "; ".join(names)

    def __generate_ls_arrow(
        self, transition: StateTransition, node_id: int
    ) -> str:
        angular_momentum, spin = self.__get_ls_coupling(transition, node_id)
        return fR" \xrightarrow[S={spin}]{{L={angular_momentum}}} "

    @staticmethod
    def __get_ls_coupling(
        transition: StateTransition, node_id: int
    ) -> Tuple[sp.Rational, sp.Rational]:
        interaction = transition.interactions[node_id]
        ang_orb_mom = sp.Rational(get_angular_momentum(interaction).magnitude)
        spin = sp.Rational(get_coupled_spin(interaction).magnitude)
        return ang_orb_mom, spin


def _get_transition_group_label(
    transition_group: List[StateTransition],
) -> str:
    label = ""
    if transition_group:
        first_transition = next(iter(transition_group))
        ise = first_transition.topology.incoming_edge_ids
        fse = first_transition.topology.outgoing_edge_ids
        is_names = _get_helicity_particles(first_transition, ise)
        fs_names = _get_helicity_particles(first_transition, fse)
        label += (
            _generate_particles_string(is_names)
            + R" \to "
            + _generate_particles_string(fs_names)
        )
    return label


def _get_helicity_particles(
    transition: StateTransition, state_ids: Iterable[int]
) -> List[State]:
    """Get a sorted list of `~qrules.transition.State` instances.

    In order to ensure correct naming of amplitude coefficients the list has to
    be sorted by name. The same coefficient names have to be created for two
    transitions that only differ from a kinematic standpoint (swapped external
    edges).
    """
    helicity_list = [transition.states[i] for i in state_ids]
    return sorted(helicity_list, key=lambda s: s.particle.name)


def _generate_particles_string(
    helicity_list: Iterable[State],
    use_helicity: bool = True,
    make_parity_partner: bool = False,
) -> str:
    output_string = ""
    for state in helicity_list:
        if state.particle.latex is not None:
            output_string += state.particle.latex
        else:
            output_string += state.particle.name
        if use_helicity:
            if make_parity_partner:
                helicity = -1 * state.spin_projection
            else:
                helicity = state.spin_projection
            helicity = sp.Rational(helicity)
            if helicity > 0:
                helicity_str = f"+{helicity}"
            else:
                helicity_str = str(helicity)
            output_string += f"_{{{helicity_str}}}"
        output_string += " "
    return output_string[:-1]


def _generate_kinematic_variable_set(
    transition: StateTransition, node_id: int
) -> TwoBodyKinematicVariableSet:
    decay = TwoBodyDecay.from_transition(transition, node_id)
    inv_mass, phi, theta = generate_kinematic_variables(transition, node_id)
    child1_mass = sp.Symbol(
        get_invariant_mass_label(transition.topology, decay.children[0].id),
        real=True,
    )
    child2_mass = sp.Symbol(
        get_invariant_mass_label(transition.topology, decay.children[1].id),
        real=True,
    )
    return TwoBodyKinematicVariableSet(
        incoming_state_mass=inv_mass,
        outgoing_state_mass1=child1_mass,
        outgoing_state_mass2=child2_mass,
        helicity_theta=theta,
        helicity_phi=phi,
        angular_momentum=decay.extract_angular_momentum(),
    )


def generate_kinematic_variables(
    transition: StateTransition, node_id: int
) -> Tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    """Generate symbol for invariant mass, phi angle, and theta angle."""
    decay = TwoBodyDecay.from_transition(transition, node_id)
    phi_label, theta_label = get_helicity_angle_label(
        transition.topology, decay.children[0].id
    )
    inv_mass_label = get_invariant_mass_label(
        transition.topology, decay.parent.id
    )
    return (
        sp.Symbol(inv_mass_label, real=True),
        sp.Symbol(phi_label, real=True),
        sp.Symbol(theta_label, real=True),
    )


class HelicityAmplitudeBuilder:  # pylint: disable=too-many-instance-attributes
    """Amplitude model generator for the helicity formalism."""

    def __init__(self, reaction: ReactionInfo) -> None:
        self.name_generator = _HelicityAmplitudeNameGenerator()
        self.__reaction = reaction
        self.__parameter_defaults: Dict[sp.Symbol, ParameterValue] = {}
        self.__components: Dict[str, sp.Expr] = {}
        self.__dynamics_choices: Dict[
            TwoBodyDecay, ResonanceDynamicsBuilder
        ] = {}

        if len(reaction.transitions) < 1:
            raise ValueError(
                f"At least one {StateTransition.__name__} required to"
                " genenerate an amplitude model!"
            )
        self.__adapter = HelicityAdapter(reaction)
        for grouping in reaction.transition_groups:
            self.__adapter.register_topology(grouping.topology)
        self.__particles = extract_particle_collection(reaction.transitions)

    def set_dynamics(
        self, particle_name: str, dynamics_builder: ResonanceDynamicsBuilder
    ) -> None:
        verify_signature(dynamics_builder, ResonanceDynamicsBuilder)
        for transition in self.__reaction.transitions:
            for node_id in transition.topology.nodes:
                decay = TwoBodyDecay.from_transition(transition, node_id)
                decay_particle = decay.parent.particle
                if decay_particle.name == particle_name:
                    self.__dynamics_choices[decay] = dynamics_builder

    def generate(self) -> HelicityModel:
        self.__components = {}
        self.__parameter_defaults = {}
        return HelicityModel(
            expression=self.__generate_intensity(),
            components=self.__components,
            parameter_defaults=self.__parameter_defaults,
            adapter=self.__adapter,
            particles=self.__particles,
        )

    def __generate_intensity(self) -> sp.Expr:
        transition_groups = group_transitions(self.__reaction.transitions)
        logging.debug("There are %d transition groups", len(transition_groups))

        self.__create_parameter_couplings(transition_groups)
        coherent_intensities = []
        for group in transition_groups:
            coherent_intensities.append(
                self.__generate_coherent_intensity(group)
            )
        if len(coherent_intensities) == 0:
            raise ValueError("List of coherent intensities cannot be empty")
        return sum(coherent_intensities)

    def __create_dynamics(
        self, transition: StateTransition, node_id: int
    ) -> sp.Expr:
        decay = TwoBodyDecay.from_transition(transition, node_id)
        if decay in self.__dynamics_choices:
            builder = self.__dynamics_choices[decay]
            variable_set = _generate_kinematic_variable_set(
                transition, node_id
            )
            expression, parameters = builder(
                decay.parent.particle, variable_set
            )
            for par, value in parameters.items():
                if par in self.__parameter_defaults:
                    previous_value = self.__parameter_defaults[par]
                    if value != previous_value:
                        logging.warning(
                            f"Default value for parameter {par.name}"
                            f" inconsistent {value} and {previous_value}"
                        )
                self.__parameter_defaults[par] = value

            return expression

        return 1

    def __create_parameter_couplings(
        self, transition_groups: List[List[StateTransition]]
    ) -> None:
        for graph_group in transition_groups:
            for transition in graph_group:
                self.name_generator.register_amplitude_coefficient_name(
                    transition
                )

    def __generate_coherent_intensity(
        self,
        transition_group: List[StateTransition],
    ) -> sp.Expr:
        graph_group_label = _get_transition_group_label(transition_group)
        sequential_expressions: List[sp.Expr] = []
        for transition in transition_group:
            sequential_graphs = (
                perform_external_edge_identical_particle_combinatorics(
                    transition.to_graph()
                )
            )
            for graph in sequential_graphs:
                transition = StateTransition.from_graph(graph)
                expression = self.__generate_sequential_decay(transition)
                sequential_expressions.append(expression)
        amplitude_sum = sum(sequential_expressions)
        coh_intensity = abs(amplitude_sum) ** 2
        self.__components[fR"I_{{{graph_group_label}}}"] = coh_intensity
        return coh_intensity

    def __generate_sequential_decay(
        self, transition: StateTransition
    ) -> sp.Expr:
        partial_decays: List[sp.Symbol] = [
            self._generate_partial_decay(transition, node_id)
            for node_id in transition.topology.nodes
        ]
        sequential_amplitudes = reduce(operator.mul, partial_decays)

        coefficient = self.__generate_amplitude_coefficient(transition)
        prefactor = self.__generate_amplitude_prefactor(transition)
        expression = coefficient * sequential_amplitudes
        if prefactor is not None:
            expression = prefactor * expression
        self.__components[
            f"A_{{{self.name_generator.generate_unique_amplitude_name(transition)}}}"
        ] = expression
        return expression

    def _generate_partial_decay(  # pylint: disable=too-many-locals
        self, transition: StateTransition, node_id: int
    ) -> sp.Expr:
        wigner_d = generate_wigner_d(transition, node_id)
        dynamics_symbol = self.__create_dynamics(transition, node_id)
        return wigner_d * dynamics_symbol

    def __generate_amplitude_coefficient(
        self, transition: StateTransition
    ) -> sp.Symbol:
        """Generate coefficient parameter for a sequential amplitude transition.

        Generally, each partial amplitude of a sequential amplitude transition
        should check itself if it or a parity partner is already defined. If so
        a coupled coefficient is introduced.
        """
        suffix = self.name_generator.generate_sequential_amplitude_suffix(
            transition
        )
        coefficient_symbol = sp.Symbol(f"C_{{{suffix}}}")
        self.__parameter_defaults[coefficient_symbol] = complex(1, 0)
        return coefficient_symbol

    def __generate_amplitude_prefactor(
        self, transition: StateTransition
    ) -> Optional[float]:
        prefactor = get_prefactor(transition)
        if prefactor != 1.0:
            for node_id in transition.topology.nodes:
                raw_suffix = (
                    self.name_generator.generate_amplitude_coefficient_name(
                        transition, node_id
                    )
                )
                if (
                    raw_suffix
                    in self.name_generator.parity_partner_coefficient_mapping
                ):
                    coefficient_suffix = (
                        self.name_generator.parity_partner_coefficient_mapping[
                            raw_suffix
                        ]
                    )
                    if coefficient_suffix != raw_suffix:
                        return prefactor
        return None


def generate_wigner_d(transition: StateTransition, node_id: int) -> sp.Symbol:
    decay = TwoBodyDecay.from_transition(transition, node_id)
    _, phi, theta = generate_kinematic_variables(transition, node_id)
    return Wigner.D(
        j=sp.Rational(decay.parent.particle.spin),
        m=sp.Rational(decay.parent.spin_projection),
        mp=sp.Rational(
            decay.children[0].spin_projection
            - decay.children[1].spin_projection
        ),
        alpha=-phi,
        beta=theta,
        gamma=0,
    )


class CanonicalAmplitudeBuilder(HelicityAmplitudeBuilder):
    r"""Amplitude model generator for the canonical helicity formalism.

    This class defines a full amplitude in the canonical formalism, using the
    helicity formalism as a foundation. The key here is that we take the full
    helicity intensity as a template, and just exchange the helicity amplitudes
    :math:`F` as a sum of canonical amplitudes :math:`A`:

    .. math::

        F^J_{\lambda_1,\lambda_2} = \sum_{LS} \mathrm{norm}(A^J_{LS})C^2.

    Here, :math:`C` stands for `Clebsch-Gordan factor
    <https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients>`_.
    """

    def __init__(self, reaction_result: ReactionInfo) -> None:
        super().__init__(reaction_result)
        self.name_generator = _CanonicalAmplitudeNameGenerator()

    def _generate_partial_decay(  # pylint: disable=too-many-locals
        self, transition: StateTransition, node_id: int
    ) -> sp.Symbol:
        amplitude = super()._generate_partial_decay(transition, node_id)
        cg_coefficients = generate_clebsch_gordan(transition, node_id)
        return cg_coefficients * amplitude


def generate_clebsch_gordan(
    transition: StateTransition, node_id: int
) -> sp.Expr:
    decay = TwoBodyDecay.from_transition(transition, node_id)

    angular_momentum = get_angular_momentum(decay.interaction)
    coupled_spin = get_coupled_spin(decay.interaction)
    if angular_momentum.projection != 0.0:
        raise ValueError(f"Projection of L is non-zero!: {angular_momentum}")

    parent = decay.parent
    child1 = decay.children[0]
    child2 = decay.children[1]

    decay_particle_lambda = child1.spin_projection - child2.spin_projection
    cg_ls = CG(
        j1=sp.Rational(angular_momentum.magnitude),
        m1=sp.Rational(angular_momentum.projection),
        j2=sp.Rational(coupled_spin.magnitude),
        m2=sp.Rational(decay_particle_lambda),
        j3=sp.Rational(parent.particle.spin),
        m3=sp.Rational(decay_particle_lambda),
    )
    cg_ss = CG(
        j1=sp.Rational(child1.particle.spin),
        m1=sp.Rational(child1.spin_projection),
        j2=sp.Rational(child2.particle.spin),
        m2=sp.Rational(-child2.spin_projection),
        j3=sp.Rational(coupled_spin.magnitude),
        m3=sp.Rational(decay_particle_lambda),
    )
    return sp.Mul(cg_ls, cg_ss, evaluate=False)


# https://github.com/sympy/sympy/issues/21001
# pylint: disable=protected-access, unused-argument
def _latex_fix(self: Type[CG], printer: LatexPrinter, *args: Any) -> str:
    j3, m3, j1, m1, j2, m2 = map(
        printer._print,
        (self.j3, self.m3, self.j1, self.m1, self.j2, self.m2),
    )
    return f"{{C^{{{j3},{m3}}}_{{{j1},{m1},{j2},{m2}}}}}"


CG._latex = _latex_fix


def extract_particle_collection(
    transitions: Iterable[StateTransition],
) -> ParticleCollection:
    particles = ParticleCollection()
    for transition in transitions:
        for state in transition.states.values():
            if state.particle not in particles:
                particles.add(state.particle)
    return particles


def get_angular_momentum(interaction: InteractionProperties) -> Spin:
    l_magnitude = interaction.l_magnitude
    l_projection = interaction.l_projection
    if l_magnitude is None or l_projection is None:
        raise TypeError(
            "Angular momentum L not defined!", l_magnitude, l_projection
        )
    return Spin(l_magnitude, l_projection)


def get_coupled_spin(interaction: InteractionProperties) -> Spin:
    s_magnitude = interaction.s_magnitude
    s_projection = interaction.s_projection
    if s_magnitude is None or s_projection is None:
        raise TypeError("Coupled spin S not defined!")
    return Spin(s_magnitude, s_projection)


def get_prefactor(
    transition: StateTransition,
) -> float:
    """Calculate the product of all prefactors defined in this transition."""
    prefactor = 1.0
    for node_id in transition.topology.nodes:
        interaction = transition.interactions[node_id]
        if interaction and interaction.parity_prefactor is not None:
            prefactor *= interaction.parity_prefactor
    return prefactor


def group_transitions(
    transitions: Iterable[StateTransition],
) -> List[List[StateTransition]]:
    """Match final and initial states in groups.

    Each `~qrules.transition.StateTransition` corresponds to a specific state
    transition amplitude. This function groups together transitions, which have
    the same initial and final state (including spin). This is needed to
    determine the coherency of the individual amplitude parts.
    """
    transition_groups: DefaultDict[
        Tuple[
            Tuple[Tuple[str, float], ...],
            Tuple[Tuple[str, float], ...],
        ],
        List[StateTransition],
    ] = defaultdict(list)
    for transition in transitions:
        initial_state = sorted(
            (
                transition.states[i].particle.name,
                transition.states[i].spin_projection,
            )
            for i in transition.topology.incoming_edge_ids
        )
        final_state = sorted(
            (
                transition.states[i].particle.name,
                transition.states[i].spin_projection,
            )
            for i in transition.topology.outgoing_edge_ids
        )
        group_key = (tuple(initial_state), tuple(final_state))
        transition_groups[group_key].append(transition)

    return list(transition_groups.values())
