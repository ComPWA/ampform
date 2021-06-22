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
from qrules.particle import ParticleCollection
from qrules.transition import ReactionInfo, StateTransition
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import Rotation as Wigner
from sympy.printing.latex import LatexPrinter

from ampform.dynamics.builder import (
    ResonanceDynamicsBuilder,
    TwoBodyKinematicVariableSet,
    verify_signature,
)
from ampform.kinematics import (
    HelicityAdapter,
    get_helicity_angle_label,
    get_invariant_mass_label,
)

from .decay import TwoBodyDecay, get_angular_momentum, get_coupled_spin
from .naming import (
    CanonicalAmplitudeNameGenerator,
    HelicityAmplitudeNameGenerator,
    generate_transition_label,
)

ParameterValue = Union[float, complex, int]


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


class HelicityAmplitudeBuilder:
    """Amplitude model generator for the helicity formalism."""

    def __init__(self, reaction: ReactionInfo) -> None:
        self._name_generator = HelicityAmplitudeNameGenerator()
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
                decaying_particle = decay.parent.particle
                if decaying_particle.name == particle_name:
                    self.__dynamics_choices[decay] = dynamics_builder

    def formulate(self) -> HelicityModel:
        self.__components = {}
        self.__parameter_defaults = {}
        return HelicityModel(
            expression=self.__formulate_top_expression(),
            components=self.__components,
            parameter_defaults=self.__parameter_defaults,
            adapter=self.__adapter,
            particles=self.__particles,
        )

    def __formulate_top_expression(self) -> sp.Expr:
        transition_groups = group_transitions(self.__reaction.transitions)
        self.__register_parameter_couplings(transition_groups)
        coherent_intensities = [
            self.__formulate_coherent_intensity(group)
            for group in transition_groups
        ]
        return sum(coherent_intensities)

    def __register_parameter_couplings(
        self, transition_groups: List[List[StateTransition]]
    ) -> None:
        for graph_group in transition_groups:
            for transition in graph_group:
                self._name_generator.register_amplitude_coefficient_name(
                    transition
                )

    def __formulate_coherent_intensity(
        self, transition_group: List[StateTransition]
    ) -> sp.Expr:
        graph_group_label = generate_transition_label(transition_group[0])
        sequential_expressions: List[sp.Expr] = []
        for transition in transition_group:
            sequential_graphs = (
                perform_external_edge_identical_particle_combinatorics(
                    transition.to_graph()
                )
            )
            for graph in sequential_graphs:
                transition = StateTransition.from_graph(graph)
                expression = self.__formulate_sequential_decay(transition)
                sequential_expressions.append(expression)
        amplitude_sum = sum(sequential_expressions)
        coherent_intensity = abs(amplitude_sum) ** 2
        self.__components[fR"I_{{{graph_group_label}}}"] = coherent_intensity
        return coherent_intensity

    def __formulate_sequential_decay(
        self, transition: StateTransition
    ) -> sp.Expr:
        partial_decays: List[sp.Symbol] = [
            self._formulate_partial_decay(transition, node_id)
            for node_id in transition.topology.nodes
        ]
        sequential_amplitudes = reduce(operator.mul, partial_decays)

        coefficient = self.__generate_amplitude_coefficient(transition)
        prefactor = self.__generate_amplitude_prefactor(transition)
        expression = coefficient * sequential_amplitudes
        if prefactor is not None:
            expression = prefactor * expression
        self.__components[
            f"A_{{{self._name_generator.generate_amplitude_name(transition)}}}"
        ] = expression
        return expression

    def _formulate_partial_decay(
        self, transition: StateTransition, node_id: int
    ) -> sp.Expr:
        wigner_d = formulate_wigner_d(transition, node_id)
        dynamics = self.__formulate_dynamics(transition, node_id)
        return wigner_d * dynamics

    def __formulate_dynamics(
        self, transition: StateTransition, node_id: int
    ) -> sp.Expr:
        decay = TwoBodyDecay.from_transition(transition, node_id)
        if decay not in self.__dynamics_choices:
            return 1

        builder = self.__dynamics_choices[decay]
        variable_set = _generate_kinematic_variable_set(transition, node_id)
        expression, parameters = builder(decay.parent.particle, variable_set)
        for par, value in parameters.items():
            if par in self.__parameter_defaults:
                previous_value = self.__parameter_defaults[par]
                if value != previous_value:
                    logging.warning(
                        f'New default value {value} for parameter "{par.name}"'
                        f" is inconsistent with existing value {previous_value}"
                    )
            self.__parameter_defaults[par] = value

        return expression

    def __generate_amplitude_coefficient(
        self, transition: StateTransition
    ) -> sp.Symbol:
        """Generate coefficient parameter for a sequential amplitude transition.

        Generally, each partial amplitude of a sequential amplitude transition
        should check itself if it or a parity partner is already defined. If so
        a coupled coefficient is introduced.
        """
        suffix = self._name_generator.generate_sequential_amplitude_suffix(
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
                raw_suffix = self._name_generator.generate_coefficient_name(
                    transition, node_id
                )
                if (
                    raw_suffix
                    in self._name_generator.parity_partner_coefficient_mapping
                ):
                    coefficient_suffix = self._name_generator.parity_partner_coefficient_mapping[
                        raw_suffix
                    ]
                    if coefficient_suffix != raw_suffix:
                        return prefactor
        return None


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
        self._name_generator = CanonicalAmplitudeNameGenerator()

    def _formulate_partial_decay(
        self, transition: StateTransition, node_id: int
    ) -> sp.Expr:
        amplitude = super()._formulate_partial_decay(transition, node_id)
        cg_coefficients = formulate_clebsch_gordan_coefficients(
            transition, node_id
        )
        return cg_coefficients * amplitude


def extract_particle_collection(
    transitions: Iterable[StateTransition],
) -> ParticleCollection:
    """Collect all particles from a collection of state transitions."""
    particles = ParticleCollection()
    for transition in transitions:
        for state in transition.states.values():
            if state.particle not in particles:
                particles.add(state.particle)
    return particles


def formulate_clebsch_gordan_coefficients(
    transition: StateTransition, node_id: int
) -> sp.Expr:
    """Compute two Clebsch-Gordan coefficients for a state transition node.

    >>> import qrules
    >>> reaction = qrules.generate_transitions(
    ...     initial_state=[("J/psi(1S)", [+1])],
    ...     final_state=[("gamma", [+1]), "f(0)(980)"],
    ... )
    >>> transition = reaction.transitions[0]
    >>> formulate_clebsch_gordan_coefficients(transition, node_id=0)
    CG(0, 0, 1, 1, 1, 1)*CG(1, 1, 0, 0, 1, 1)

    .. seealso:: :doc:`sympy:modules/physics/quantum/cg`
    """
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


def formulate_wigner_d(transition: StateTransition, node_id: int) -> sp.Expr:
    """Compute `~sympy.physics.quantum.spin.WignerD` for a state transition node.

    >>> import qrules
    >>> reaction = qrules.generate_transitions(
    ...     initial_state=[("J/psi(1S)", [+1])],
    ...     final_state=[("gamma", [+1]), "f(0)(980)"],
    ... )
    >>> transition = reaction.transitions[0]
    >>> formulate_wigner_d(transition, node_id=0)
    WignerD(1, 1, 1, -phi_0, theta_0, 0)
    """
    decay = TwoBodyDecay.from_transition(transition, node_id)
    _, phi, theta = _generate_kinematic_variables(transition, node_id)
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


def get_prefactor(transition: StateTransition) -> float:
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


def _generate_kinematic_variable_set(
    transition: StateTransition, node_id: int
) -> TwoBodyKinematicVariableSet:
    decay = TwoBodyDecay.from_transition(transition, node_id)
    inv_mass, phi, theta = _generate_kinematic_variables(transition, node_id)
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


def _generate_kinematic_variables(
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


# https://github.com/sympy/sympy/issues/21001
# pylint: disable=protected-access, unused-argument
def _latex_fix(self: Type[CG], printer: LatexPrinter, *args: Any) -> str:
    j3, m3, j1, m1, j2, m2 = map(
        printer._print,
        (self.j3, self.m3, self.j1, self.m1, self.j2, self.m2),
    )
    return f"{{C^{{{j3},{m3}}}_{{{j1},{m1},{j2},{m2}}}}}"


CG._latex = _latex_fix
