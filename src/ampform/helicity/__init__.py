# pylint: disable=import-outside-toplevel
"""Generate an amplitude model with the helicity formalism."""

import collections
import logging
import operator
import re
from collections import OrderedDict
from difflib import get_close_matches
from functools import reduce
from typing import (
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import attr
import sympy as sp
from attr.validators import instance_of
from qrules.combinatorics import (
    perform_external_edge_identical_particle_combinatorics,
)
from qrules.transition import ReactionInfo, StateTransition

from ampform.dynamics.builder import (
    ResonanceDynamicsBuilder,
    TwoBodyKinematicVariableSet,
)
from ampform.kinematics import (
    HelicityAdapter,
    get_helicity_angle_label,
    get_invariant_mass_label,
)

from .decay import TwoBodyDecay
from .naming import (
    CanonicalAmplitudeNameGenerator,
    HelicityAmplitudeNameGenerator,
    generate_transition_label,
)

ParameterValue = Union[float, complex, int]
"""Allowed value types for parameters."""


def _order_component_mapping(
    mapping: Mapping[str, ParameterValue]
) -> "OrderedDict[str, ParameterValue]":
    return collections.OrderedDict(
        [(key, mapping[key]) for key in sorted(mapping, key=_natural_sorting)]
    )


def _order_symbol_mapping(
    mapping: Mapping[sp.Symbol, sp.Expr]
) -> "OrderedDict[sp.Symbol, sp.Expr]":
    return collections.OrderedDict(
        [
            (symbol, mapping[symbol])
            for symbol in sorted(
                mapping, key=lambda s: _natural_sorting(s.name)
            )
        ]
    )


def _natural_sorting(text: str) -> List[Union[float, str]]:
    # https://stackoverflow.com/a/5967539/13219025
    return [
        __attempt_number_cast(c)
        for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)
    ]


def __attempt_number_cast(text: str) -> Union[float, str]:
    try:
        return float(text)
    except ValueError:
        return text


@attr.frozen
class HelicityModel:  # noqa: R701
    expression: sp.Expr = attr.ib(
        validator=attr.validators.instance_of(sp.Expr)
    )
    parameter_defaults: "OrderedDict[sp.Symbol, ParameterValue]" = attr.ib(
        converter=_order_symbol_mapping
    )
    """A mapping of suggested parameter values.

    Keys are `~sympy.core.symbol.Symbol` instances from the main
    :attr:`expression` that should be interpreted as parameters (as opposed to
    variables). The symbols are ordered alphabetically by name with `natural
    sort order <https://en.wikipedia.org/wiki/Natural_sort_order>`_. Values
    have been extracted from the input `~qrules.transition.ReactionInfo`.
    """
    components: "OrderedDict[str, sp.Expr]" = attr.ib(
        converter=_order_component_mapping
    )
    """A mapping for identifying main components in the :attr:`expression`.

    Keys are the component names (`str`), formatted as LaTeX, and values are
    sub-expressions in the main :attr:`expression`. The mapping is an
    `~collections.OrderedDict` that orders the component names alphabetically
    with `natural sort order
    <https://en.wikipedia.org/wiki/Natural_sort_order>`_.
    """
    kinematic_variables: "OrderedDict[sp.Symbol, sp.Expr]" = attr.ib(
        converter=_order_symbol_mapping
    )
    """Expressions for converting four-momenta to kinematic variables."""
    reaction_info: ReactionInfo = attr.ib(validator=instance_of(ReactionInfo))

    def sum_components(  # noqa: R701
        self, components: Iterable[str]
    ) -> sp.Expr:
        """Coherently or incoherently add components of a helicity model."""
        components = list(components)  # copy
        for component in components:
            if component not in self.components:
                first_letter = component[0]
                candidates = get_close_matches(
                    component,
                    filter(
                        lambda c: c.startswith(
                            first_letter  # pylint: disable=cell-var-from-loop
                        ),
                        self.components,
                    ),
                )
                raise KeyError(
                    f'Component "{component}" not in model components. '
                    "Did you mean any of these?",
                    candidates,
                )
        if any(map(lambda c: c.startswith("I"), components)) and any(
            map(lambda c: c.startswith("A"), components)
        ):
            intensity_sum = self.sum_components(
                components=filter(lambda c: c.startswith("I"), components),
            )
            amplitude_sum = self.sum_components(
                components=filter(lambda c: c.startswith("A"), components),
            )
            return intensity_sum + amplitude_sum
        if all(map(lambda c: c.startswith("I"), components)):
            return sum(self.components[c] for c in components)
        if all(map(lambda c: c.startswith("A"), components)):
            return abs(sum(self.components[c] for c in components)) ** 2
        raise ValueError(
            'Not all component names started with either "A" or "I"'
        )


class HelicityAmplitudeBuilder:  # pylint: disable=too-many-instance-attributes
    r"""Amplitude model generator for the helicity formalism.

    Args:
        reaction: The `~qrules.transition.ReactionInfo` from which to
            :meth:`formulate` an amplitude model.
        stable_final_state_ids: Put final state 'invariant' masses
            (:math:`m_0, m_1, \dots`) under `.HelicityModel.parameter_defaults`
            (with a *scalar* suggested value) instead of
            `~.HelicityModel.kinematic_variables` (which are expressions to
            compute an event-wise array of invariant masses). This is useful
            if final state particles are stable.

            .. seealso:: :ref:`usage/amplitude:Stable final states`
    """

    def __init__(
        self,
        reaction: ReactionInfo,
        stable_final_state_ids: Optional[Iterable[int]] = None,
    ) -> None:
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
        self.stable_final_state_ids = stable_final_state_ids  # type: ignore[assignment]
        for grouping in reaction.transition_groups:
            self.__adapter.register_topology(grouping.topology)

    @property
    def adapter(self) -> HelicityAdapter:
        """Converter for computing kinematic variables from four-momenta."""
        return self.__adapter

    @property
    def stable_final_state_ids(self) -> Optional[Set[int]]:
        # noqa: D403
        """IDs of the final states that should be considered stable.

        The 'invariant' mass symbols for these final states will be inserted as
        **scalar** values into the `.parameter_defaults`.
        """
        return self.__stable_final_state_ids

    @stable_final_state_ids.setter
    def stable_final_state_ids(self, value: Optional[Iterable[int]]) -> None:
        self.__stable_final_state_ids = None
        if value is not None:
            self.__stable_final_state_ids = set(value)
            if not self.__stable_final_state_ids <= set(
                self.__reaction.final_state
            ):
                raise ValueError(
                    "Final state IDs are"
                    f" {sorted(self.__reaction.final_state)}, but trying to"
                    " set stable final state IDs"
                    f" {self.__stable_final_state_ids}"
                )

    def set_dynamics(
        self, particle_name: str, dynamics_builder: ResonanceDynamicsBuilder
    ) -> None:
        found_particle = False
        for transition in self.__reaction.transitions:
            for node_id in transition.topology.nodes:
                decay = TwoBodyDecay.from_transition(transition, node_id)
                decaying_particle = decay.parent.particle
                if decaying_particle.name == particle_name:
                    self.__dynamics_choices[decay] = dynamics_builder
                    found_particle = True
        if not found_particle:
            logging.warning(
                f'Model contains no resonance with name "{particle_name}"'
            )

    def formulate(self) -> HelicityModel:
        self.__components = {}
        self.__parameter_defaults = {}
        top_expression = self.__formulate_top_expression()
        kinematic_variables = {
            sp.Symbol(var_name, real=True): expr
            for var_name, expr in self.__adapter.create_expressions().items()
        }
        if self.__stable_final_state_ids is not None:
            for state_id in self.__stable_final_state_ids:
                mass_symbol = sp.Symbol(f"m_{state_id}", real=True)
                particle = self.__reaction.final_state[state_id]
                self.__parameter_defaults[mass_symbol] = particle.mass
                del kinematic_variables[mass_symbol]

        return HelicityModel(
            expression=top_expression,
            components=self.__components,
            parameter_defaults=self.__parameter_defaults,
            kinematic_variables=kinematic_variables,
            reaction_info=self.__reaction,
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
                        " is inconsistent with existing value"
                        f" {previous_value}"
                    )
            self.__parameter_defaults[par] = value

        return expression

    def __generate_amplitude_coefficient(
        self, transition: StateTransition
    ) -> sp.Symbol:
        """Generate coefficient parameter for a sequential amplitude.

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

    .. seealso:: `HelicityAmplitudeBuilder` and :doc:`/usage/formalism`.
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


def formulate_clebsch_gordan_coefficients(
    transition: StateTransition, node_id: int
) -> sp.Expr:
    r"""Compute the two Clebsch-Gordan coefficients for a state transition node.

    In the **canonical basis** (also called **partial wave basis**),
    :doc:`Clebsch-Gordan coefficients <sympy:modules/physics/quantum/cg>`
    ensure that the projection of angular momentum is conserved
    (:cite:`kutschkeAngularDistributionCookbook1996`, p. 4). When calling
    :func:`~qrules.generate_transitions` with
    :code:`formalism="canonical-helicity"`, AmpForm formulates the amplitude in
    the canonical basis from amplitudes in the helicity basis using the
    transformation in :cite:`chungSpinFormalismsUpdated2014`, Eq. (4.32). See
    also :cite:`kutschkeAngularDistributionCookbook1996`, Eq. (28).

    This function produces the two Clebsch-Gordan coefficients in
    :cite:`chungSpinFormalismsUpdated2014`, Eq. (4.32). For a two-body decay
    :math:`1 \to 2, 3`, we get:

    .. math:: C^{s_1,\lambda}_{L,0,S,\lambda} C^{S,\lambda}_{s_2,\lambda_2,s_3,-\lambda_3}
        :label: formulate_clebsch_gordan_coefficients

    with:

    - :math:`s_i` the intrinsic `Spin.magnitude
      <qrules.particle.Spin.magnitude>` of each state :math:`i`,
    - :math:`\lambda_{2}, \lambda_{3}` the helicities of the decay products
      (can be taken to be their `~qrules.transition.State.spin_projection` when
      following a constistent boosting procedure),
    - :math:`\lambda=\lambda_{2}-\lambda_{3}`,
    - :math:`L` the *total* angular momentum of the final state pair
      (`~qrules.quantum_numbers.InteractionProperties.l_magnitude`),
    - :math:`S` the coupled spin magnitude of the final state pair
      (`~qrules.quantum_numbers.InteractionProperties.s_magnitude`),
    - and :math:`C^{j_3,m_3}_{j_1,m_1,j_2,m_2} = \langle
      j1,m1;j2,m2|j3,m3\rangle`, as in :doc:`sympy:modules/physics/quantum/cg`.

    Example
    -------
    >>> import qrules
    >>> reaction = qrules.generate_transitions(
    ...     initial_state=[("J/psi(1S)", [+1])],
    ...     final_state=[("gamma", [-1]), "f(0)(980)"],
    ... )
    >>> transition = reaction.transitions[1]  # angular momentum 2
    >>> formulate_clebsch_gordan_coefficients(transition, node_id=0)
    CG(1, -1, 0, 0, 1, -1)*CG(2, 0, 1, -1, 1, -1)

    .. math::
        C^{s_1,\lambda}_{L,0,S,\lambda} C^{S,\lambda}_{s_2,\lambda_2,s_3,-\lambda_3}
        = C^{1,(-1-0)}_{2,0,1,(-1-0)} C^{1,(-1-0)}_{1,-1,0,0}
        = C^{1,-1}_{2,0,1,-1} C^{1,-1}_{1,-1,0,0}
    """
    from sympy.physics.quantum.cg import CG

    decay = TwoBodyDecay.from_transition(transition, node_id)

    angular_momentum = decay.interaction.l_magnitude
    coupled_spin = decay.interaction.s_magnitude

    parent = decay.parent
    child1 = decay.children[0]
    child2 = decay.children[1]

    decay_particle_lambda = child1.spin_projection - child2.spin_projection
    cg_ls = CG(
        j1=sp.Rational(angular_momentum),
        m1=0,
        j2=sp.Rational(coupled_spin),
        m2=sp.Rational(decay_particle_lambda),
        j3=sp.Rational(parent.particle.spin),
        m3=sp.Rational(decay_particle_lambda),
    )
    cg_ss = CG(
        j1=sp.Rational(child1.particle.spin),
        m1=sp.Rational(child1.spin_projection),
        j2=sp.Rational(child2.particle.spin),
        m2=sp.Rational(-child2.spin_projection),
        j3=sp.Rational(coupled_spin),
        m3=sp.Rational(decay_particle_lambda),
    )
    return sp.Mul(cg_ls, cg_ss, evaluate=False)


def formulate_wigner_d(transition: StateTransition, node_id: int) -> sp.Expr:
    r"""Compute `~sympy.physics.quantum.spin.WignerD` for a transition node.

    Following :cite:`kutschkeAngularDistributionCookbook1996`, Eq. (10). For a
    two-body decay :math:`1 \to 2, 3`, we get

    .. math:: D^{s_1}_{m_1,\lambda_2-\lambda_3}(-\phi,\theta,0)
        :label: formulate_wigner_d

    with:

    - :math:`s_1` the `Spin.magnitude <qrules.particle.Spin.magnitude>` of the
      decaying state,
    - :math:`m_1` the `~qrules.transition.State.spin_projection` of the
      decaying state,
    - :math:`\lambda_{2}, \lambda_{3}` the helicities of the decay products in
      in the restframe of :math:`1` (can be taken to be their intrinsic
      `~qrules.transition.State.spin_projection` when following a constistent
      boosting procedure),
    - and :math:`\phi` and :math:`\theta` the helicity angles (see also
      :func:`.get_helicity_angle_label`).

    Note that :math:`\lambda_2, \lambda_3` are ordered by their number of
    children, then by their state ID (see :class:`.TwoBodyDecay`).

    See :cite:`kutschkeAngularDistributionCookbook1996`, Eq. (30) for an
    example of Wigner-:math:`D` functions in a *sequential* two-body decay.

    Example
    -------
    >>> import qrules
    >>> reaction = qrules.generate_transitions(
    ...     initial_state=[("J/psi(1S)", [+1])],
    ...     final_state=[("gamma", [-1]), "f(0)(980)"],
    ... )
    >>> transition = reaction.transitions[0]
    >>> formulate_wigner_d(transition, node_id=0)
    WignerD(1, 1, -1, -phi_0, theta_0, 0)

    .. math::
        D^{s_1}_{m_1,\lambda_2-\lambda_3}\left(-\phi,\theta,0\right)
        = D^{1}_{+1,(-1-0)}\left(-\phi_0,\theta_0,0\right)
        = D^{1}_{1,-1}\left(-\phi_0,\theta_0,0\right)
    """
    from sympy.physics.quantum.spin import Rotation as Wigner

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
    """Calculate the product of all prefactors defined in this transition.

    .. seealso:: `qrules.quantum_numbers.InteractionProperties.parity_prefactor`
    """
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
    ] = collections.defaultdict(list)
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
