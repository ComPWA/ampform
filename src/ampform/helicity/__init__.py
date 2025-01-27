"""Generate an amplitude model with the helicity formalism.

.. autolink-preface::

    import sympy as sp
"""

from __future__ import annotations

import collections
import logging
import operator
import sys
import warnings
from collections import OrderedDict, abc
from fractions import Fraction
from functools import reduce, singledispatchmethod
from typing import TYPE_CHECKING, Union

import attrs
import sympy as sp
from attrs import define, field, frozen
from attrs.validators import deep_iterable, instance_of, optional
from qrules.combinatorics import perform_external_edge_identical_particle_combinatorics
from qrules.particle import Particle
from qrules.transition import (
    InteractionProperties,
    ReactionInfo,
    State,
    StateTransition,
)

from ampform._qrules import get_qrules_version
from ampform.dynamics.builder import (
    ResonanceDynamicsBuilder,
    TwoBodyKinematicVariableSet,
    create_non_dynamic,
)
from ampform.helicity.align import NoAlignment, SpinAlignment
from ampform.helicity.decay import (
    TwoBodyDecay,
    get_prefactor,
    group_by_spin_projection,
    group_by_topology,
)
from ampform.helicity.naming import (
    CanonicalAmplitudeNameGenerator,
    HelicityAmplitudeNameGenerator,
    NameGenerator,
    collect_spin_projections,
    create_amplitude_symbol,
    generate_transition_label,
    get_helicity_angle_symbols,
    natural_sorting,
)
from ampform.kinematics import HelicityAdapter
from ampform.kinematics.lorentz import (
    InvariantMass,
    create_four_momentum_symbols,
    get_invariant_mass_symbol,
)
from ampform.sympy import PoolSum, determine_indices
from ampform.sympy._array_expressions import ArraySum

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
if TYPE_CHECKING:
    from collections.abc import (
        ItemsView,
        Iterable,
        Iterator,
        KeysView,
        Mapping,
        Sequence,
        ValuesView,
    )

    from IPython.lib.pretty import PrettyPrinter
    from qrules.topology import MutableTransition

_LOGGER = logging.getLogger(__name__)


def _order_component_mapping(
    mapping: Mapping[str, sp.Expr],
) -> OrderedDict[str, sp.Expr]:
    return collections.OrderedDict([
        (key, mapping[key]) for key in sorted(mapping, key=natural_sorting)
    ])


def _order_symbol_mapping(
    mapping: Mapping[sp.Symbol, sp.Expr],
) -> OrderedDict[sp.Symbol, sp.Expr]:
    return collections.OrderedDict([
        (symbol, mapping[symbol])
        for symbol in sorted(mapping, key=lambda s: natural_sorting(s.name))
    ])


def _order_amplitudes(
    mapping: Mapping[sp.Indexed, sp.Expr],
) -> OrderedDict[sp.Indexed, sp.Expr]:
    return collections.OrderedDict([
        (key, mapping[key])
        for key in sorted(mapping, key=lambda a: natural_sorting(str(a)))
    ])


def _to_parameter_values(mapping: Mapping[sp.Basic, ParameterValue]) -> ParameterValues:
    return ParameterValues(mapping)


@frozen
class HelicityModel:
    intensity: PoolSum = field(validator=instance_of(PoolSum))
    """Main expression describing the intensity over `kinematic_variables`."""
    amplitudes: OrderedDict[sp.Indexed, sp.Expr] = field(converter=_order_amplitudes)
    """Definitions for the amplitudes that appear in `intensity`.

    The main `intensity` is a sum over amplitudes for each initial and final state
    helicity combination. These amplitudes are indicated with as `sp.Indexed
    <sympy.tensor.indexed.Indexed>` instances and this attribute provides the
    definitions for each of these. See also
    :ref:`TR-014 <compwa-report:tr-014-solution-2>`.
    """
    parameter_defaults: ParameterValues = field(converter=_to_parameter_values)
    """A mapping of suggested parameter values.

    Keys are `~sympy.core.basic.Basic` instances from the main :attr:`expression` that
    should be interpreted as parameters (as opposed to `kinematic_variables`). The
    symbols are ordered alphabetically by name with natural sort order
    (:func:`.natural_sorting`). Values have been extracted from the input
    `~qrules.transition.ReactionInfo`.
    """
    kinematic_variables: OrderedDict[sp.Symbol, sp.Expr] = field(
        converter=_order_symbol_mapping
    )
    """Expressions for converting four-momenta to kinematic variables."""
    components: OrderedDict[str, sp.Expr] = field(converter=_order_component_mapping)
    """A mapping for identifying main components in the :attr:`expression`.

    Keys are the component names (`str`), formatted as LaTeX, and values are sub-
    expressions in the main :attr:`expression`. The mapping is an
    `~collections.OrderedDict` that orders the component names alphabetically with
    natural sort order (:func:`.natural_sorting`).
    """
    reaction_info: ReactionInfo = field(validator=instance_of(ReactionInfo))

    @property
    def expression(self) -> sp.Expr:
        """Expression for the `intensity` with all amplitudes fully expressed.

        Constructed from `intensity` by substituting its amplitude symbols with the
        definitions with `amplitudes`.
        """

        def unfold_poolsums(expr: sp.Expr) -> sp.Expr:
            new_expr = expr
            for node in sp.postorder_traversal(expr):
                if isinstance(node, PoolSum):
                    new_expr = new_expr.xreplace({node: node.evaluate()})
            return new_expr

        intensity = self.intensity.evaluate()
        intensity = unfold_poolsums(intensity)
        return intensity.xreplace(self.amplitudes)

    def rename_symbols(
        self, renames: Iterable[tuple[str, str]] | Mapping[str, str]
    ) -> HelicityModel:
        """Rename certain symbols in the model.

        Renames all `~sympy.core.symbol.Symbol` instance that appear in `expression`,
        `parameter_defaults`, `components`, and `kinematic_variables`. This method can
        be used to :ref:`couple parameters <usage/modify:Couple parameters>`.

        Args:
            renames: A mapping from old to new names.

        Returns:
            A **new** instance of a `HelicityModel` with symbols in all attributes
            renamed accordingly.
        """
        renames = dict(renames)
        if not renames:
            return self
        symbols = self.__collect_symbols()
        symbol_names = {s.name for s in symbols}
        for name in renames:
            if name not in symbol_names:
                _LOGGER.warning(f"There is no symbol with name {name}")
        symbol_mapping = {
            s: sp.Symbol(renames[s.name], **s.assumptions0) if s.name in renames else s
            for s in symbols
        }
        return attrs.evolve(
            self,
            intensity=self.intensity.xreplace(symbol_mapping),
            amplitudes={
                amp: expr.xreplace(symbol_mapping)
                for amp, expr in self.amplitudes.items()
            },
            parameter_defaults={
                symbol_mapping.get(par, par): value  # type: ignore[call-overload]
                for par, value in self.parameter_defaults.items()
            },
            components={
                name: expr.xreplace(symbol_mapping)
                for name, expr in self.components.items()
            },
            kinematic_variables={
                symbol_mapping.get(var, var): expr.xreplace(symbol_mapping)
                for var, expr in self.kinematic_variables.items()
            },
        )

    def __collect_symbols(self) -> set[sp.Symbol]:
        symbols: set[sp.Symbol] = self.expression.free_symbols  # type: ignore[assignment]
        symbols |= set(self.kinematic_variables)
        for expr in self.kinematic_variables.values():
            symbols |= expr.free_symbols  # type: ignore[arg-type]
        return symbols


class ParameterValues(abc.Mapping):
    """Ordered mapping to `ParameterValue` with convenient getter and setter.

    This class makes it possible to search through a mapping of :mod:`sympy` symbols to
    their values (a "parameter mapping") by symbol name or by index in the (ordered)
    dictionary.

    >>> a, b, c = sp.symbols("a b c")
    >>> parameters = ParameterValues({a: 0.0, b: 1 + 1j, c: -2})
    >>> parameters[a]
    0.0
    >>> parameters["b"]
    (1+1j)
    >>> parameters["b"] = 3
    >>> parameters[1]
    3
    >>> parameters[2]
    -2
    >>> parameters[2] = 3.14
    >>> parameters[c]
    3.14

    .. automethod:: __getitem__
    .. automethod:: __setitem__
    """

    def __init__(self, parameters: Mapping[sp.Basic, ParameterValue]) -> None:
        self.__parameters = dict(parameters)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__parameters})"

    def _repr_pretty_(self, p: PrettyPrinter, cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}({{"):
                p.breakable()
                for par, value in self.items():
                    p.pretty(par)  # type: ignore[attr-defined]
                    p.text(": ")
                    p.pretty(value)  # type: ignore[attr-defined]
                    p.text(",")
                    p.breakable()
            p.text("})")

    def __getitem__(self, key: sp.Basic | int | str) -> ParameterValue:
        par = self._get_parameter(key)
        return self.__parameters[par]

    def __setitem__(self, key: sp.Basic | int | str, value: ParameterValue) -> None:
        par = self._get_parameter(key)
        self.__parameters[par] = value

    @singledispatchmethod
    def _get_parameter(self, key: sp.Basic | int | str) -> sp.Basic:  # noqa: PLR6301
        msg = f"Cannot find parameter for key type {type(key).__name__}"
        raise KeyError(msg)  # no TypeError because of sympy.core.expr.Expr.xreplace

    @_get_parameter.register(sp.Basic)
    def _(self, par: sp.Basic) -> sp.Basic:
        if par not in self.__parameters:
            msg = f"{type(self).__name__} has no parameter {par}"
            raise KeyError(msg)
        return par

    @_get_parameter.register(str)
    def _(self, name: str) -> sp.Basic:
        for parameter in self.__parameters:
            if str(parameter) == name:
                return parameter
        msg = f"No parameter available with name {name}"
        raise KeyError(msg)

    @_get_parameter.register(int)
    def _(self, key: int) -> sp.Basic:
        for i, parameter in enumerate(self.__parameters):
            if i == key:
                return parameter
        msg = (
            f"Parameter mapping has {len(self)} parameters, but trying to get parameter"
            f" number {key}"
        )
        raise KeyError(msg)

    def __len__(self) -> int:
        return len(self.__parameters)

    def __iter__(self) -> Iterator[sp.Basic]:
        return iter(self.__parameters)

    def items(self) -> ItemsView[sp.Basic, ParameterValue]:
        return self.__parameters.items()

    def keys(self) -> KeysView[sp.Basic]:
        return self.__parameters.keys()

    def values(self) -> ValuesView[ParameterValue]:
        return self.__parameters.values()


ParameterValue = Union[float, complex, int]
"""Allowed value types for parameters."""


class HelicityAmplitudeBuilder:
    """Amplitude model generator for the helicity formalism."""

    def __init__(self, reaction: ReactionInfo) -> None:
        if len(reaction.transitions) < 1:
            msg = (
                f"At least one {StateTransition.__name__} required to genenerate an"
                " amplitude model!"
            )
            raise ValueError(msg)
        self.__reaction = reaction
        self.__adapter = HelicityAdapter(reaction)
        self.__config = BuilderConfiguration(
            spin_alignment=NoAlignment(),
            scalar_initial_state_mass=False,
            stable_final_state_ids=None,
            use_helicity_couplings=False,
        )
        self.__dynamics = DynamicsSelector(reaction)
        self._naming: NameGenerator = HelicityAmplitudeNameGenerator(reaction)
        self.__ingredients = _HelicityModelIngredients()

    @property
    def adapter(self) -> HelicityAdapter:
        """Converter for computing kinematic variables from four-momenta."""
        return self.__adapter

    @property
    def config(self) -> BuilderConfiguration:
        return self.__config

    @property
    def dynamics(self) -> DynamicsSelector:
        return self.__dynamics

    @property
    def naming(self) -> NameGenerator:
        return self._naming

    @property
    def reaction(self) -> ReactionInfo:
        return self.__reaction

    def set_dynamics(
        self, particle_name: str, dynamics_builder: ResonanceDynamicsBuilder
    ) -> None:
        """Assign a `.ResonanceDynamicsBuilder` for a specific resonance.

        .. deprecated:: 0.16.0
            Use the `~.DynamicsSelector.assign()` method of the `.dynamics` attribute
            instead.
        """
        warnings.warn(
            "set_dynamics() will be removed in favor of dynamics.assign()",
            category=DeprecationWarning,
            stacklevel=1,
        )
        self.dynamics.assign(particle_name, dynamics_builder)

    def formulate(self) -> HelicityModel:
        self.__ingredients.reset()
        main_intensity = self.__formulate_top_expression()
        kinematic_variables = self.adapter.create_expressions()
        if self.config.stable_final_state_ids is not None:
            for state_id in self.config.stable_final_state_ids:
                mass_symbol = sp.Symbol(f"m_{state_id}", nonnegative=True)
                particle = self.reaction.final_state[state_id]
                self.__ingredients.parameter_defaults[mass_symbol] = particle.mass
                del kinematic_variables[mass_symbol]
        if self.config.scalar_initial_state_mass:
            subscript = "".join(map(str, sorted(self.reaction.final_state)))
            mass_symbol = sp.Symbol(f"m_{subscript}", nonnegative=True)
            particle = next(iter(self.reaction.initial_state.values()))
            self.__ingredients.parameter_defaults[mass_symbol] = particle.mass
            del kinematic_variables[mass_symbol]

        alignment_symbols = self.config.spin_alignment.define_symbols(self.reaction)
        p = create_four_momentum_symbols(self.reaction.transitions[0].topology)
        for angle_symbol, angle_expr in alignment_symbols.items():
            angle_expr = angle_expr.xreplace(kinematic_variables)
            remaining_mass_symbols = [
                s
                for s in sorted(angle_expr.free_symbols, key=str)
                if isinstance(s, sp.Symbol)
                if s.name.startswith("m_")
                if s.is_nonnegative  # type: ignore[attr-defined]
            ]
            for mass_symbol in remaining_mass_symbols:
                indices = _get_final_state_ids(mass_symbol)
                if set(indices) == set(self.reaction.initial_state):
                    if self.config.scalar_initial_state_mass:
                        self.__ingredients.parameter_defaults[mass_symbol] = (
                            self.reaction.initial_state[0].mass
                        )
                        continue
                    indices = tuple(sorted(self.reaction.final_state))
                if (
                    len(indices) == 1
                    and self.config.stable_final_state_ids is not None
                    and indices[0] in self.config.stable_final_state_ids
                ):
                    continue
                momentum = ArraySum(*[p[i] for i in sorted(indices)])
                kinematic_variables[mass_symbol] = InvariantMass(momentum)
            angle_expr = angle_expr.xreplace(kinematic_variables)
            alignment_symbols[angle_symbol] = angle_expr
        kinematic_variables.update(alignment_symbols)

        return HelicityModel(
            intensity=main_intensity,
            amplitudes=self.__ingredients.amplitudes,
            parameter_defaults=self.__ingredients.parameter_defaults,
            kinematic_variables=kinematic_variables,
            components=self.__ingredients.components,
            reaction_info=self.reaction,
        )

    def __formulate_top_expression(self) -> PoolSum:
        spin_groups = group_by_spin_projection(self.reaction.transitions)
        for group in spin_groups:
            self.__register_amplitudes(group)

        amplitude = self.config.spin_alignment.formulate_amplitude(self.reaction)
        spin_projections = collect_spin_projections(self.reaction)
        return PoolSum(sp.Abs(amplitude) ** 2, *spin_projections.items())

    def __register_amplitudes(self, transition_group: list[StateTransition]) -> None:
        transition_by_topology = group_by_topology(transition_group)
        expression = sum(
            self.__formulate_topology_amplitude(transitions)
            for transitions in transition_by_topology.values()
        )
        first_transition = transition_group[0]
        graph_group_label = generate_transition_label(first_transition)
        component_name = f"I_{{{graph_group_label}}}"
        self.__ingredients.components[component_name] = sp.Abs(expression) ** 2

    def __formulate_topology_amplitude(
        self, transitions: Sequence[StateTransition]
    ) -> sp.Expr:
        sequential_expressions: list[sp.Expr] = []
        for transition in transitions:
            sequential_graphs = _perform_combinatorics(transition)
            for graph in sequential_graphs:
                first_transition = _freeze(graph)
                expression = self.__formulate_sequential_decay(first_transition)
                sequential_expressions.append(expression)

        first_transition = transitions[0]
        symbol = create_amplitude_symbol(first_transition)
        expression = sum(sequential_expressions)  # type: ignore[assignment]
        self.__ingredients.amplitudes[symbol] = expression
        return expression

    def __formulate_sequential_decay(self, transition: StateTransition) -> sp.Expr:
        partial_decays: list[sp.Expr] = [
            self._formulate_partial_decay(transition, node_id)
            for node_id in transition.topology.nodes
        ]
        sequential_amplitudes = reduce(operator.mul, partial_decays)

        if self.config.use_helicity_couplings:
            expression = sequential_amplitudes
        else:
            coefficient = self.__generate_amplitude_coefficient(transition)
            expression = coefficient * sequential_amplitudes
        prefactor = self.__generate_amplitude_prefactor(transition)
        if prefactor is not None:
            expression *= prefactor
        subscript = self.naming.generate_amplitude_name(transition)
        self.__ingredients.components[f"A_{{{subscript}}}"] = expression
        return expression

    def _formulate_partial_decay(
        self, transition: StateTransition, node_id: int
    ) -> sp.Expr:
        wigner_d = formulate_isobar_wigner_d(transition, node_id)
        dynamics = self.__formulate_dynamics(transition, node_id)
        if self.config.use_helicity_couplings:
            coupling = self.__generate_helicity_coupling(transition, node_id)
            return coupling * wigner_d * dynamics
        return wigner_d * dynamics

    def __formulate_dynamics(
        self, transition: StateTransition, node_id: int
    ) -> sp.Expr:
        decay = TwoBodyDecay.from_transition(transition, node_id)
        if decay not in self.dynamics:
            return sp.S.One

        builder = self.dynamics[decay]
        variable_set = _generate_kinematic_variable_set(transition, node_id)
        expression, parameters = builder(decay.parent.particle, variable_set)
        for par, value in parameters.items():
            if par in self.__ingredients.parameter_defaults:
                previous_value = self.__ingredients.parameter_defaults[par]
                if value != previous_value:
                    _LOGGER.warning(
                        f'New default value {value} for parameter "{par.name}"'
                        " is inconsistent with existing value"
                        f" {previous_value}"
                    )
            self.__ingredients.parameter_defaults[par] = value

        return expression

    def __generate_amplitude_coefficient(
        self, transition: StateTransition
    ) -> sp.Symbol:
        """Generate coefficient parameter for a sequential amplitude.

        Generally, each partial amplitude of a sequential amplitude transition should
        check itself if it or a parity partner is already defined. If so a coupled
        coefficient is introduced.
        """
        suffix = self.naming.generate_sequential_amplitude_suffix(transition)
        symbol = sp.Symbol(f"C_{{{suffix}}}")
        value = complex(1, 0)
        self.__ingredients.parameter_defaults[symbol] = value
        return symbol

    def __generate_helicity_coupling(
        self, transition: StateTransition, node_id: int
    ) -> sp.Symbol:
        suffix = self.naming.generate_two_body_decay_suffix(transition, node_id)
        symbol = sp.Symbol(f"H_{{{suffix}}}")
        value = complex(1, 0)
        self.__ingredients.parameter_defaults[symbol] = value
        return symbol

    def __generate_amplitude_prefactor(
        self, transition: StateTransition
    ) -> sp.Rational | None:
        prefactor = get_prefactor(transition)
        if prefactor != 1.0:
            for node_id in transition.topology.nodes:
                raw_suffix = self.naming.generate_two_body_decay_suffix(
                    transition, node_id
                )
                if raw_suffix in self.naming.parity_partner_coefficient_mapping:
                    coefficient_suffix = self.naming.parity_partner_coefficient_mapping[
                        raw_suffix
                    ]
                    if coefficient_suffix != raw_suffix:
                        return sp.Rational(prefactor)
        return None


def _perform_combinatorics(
    transition: StateTransition,
) -> list[MutableTransition[State, InteractionProperties]]:
    if get_qrules_version() < (0, 10):
        return perform_external_edge_identical_particle_combinatorics(
            transition.to_graph()  # type: ignore[attr-defined]
        )
    graph = transition.convert(lambda s: (s.particle, s.spin_projection)).unfreeze()
    combinations = perform_external_edge_identical_particle_combinatorics(graph)
    return [g.freeze().convert(lambda s: State(*s)).unfreeze() for g in combinations]


def _freeze(graph: MutableTransition[State, InteractionProperties]) -> StateTransition:
    if get_qrules_version() < (0, 10):
        return StateTransition.from_graph(graph)  # type: ignore[attr-defined]
    return graph.freeze()


class CanonicalAmplitudeBuilder(HelicityAmplitudeBuilder):
    r"""Amplitude model generator for the canonical helicity formalism.

    This class defines a full amplitude in the canonical formalism, using the helicity
    formalism as a foundation. The key here is that we take the full helicity intensity
    as a template, and just exchange the helicity amplitudes :math:`F` as a sum of
    canonical amplitudes :math:`A`:

    .. math::

        F^J_{\lambda_1,\lambda_2} = \sum_{LS} \mathrm{norm}(A^J_{LS})C^2.

    Here, :math:`C` stands for `Clebsch-Gordan factor
    <https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients>`_.

    .. seealso:: `HelicityAmplitudeBuilder` and :doc:`/usage/helicity/formalism`.
    """

    @override
    def __init__(self, reaction: ReactionInfo) -> None:
        super().__init__(reaction)
        self._naming = CanonicalAmplitudeNameGenerator(reaction)

    @override
    def _formulate_partial_decay(
        self, transition: StateTransition, node_id: int
    ) -> sp.Expr:
        amplitude = super()._formulate_partial_decay(transition, node_id)
        cg_coefficients = formulate_isobar_cg_coefficients(transition, node_id)
        return cg_coefficients * amplitude


def _to_optional_set(values: Iterable[int] | None) -> set[int] | None:
    if values is None:
        return None
    return set(values)


@define
class BuilderConfiguration:
    """Configuration class for a `.HelicityAmplitudeBuilder`."""

    spin_alignment: SpinAlignment = field(validator=instance_of(SpinAlignment))  # type: ignore[type-abstract]
    """Method for :doc:`aligning spin </usage/helicity/spin-alignment>`."""
    scalar_initial_state_mass: bool = field(validator=instance_of(bool))
    r"""Add initial state mass as scalar value to `.parameter_defaults`.

    Put the invariant of the initial state (:math:`m_{012\dots}`) under
    `.HelicityModel.parameter_defaults` (with a *scalar* suggested value) instead of
    `~.HelicityModel.kinematic_variables`. This is useful if four-momenta were generated
    with or kinematically fit to a specific initial state energy.

    .. seealso:: :ref:`usage/amplitude:Scalar masses`
    """
    stable_final_state_ids: set[int] | None = field(
        converter=_to_optional_set,
        validator=optional(deep_iterable(member_validator=instance_of(int))),  # type: ignore[arg-type]
    )
    r"""IDs of the final states that should be considered stable.

    Put final state 'invariant' masses (:math:`m_0, m_1, \dots`) under
    `.HelicityModel.parameter_defaults` (with a *scalar* suggested value) instead of
    `~.HelicityModel.kinematic_variables` (which are expressions to compute an event-
    wise array of invariant masses). This is useful if final state particles are stable.
    """
    use_helicity_couplings: bool = field(validator=instance_of(bool))
    """Use helicity couplings instead of amplitude coefficients.

    Helicity couplings are a measure for the strength of each partial two-body decay.
    Amplitude coefficients are the product of those couplings.
    """


class DynamicsSelector(abc.Mapping):
    """Configure which `.ResonanceDynamicsBuilder` to use for each node."""

    def __init__(self, transitions: ReactionInfo | Iterable[StateTransition]) -> None:
        if isinstance(transitions, ReactionInfo):
            transitions = transitions.transitions
        self.__choices: dict[TwoBodyDecay, ResonanceDynamicsBuilder] = {}
        for transition in transitions:
            for node_id in transition.topology.nodes:
                decay = TwoBodyDecay.from_transition(transition, node_id)
                self.__choices[decay] = create_non_dynamic

    @singledispatchmethod
    def assign(self, selection, builder: ResonanceDynamicsBuilder) -> None:
        """Assign a `.ResonanceDynamicsBuilder` to a selection of nodes.

        Currently, the following types of selections are implements:

        - `str`: Select transition nodes by the name of the `~.TwoBodyDecay.parent`
          `~qrules.particle.Particle`.
        - `.TwoBodyDecay` or `tuple` of a `~qrules.topology.Transition` with a
          node ID: set dynamics for one specific transition node.
        """
        msg = (
            f"Cannot set dynamics builder for selection type {type(selection).__name__}"
        )
        raise NotImplementedError(msg)

    @assign.register(TwoBodyDecay)
    def _(self, decay: TwoBodyDecay, builder: ResonanceDynamicsBuilder) -> None:
        self.__choices[decay] = builder

    @assign.register(tuple)
    def _(
        self,
        transition_node: tuple[StateTransition, int],
        builder: ResonanceDynamicsBuilder,
    ) -> None:
        decay = TwoBodyDecay.create(transition_node)
        return self.assign(decay, builder)

    @assign.register(str)
    def _(self, particle_name: str, builder: ResonanceDynamicsBuilder) -> None:
        found_particle = False
        for decay in self.__choices:
            decaying_particle = decay.parent.particle
            if decaying_particle.name == particle_name:
                self.__choices[decay] = builder
                found_particle = True
        if not found_particle:
            _LOGGER.warning(f'Model contains no resonance with name "{particle_name}"')

    @assign.register(Particle)
    def _(self, particle: Particle, builder: ResonanceDynamicsBuilder) -> None:
        return self.assign(particle.name, builder)

    def __getitem__(
        self, k: TwoBodyDecay | tuple[StateTransition, int], /
    ) -> ResonanceDynamicsBuilder:
        k = TwoBodyDecay.create(k)
        return self.__choices[k]

    def __len__(self) -> int:
        return len(self.__choices)

    def __iter__(self) -> Iterator[TwoBodyDecay]:
        return iter(self.__choices)

    def items(self) -> ItemsView[TwoBodyDecay, ResonanceDynamicsBuilder]:
        return self.__choices.items()

    def keys(self) -> KeysView[TwoBodyDecay]:
        return self.__choices.keys()

    def values(self) -> ValuesView[ResonanceDynamicsBuilder]:
        return self.__choices.values()


@define
class _HelicityModelIngredients:
    parameter_defaults: dict[sp.Basic, ParameterValue] = field(factory=dict)
    amplitudes: dict[sp.Indexed, sp.Expr] = field(factory=dict)
    components: dict[str, sp.Expr] = field(factory=dict)
    kinematic_variables: dict[sp.Symbol, sp.Expr] = field(factory=dict)

    def reset(self) -> None:
        self.parameter_defaults = {}
        self.amplitudes = {}
        self.components = {}
        self.kinematic_variables = {}


def formulate_isobar_cg_coefficients(
    transition: StateTransition, node_id: int
) -> sp.Expr:
    r"""Compute the two Clebsch-Gordan coefficients for an isobar node.

    In the **canonical basis** (also called **partial wave basis**),
    :doc:`Clebsch-Gordan coefficients <sympy:modules/physics/quantum/cg>` ensure that
    the projection of angular momentum is conserved
    (:cite:`kutschkeAngularDistributionCookbook1996`, p. 4). When calling
    :func:`~qrules.generate_transitions` with :code:`formalism="canonical-helicity"`,
    AmpForm formulates the amplitude in the canonical basis from amplitudes in the
    helicity basis using the transformation in :cite:`chungSpinFormalismsUpdated2014`,
    Eq. (4.32). See also :cite:`kutschkeAngularDistributionCookbook1996`, Eq. (28).

    This function produces the two Clebsch-Gordan coefficients in
    :cite:`chungSpinFormalismsUpdated2014`, Eq. (4.32). For a two-body decay :math:`1
    \to 2, 3`, we get:

    .. math:: C^{s_1,\lambda}_{L,0,S,\lambda} C^{S,\lambda}_{s_2,\lambda_2,s_3,-\lambda_3}
        :label: formulate_isobar_cg_coefficients

    with:

    - :math:`s_i` the intrinsic `Spin.magnitude <qrules.particle.Spin.magnitude>` of
      each state :math:`i`,
    - :math:`\lambda_{2}, \lambda_{3}` the helicities of the decay products (can be
      taken to be their `~qrules.transition.State.spin_projection` when following a
      constistent boosting procedure),
    - :math:`\lambda=\lambda_{2}-\lambda_{3}`,
    - :math:`L` the *total* angular momentum of the final state pair
      (`~qrules.quantum_numbers.InteractionProperties.l_magnitude`),
    - :math:`S` the coupled spin magnitude of the final state pair
      (`~qrules.quantum_numbers.InteractionProperties.s_magnitude`),
    - and :math:`C^{j_3,m_3}_{j_1,m_1,j_2,m_2} = \langle j1,m1;j2,m2|j3,m3\rangle`, as
      in :doc:`sympy:modules/physics/quantum/cg`.

    Example
    -------
    >>> import qrules
    >>> reaction = qrules.generate_transitions(
    ...     initial_state=[("J/psi(1S)", [+1])],
    ...     final_state=[("gamma", [-1]), "f(0)(980)"],
    ... )
    >>> transition = reaction.transitions[1]  # angular momentum 2
    >>> formulate_isobar_cg_coefficients(transition, node_id=0)
    CG(1, -1, 0, 0, 1, -1)*CG(2, 0, 1, -1, 1, -1)

    .. math::
        C^{s_1,\lambda}_{L,0,S,\lambda} C^{S,\lambda}_{s_2,\lambda_2,s_3,-\lambda_3}
        = C^{1,(-1-0)}_{2,0,1,(-1-0)} C^{1,(-1-0)}_{1,-1,0,0}
        = C^{1,-1}_{2,0,1,-1} C^{1,-1}_{1,-1,0,0}
    """
    from sympy.physics.quantum.cg import CG  # noqa: PLC0415

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


def formulate_isobar_wigner_d(transition: StateTransition, node_id: int) -> sp.Expr:
    r"""Compute `~sympy.physics.quantum.spin.WignerD` for an isobar node.

    Following :cite:`chungSpinFormalismsUpdated2014`, `Eq. (4.16)
    <https://suchung.web.cern.ch/spinfm1.pdf#page=16>`_, but taking the complex
    conjugate by flipping the sign of the azimuthal angle :math:`\phi` (see relation
    between Wigner-:math:`D` and Wigner-:math:`d` in `Eq. (A.1)
    <https://suchung.web.cern.ch/spinfm1.pdf#page=83>`_).

    For a two-body decay :math:`1 \to 2, 3`, this gives us:

    .. math:: D^{s_1}_{m_1,\lambda_2-\lambda_3}(-\phi,\theta,0)
        :label: formulate_isobar_wigner_d

    with:

    - :math:`s_1` the `Spin.magnitude <qrules.particle.Spin.magnitude>` of the decaying
      state,
    - :math:`m_1` the `~qrules.transition.State.spin_projection` of the decaying state,
    - :math:`\lambda_{2}, \lambda_{3}` the helicities of the decay products in in the
      restframe of :math:`1` (can be taken to be their intrinsic
      `~qrules.transition.State.spin_projection` when following a constistent boosting
      procedure),
    - and :math:`\phi` and :math:`\theta` the helicity angles (see also
      :func:`.get_helicity_angle_symbols`).

    Note that :math:`\lambda_2, \lambda_3` are ordered by their number of children, then
    by their state ID (see :class:`.TwoBodyDecay`).

    See :cite:`kutschkeAngularDistributionCookbook1996`, Eq. (30) for an example of
    Wigner-:math:`D` functions in a *sequential* two-body decay. Note that this source
    chose :math:`\Omega=(\phi,\theta,-\phi)` as argument to the (conjugated)
    Wigner-:math:`D` function, just like the original paper by Jacob & Wick
    :cite:`Jacob:1959at`, Eq. (24). See p.119-120 and p.199 in :cite:`Martin:1970hmp`
    for the two conventions, :math:`\gamma=0` versus :math:`\gamma=-\phi`.

    Example
    -------
    >>> import qrules
    >>> reaction = qrules.generate_transitions(
    ...     initial_state=[("J/psi(1S)", [+1])],
    ...     final_state=[("gamma", [-1]), "f(0)(980)"],
    ... )
    >>> transition = reaction.transitions[0]
    >>> formulate_isobar_wigner_d(transition, node_id=0)
    WignerD(1, 1, -1, -phi_0, theta_0, 0)
    """
    from sympy.physics.quantum.spin import Rotation as Wigner  # noqa: PLC0415

    decay = TwoBodyDecay.from_transition(transition, node_id)
    _, phi, theta = _generate_kinematic_variables(transition, node_id)
    return Wigner.D(
        j=sp.Rational(decay.parent.particle.spin),
        m=sp.Rational(decay.parent.spin_projection),
        mp=sp.Rational(
            decay.children[0].spin_projection - decay.children[1].spin_projection
        ),
        alpha=-phi,  # complex conjugate
        beta=theta,
        gamma=0,
    )


def _get_final_state_ids(mass: sp.Symbol) -> tuple[int, ...]:
    """Extract the final state IDs from a mass symbol.

    >>> _get_final_state_ids(sp.Symbol("m_1"))
    (1,)
    >>> _get_final_state_ids(sp.Symbol("m_123"))
    (1, 2, 3)
    """
    subscript_indices = determine_indices(mass)
    if len(subscript_indices) != 1:
        msg = f"Could not determine indices from mass symbol {mass}"
        raise ValueError(msg)
    subscript = str(subscript_indices[0])
    return tuple(int(s) for s in subscript)


def _generate_kinematic_variable_set(
    transition: StateTransition, node_id: int
) -> TwoBodyKinematicVariableSet:
    def is_integer(spin: float | Fraction) -> bool:
        if isinstance(spin, float):
            return spin.is_integer()
        if isinstance(spin, Fraction):
            return spin.denominator == 1
        msg = "spin has to of type float (qrules 0.9) or Fraction (qrules >= 0.10)"
        raise TypeError(msg)

    decay = TwoBodyDecay.from_transition(transition, node_id)
    inv_mass, phi, theta = _generate_kinematic_variables(transition, node_id)
    topology = transition.topology
    child1_mass = get_invariant_mass_symbol(topology, decay.children[0].id)
    child2_mass = get_invariant_mass_symbol(topology, decay.children[1].id)
    angular_momentum: int | None = decay.interaction.l_magnitude
    if angular_momentum is None and is_integer(decay.parent.particle.spin):
        angular_momentum = int(decay.parent.particle.spin)
    return TwoBodyKinematicVariableSet(
        incoming_state_mass=inv_mass,
        outgoing_state_mass1=child1_mass,
        outgoing_state_mass2=child2_mass,
        helicity_theta=theta,
        helicity_phi=phi,
        angular_momentum=angular_momentum,
    )


def _generate_kinematic_variables(
    transition: StateTransition, node_id: int
) -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    """Generate symbol for invariant mass, phi angle, and theta angle."""
    decay = TwoBodyDecay.from_transition(transition, node_id)
    topology = transition.topology
    phi, theta = get_helicity_angle_symbols(topology, decay.children[0].id)
    invariant_mass = get_invariant_mass_symbol(topology, decay.parent.id)
    return invariant_mass, phi, theta
