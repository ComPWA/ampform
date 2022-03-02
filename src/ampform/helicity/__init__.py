# pylint: disable=import-outside-toplevel, too-many-arguments, too-many-lines
"""Generate an amplitude model with the helicity formalism.

.. autolink-preface::

    import sympy as sp
"""

import collections
import logging
import operator
import sys
from collections import OrderedDict, abc
from decimal import Decimal
from difflib import get_close_matches
from functools import reduce, singledispatch
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    Generator,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    ValuesView,
    overload,
)

import attrs
import sympy as sp
from attrs import define, field, frozen
from attrs.validators import instance_of
from qrules.combinatorics import (
    perform_external_edge_identical_particle_combinatorics,
)
from qrules.particle import Particle
from qrules.topology import Topology
from qrules.transition import ReactionInfo, StateTransition

from ampform.dynamics.builder import (
    ResonanceDynamicsBuilder,
    TwoBodyKinematicVariableSet,
    create_non_dynamic,
)
from ampform.kinematics import HelicityAdapter, get_invariant_mass_label
from ampform.sympy import PoolSum

from .decay import (
    TwoBodyDecay,
    collect_topologies,
    get_parent_id,
    get_sibling_state_id,
    is_opposite_helicity_state,
)
from .naming import (
    CanonicalAmplitudeNameGenerator,
    HelicityAmplitudeNameGenerator,
    generate_transition_label,
    get_helicity_angle_label,
    get_helicity_suffix,
    get_topology_identifier,
    natural_sorting,
)

if sys.version_info >= (3, 8):
    from functools import singledispatchmethod
    from typing import Literal
else:
    from singledispatchmethod import singledispatchmethod
    from typing_extensions import Literal

if TYPE_CHECKING:
    from IPython.lib.pretty import PrettyPrinter

ParameterValue = Union[float, complex, int]
"""Allowed value types for parameters."""


class ParameterValues(abc.Mapping):
    """Ordered mapping to `ParameterValue` with convenient getter and setter.

    >>> a, b, c = sp.symbols("a b c")
    >>> parameters = ParameterValues({a: 0.0, b: 1+1j, c: -2})
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

    def __init__(self, parameters: Mapping[sp.Symbol, ParameterValue]) -> None:
        self.__parameters = dict(parameters)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__parameters})"

    def _repr_pretty_(self, p: "PrettyPrinter", cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}({{"):
                p.breakable()
                for par, value in self.items():
                    p.pretty(par)
                    p.text(": ")
                    p.pretty(value)
                    p.text(",")
                    p.breakable()
            p.text("})")

    def __getitem__(self, key: Union[sp.Symbol, int, str]) -> "ParameterValue":
        par = self._get_parameter(key)
        return self.__parameters[par]

    def __setitem__(
        self, key: Union[sp.Symbol, int, str], value: "ParameterValue"
    ) -> None:
        par = self._get_parameter(key)
        self.__parameters[par] = value

    @singledispatchmethod
    def _get_parameter(self, key: Union[sp.Symbol, int, str]) -> sp.Symbol:
        # pylint: disable=no-self-use
        raise KeyError(  # no TypeError because of sympy.core.expr.Expr.xreplace
            f"Cannot find parameter for key type {type(key).__name__}"
        )

    @_get_parameter.register(sp.Symbol)
    def _(self, par: sp.Symbol) -> sp.Symbol:
        if par not in self.__parameters:
            raise KeyError(f"{type(self).__name__} has no parameter {par}")
        return par

    @_get_parameter.register(str)
    def _(self, name: str) -> sp.Symbol:
        for parameter in self.__parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"No parameter available with name {name}")

    @_get_parameter.register(int)
    def _(self, key: int) -> sp.Symbol:
        for i, parameter in enumerate(self.__parameters):
            if i == key:
                return parameter
        raise KeyError(
            f"Parameter mapping has {len(self)} parameters, but trying to get"
            f" parameter number {key}"
        )

    def __len__(self) -> int:
        return len(self.__parameters)

    def __iter__(self) -> Iterator[sp.Symbol]:
        return iter(self.__parameters)

    def items(self) -> ItemsView[sp.Symbol, ParameterValue]:
        return self.__parameters.items()

    def keys(self) -> KeysView[sp.Symbol]:
        return self.__parameters.keys()

    def values(self) -> ValuesView[ParameterValue]:
        return self.__parameters.values()


def _order_component_mapping(
    mapping: Mapping[str, sp.Expr]
) -> "OrderedDict[str, sp.Expr]":
    return collections.OrderedDict(
        [(key, mapping[key]) for key in sorted(mapping, key=natural_sorting)]
    )


def _order_symbol_mapping(
    mapping: Mapping[sp.Symbol, sp.Expr]
) -> "OrderedDict[sp.Symbol, sp.Expr]":
    return collections.OrderedDict(
        [
            (symbol, mapping[symbol])
            for symbol in sorted(
                mapping, key=lambda s: natural_sorting(s.name)
            )
        ]
    )


def _order_amplitudes(
    mapping: Mapping[sp.Indexed, sp.Expr]
) -> "OrderedDict[str,  sp.Expr]":
    return collections.OrderedDict(
        [
            (key, mapping[key])
            for key in sorted(mapping, key=lambda a: natural_sorting(str(a)))
        ]
    )


@frozen
class HelicityModel:  # noqa: R701
    intensity: PoolSum = field(validator=instance_of(PoolSum))
    """Main expression describing the intensity over `kinematic_variables`."""
    amplitudes: "OrderedDict[sp.Indexed, sp.Expr]" = field(
        converter=_order_amplitudes
    )
    """Definitions for the amplitudes that appear in `intensity`.

    The main `intensity` is a sum over amplitudes for each initial and final
    state helicity combination. These amplitudes are indicated with as
    `sp.Indexed <sympy.tensor.indexed.Indexed>` instances and this attribute
    provides the definitions for each of these. See also :ref:`TR-014
    <compwa-org:tr-014-solution-2>`.
    """
    parameter_defaults: ParameterValues = field(converter=ParameterValues)
    """A mapping of suggested parameter values.

    Keys are `~sympy.core.symbol.Symbol` instances from the main
    :attr:`expression` that should be interpreted as parameters (as opposed to
    `kinematic_variables`). The symbols are ordered alphabetically by name with
    natural sort order (:func:`.natural_sorting`). Values have been extracted
    from the input `~qrules.transition.ReactionInfo`.
    """
    kinematic_variables: "OrderedDict[sp.Symbol, sp.Expr]" = field(
        converter=_order_symbol_mapping
    )
    """Expressions for converting four-momenta to kinematic variables."""
    components: "OrderedDict[str, sp.Expr]" = field(
        converter=_order_component_mapping
    )
    """A mapping for identifying main components in the :attr:`expression`.

    Keys are the component names (`str`), formatted as LaTeX, and values are
    sub-expressions in the main :attr:`expression`. The mapping is an
    `~collections.OrderedDict` that orders the component names alphabetically
    with natural sort order (:func:`.natural_sorting`).
    """
    reaction_info: ReactionInfo = field(validator=instance_of(ReactionInfo))

    @property
    def expression(self) -> sp.Expr:
        """Expression for the `intensity` with all amplitudes fully expressed.

        Constructed from `intensity` by substituting its amplitude symbols with
        the definitions with `amplitudes`.
        """

        def unfold_poolsums(expr: sp.Expr) -> sp.Expr:
            new_expr = expr
            for node in sp.postorder_traversal(expr):
                if isinstance(node, PoolSum):
                    new_expr = new_expr.xreplace({node: node.evaluate()})
            return new_expr

        intensity = self.intensity.evaluate()
        intensity = unfold_poolsums(intensity)
        return intensity.subs(self.amplitudes)

    def rename_symbols(  # noqa: R701
        self, renames: Union[Iterable[Tuple[str, str]], Mapping[str, str]]
    ) -> "HelicityModel":
        """Rename certain symbols in the model.

        Renames all `~sympy.core.symbol.Symbol` instance that appear in
        `expression`, `parameter_defaults`, `components`, and
        `kinematic_variables`. This method can be used to :ref:`couple
        parameters <usage/modify:Couple parameters>`.

        Args:
            renames: A mapping from old to new names.

        Returns:
            A **new** instance of a `HelicityModel` with symbols in all
            attributes renamed accordingly.
        """
        renames = dict(renames)
        symbols = self.__collect_symbols()
        symbol_names = {s.name for s in symbols}
        for name in renames:
            if name not in symbol_names:
                logging.warning(f"There is no symbol with name {name}")
        symbol_mapping = {
            s: sp.Symbol(renames[s.name], **s.assumptions0)
            if s.name in renames
            else s
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
                symbol_mapping[par]: value
                for par, value in self.parameter_defaults.items()
            },
            components={
                name: expr.xreplace(symbol_mapping)
                for name, expr in self.components.items()
            },
            kinematic_variables={
                symbol_mapping[var]: expr.xreplace(symbol_mapping)
                for var, expr in self.kinematic_variables.items()
            },
        )

    def __collect_symbols(self) -> Set[sp.Symbol]:
        symbols: Set[sp.Symbol] = self.expression.free_symbols
        symbols |= set(self.kinematic_variables)
        for expr in self.kinematic_variables.values():
            symbols |= expr.free_symbols
        return symbols

    def sum_components(  # noqa: R701
        self, components: Iterable[str]
    ) -> sp.Expr:
        """Coherently or incoherently add components of a helicity model."""
        components = list(components)  # copy
        for component in components:
            if component not in self.components:
                first_letter = component[0]
                # pylint: disable=cell-var-from-loop
                candidates = get_close_matches(
                    component,
                    filter(
                        lambda c: c.startswith(first_letter), self.components
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


@define
class _HelicityModelIngredients:
    parameter_defaults: Dict[sp.Symbol, ParameterValue] = field(factory=dict)
    amplitudes: Dict[sp.Indexed, sp.Expr] = field(factory=dict)
    components: Dict[str, sp.Expr] = field(factory=dict)
    kinematic_variables: Dict[sp.Symbol, sp.Expr] = field(factory=dict)

    def reset(self) -> None:
        self.parameter_defaults = {}
        self.amplitudes = {}
        self.components = {}
        self.kinematic_variables = {}


class DynamicsSelector(abc.Mapping):
    """Configure which `.ResonanceDynamicsBuilder` to use for each node."""

    def __init__(
        self, transitions: Union[ReactionInfo, Iterable[StateTransition]]
    ) -> None:
        if isinstance(transitions, ReactionInfo):
            transitions = transitions.transitions
        self.__choices: Dict[TwoBodyDecay, ResonanceDynamicsBuilder] = {}
        for transition in transitions:
            for node_id in transition.topology.nodes:
                decay = TwoBodyDecay.from_transition(transition, node_id)
                self.__choices[decay] = create_non_dynamic

    @singledispatchmethod
    def assign(
        self, selection: Any, builder: ResonanceDynamicsBuilder
    ) -> None:
        """Assign a `.ResonanceDynamicsBuilder` to a selection of nodes.

        Currently, the following types of selections are implements:

        - `str`: Select transition nodes by the name of the
          `~.TwoBodyDecay.parent` `~qrules.particle.Particle`.
        - `.TwoBodyDecay` or `tuple` of a `~qrules.transition.StateTransition`
          with a node ID: set dynamics for one specific transition node.
        """
        raise NotImplementedError(
            "Cannot set dynamics builder for selection type"
            f" {type(selection).__name__}"
        )

    @assign.register(TwoBodyDecay)
    def _(
        self, decay: TwoBodyDecay, builder: ResonanceDynamicsBuilder
    ) -> None:
        self.__choices[decay] = builder

    @assign.register(tuple)
    def _(
        self,
        transition_node: Tuple[StateTransition, int],
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
            logging.warning(
                f'Model contains no resonance with name "{particle_name}"'
            )

    @assign.register(Particle)
    def _(self, particle: Particle, builder: ResonanceDynamicsBuilder) -> None:
        return self.assign(particle.name, builder)

    def __getitem__(
        self, __k: Union[TwoBodyDecay, Tuple[StateTransition, int]]
    ) -> ResonanceDynamicsBuilder:
        __k = TwoBodyDecay.create(__k)
        return self.__choices[__k]

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
        stable_final_state_ids: Put the invariant of the initial state
            (:math:`m_{012\dots}`) under `.HelicityModel.parameter_defaults`
            (with a *scalar* suggested value) instead of
            `~.HelicityModel.kinematic_variables`. This is useful if
            four-momenta were generated with or kinematically fit to a specific
            initial state energy.

            .. seealso:: :ref:`usage/amplitude:Scalar masses`
    """

    def __init__(
        self,
        reaction: ReactionInfo,
        stable_final_state_ids: Optional[Iterable[int]] = None,
        scalar_initial_state_mass: bool = False,
    ) -> None:
        if len(reaction.transitions) < 1:
            raise ValueError(
                f"At least one {StateTransition.__name__} required to"
                " genenerate an amplitude model!"
            )
        self.__reaction = reaction
        self._name_generator = HelicityAmplitudeNameGenerator(reaction)
        self.__ingredients = _HelicityModelIngredients()
        self.__dynamics_choices = DynamicsSelector(reaction)
        self.__adapter = HelicityAdapter(reaction)
        self.align_spin: Optional[bool] = None
        """(De)activate :doc:`spin alignment </usage/helicity/spin-alignment>`."""
        self.stable_final_state_ids = stable_final_state_ids  # type: ignore[assignment]
        self.scalar_initial_state_mass = scalar_initial_state_mass  # type: ignore[assignment]

    @property
    def adapter(self) -> HelicityAdapter:
        """Converter for computing kinematic variables from four-momenta."""
        return self.__adapter

    @property
    def dynamics_choices(self) -> DynamicsSelector:
        return self.__dynamics_choices

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

    @property
    def scalar_initial_state_mass(self) -> bool:
        """Add initial state mass as scalar value to `.parameter_defaults`.

        .. seealso:: :ref:`usage/amplitude:Scalar masses`
        """
        return self.__scalar_initial_state_mass

    @scalar_initial_state_mass.setter
    def scalar_initial_state_mass(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError
        self.__scalar_initial_state_mass = value

    def set_dynamics(
        self, particle_name: str, dynamics_builder: ResonanceDynamicsBuilder
    ) -> None:
        self.__dynamics_choices.assign(particle_name, dynamics_builder)

    def formulate(self) -> HelicityModel:
        self.__ingredients.reset()
        main_intensity = self.__formulate_top_expression()
        _str_kinematic_variables = self.__adapter.create_expressions(
            generate_wigner_angles=self.__is_align_spin
        )
        kinematic_variables = {
            sp.Symbol(var_name, real=True): expr
            for var_name, expr in _str_kinematic_variables.items()
        }
        if self.stable_final_state_ids is not None:
            for state_id in self.stable_final_state_ids:
                symbol = sp.Symbol(f"m_{state_id}", real=True)
                particle = self.__reaction.final_state[state_id]
                self.__ingredients.parameter_defaults[symbol] = particle.mass
                del kinematic_variables[symbol]
        if self.scalar_initial_state_mass:
            subscript = "".join(map(str, sorted(self.__reaction.final_state)))
            symbol = sp.Symbol(f"m_{subscript}", real=True)
            particle = self.__reaction.initial_state[-1]
            self.__ingredients.parameter_defaults[symbol] = particle.mass
            del kinematic_variables[symbol]

        return HelicityModel(
            intensity=main_intensity,
            amplitudes=self.__ingredients.amplitudes,
            parameter_defaults=self.__ingredients.parameter_defaults,
            kinematic_variables=kinematic_variables,
            components=self.__ingredients.components,
            reaction_info=self.__reaction,
        )

    def __formulate_top_expression(self) -> PoolSum:
        # pylint: disable=too-many-locals
        outer_state_ids = _get_outer_state_ids(self.__reaction)
        spin_projections: DefaultDict[
            sp.Symbol, Set[sp.Rational]
        ] = collections.defaultdict(set)
        spin_groups = group_by_spin_projection(self.__reaction.transitions)
        for group in spin_groups:
            self.__register_amplitudes(group)
            for transition in group:
                for i in outer_state_ids:
                    state = transition.states[i]
                    symbol = _create_spin_projection_symbol(i)
                    value = sp.Rational(state.spin_projection)
                    spin_projections[symbol].add(value)

        topology_groups = group_by_topology(self.__reaction.transitions)
        if self.__is_align_spin:
            amplitude = self.__formulate_aligned_amplitude(topology_groups)
        else:
            indices = list(spin_projections)
            amplitude = sum(
                _create_amplitude_base(topology)[indices]
                for topology in topology_groups
            )
        return PoolSum(abs(amplitude) ** 2, *spin_projections.items())

    def __formulate_aligned_amplitude(
        self, topology_groups: Dict[Topology, List[StateTransition]]
    ) -> sp.Expr:
        outer_state_ids = _get_outer_state_ids(self.__reaction)
        amplitude = sp.S.Zero
        for topology, transitions in topology_groups.items():
            base = _create_amplitude_base(topology)
            helicities = [
                _get_opposite_helicity_sign(topology, i)
                * _create_helicity_symbol(topology, i)
                for i in outer_state_ids
            ]
            amplitude_symbol = base[helicities]
            first_transition = transitions[0]
            alignment_sum = formulate_spin_alignment(first_transition)
            amplitude += PoolSum(
                alignment_sum.expression * amplitude_symbol,
                *alignment_sum.indices,
            )
        return amplitude

    @property
    def __is_align_spin(self) -> bool:
        if self.align_spin is None:
            topologies = collect_topologies(self.__reaction.transitions)
            return len(topologies) > 1
        return self.align_spin

    def __register_amplitudes(
        self, transition_group: List[StateTransition]
    ) -> None:
        transition_by_topology = group_by_topology(transition_group)
        expression = sum(
            self.__formulate_topology_amplitude(transitions)
            for transitions in transition_by_topology.values()
        )
        first_transition = transition_group[0]
        graph_group_label = generate_transition_label(first_transition)
        component_name = f"I_{{{graph_group_label}}}"
        self.__ingredients.components[component_name] = abs(expression) ** 2

    def __formulate_topology_amplitude(
        self, transitions: Sequence[StateTransition]
    ) -> sp.Expr:
        sequential_expressions: List[sp.Expr] = []
        for transition in transitions:
            sequential_graphs = (
                perform_external_edge_identical_particle_combinatorics(
                    transition.to_graph()
                )
            )
            for graph in sequential_graphs:
                first_transition = StateTransition.from_graph(graph)
                expression = self.__formulate_sequential_decay(
                    first_transition
                )
                sequential_expressions.append(expression)

        first_transition = transitions[0]
        symbol = _create_amplitude_symbol(first_transition)
        expression = sum(sequential_expressions)
        self.__ingredients.amplitudes[symbol] = expression
        return expression

    def __formulate_sequential_decay(
        self, transition: StateTransition
    ) -> sp.Expr:
        partial_decays: List[sp.Expr] = [
            self._formulate_partial_decay(transition, node_id)
            for node_id in transition.topology.nodes
        ]
        sequential_amplitudes = reduce(operator.mul, partial_decays)

        coefficient = self.__generate_amplitude_coefficient(transition)
        prefactor = self.__generate_amplitude_prefactor(transition)
        expression = coefficient * sequential_amplitudes
        if prefactor is not None:
            expression = prefactor * expression
        subscript = self._name_generator.generate_amplitude_name(transition)
        self.__ingredients.components[f"A_{{{subscript}}}"] = expression
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
            if par in self.__ingredients.parameter_defaults:
                previous_value = self.__ingredients.parameter_defaults[par]
                if value != previous_value:
                    logging.warning(
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

        Generally, each partial amplitude of a sequential amplitude transition
        should check itself if it or a parity partner is already defined. If so
        a coupled coefficient is introduced.
        """
        suffix = self._name_generator.generate_sequential_amplitude_suffix(
            transition
        )
        symbol = sp.Symbol(f"C_{{{suffix}}}")
        value = complex(1, 0)
        self.__ingredients.parameter_defaults[symbol] = value
        return symbol

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


def _create_amplitude_symbol(transition: StateTransition) -> sp.Indexed:
    outer_state_ids = _get_outer_state_ids(transition)
    helicities = tuple(
        sp.Rational(transition.states[i].spin_projection)
        for i in outer_state_ids
    )
    base = _create_amplitude_base(transition.topology)
    return base[helicities]


def _get_opposite_helicity_sign(
    topology: Topology, state_id: int
) -> Literal[-1, 1]:
    if state_id != -1 and is_opposite_helicity_state(topology, state_id):
        return -1
    return 1


def _create_amplitude_base(topology: Topology) -> sp.IndexedBase:
    superscript = get_topology_identifier(topology)
    return sp.IndexedBase(f"A^{superscript}", complex=True)


def _create_helicity_symbol(
    topology: Topology, state_id: int, root: str = "lambda"
) -> sp.Symbol:
    if state_id == -1:  # initial state
        name = "m_A"
    else:
        suffix = get_helicity_suffix(topology, state_id)
        name = f"{root}{suffix}"
    return sp.Symbol(name, rational=True)


def _create_spin_projection_symbol(state_id: int) -> sp.Symbol:
    if state_id == -1:  # initial state
        suffix = "_A"
    else:
        suffix = str(state_id)
    return sp.Symbol(f"m{suffix}", rational=True)


@singledispatch
def _get_outer_state_ids(
    obj: Union[ReactionInfo, StateTransition]
) -> List[int]:
    raise NotImplementedError(
        f"Cannot get outer state IDs from a {type(obj).__name__}"
    )


@_get_outer_state_ids.register(StateTransition)
def _(transition: StateTransition) -> List[int]:
    outer_state_ids = list(transition.initial_states)
    outer_state_ids += sorted(transition.final_states)
    return outer_state_ids


@_get_outer_state_ids.register(ReactionInfo)
def _(reaction: ReactionInfo) -> List[int]:
    return _get_outer_state_ids(reaction.transitions[0])


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

    .. seealso:: `HelicityAmplitudeBuilder` and :doc:`/usage/helicity/formalism`.
    """

    def __init__(self, reaction: ReactionInfo) -> None:
        super().__init__(reaction)
        self._name_generator = CanonicalAmplitudeNameGenerator(reaction)

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


def group_by_spin_projection(
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


def group_by_topology(
    transitions: Iterable[StateTransition],
) -> Dict[Topology, List[StateTransition]]:
    """Group state transitions by different `~qrules.topology.Topology`."""
    transition_groups = collections.defaultdict(list)
    for transition in transitions:
        transition_groups[transition.topology].append(transition)
    return dict(transition_groups)


def formulate_spin_alignment(
    transition: StateTransition,
) -> PoolSum:
    """Generate all Wigner-:math:`D` combinations for a spin alignment sum.

    Generate all Wigner-:math:`D` function combinations that appear in
    :cite:`marangottoHelicityAmplitudesGeneric2020`, Eq.(45), but for a generic
    multibody decay. Each element in the returned `list` is a `tuple` of
    Wigner-:math:`D` functions that appear in the summation, for a specific set
    of helicities were are summing over. To generate the full sum, make a
    multiply the Wigner-:math:`D` functions in each `tuple` and sum over all
    these products.
    """
    rotations = PoolSum(1)
    for rotated_state_id in transition.final_states:
        additional_rotations = formulate_rotation_chain(
            transition, rotated_state_id
        )
        rotations = __multiply_pool_sums([rotations, additional_rotations])
    return rotations


__GREEK_INDEX_NAMES = ("lambda", "mu", "nu", "xi", "alpha", "beta", "gamma")


def formulate_rotation_chain(
    transition: StateTransition, rotated_state_id: int
) -> PoolSum:
    """Formulate the spin alignment sum for a specific chain.

    See Eq.(45) from :cite:`marangottoHelicityAmplitudesGeneric2020`. The chain
    consists of a series of helicity rotations (see
    :func:`formulate_helicity_rotation_chain`) plus a Wigner rotation (see
    :func:`.formulate_wigner_rotation`) in case there is more than one helicity
    rotation.
    """
    helicity_symbol = _create_spin_projection_symbol(rotated_state_id)
    helicity_rotations = formulate_helicity_rotation_chain(
        transition, rotated_state_id, helicity_symbol
    )
    if len(helicity_rotations.indices) == 1:
        return helicity_rotations
    idx_root = __GREEK_INDEX_NAMES[len(helicity_rotations.indices)]
    idx_suffix = get_helicity_suffix(transition.topology, rotated_state_id)
    wigner_rotation = formulate_wigner_rotation(
        transition,
        rotated_state_id,
        helicity_symbol=helicity_symbol,
        m_prime=sp.Symbol(f"{idx_root}{idx_suffix}", rational=True),
    )
    return __multiply_pool_sums([helicity_rotations, wigner_rotation])


def formulate_helicity_rotation_chain(
    transition: StateTransition,
    rotated_state_id: int,
    helicity_symbol: sp.Symbol,
) -> PoolSum:
    """Formulate a Wigner-:math:`D` for each helicity rotation up some state.

    The helicity rotations are performed going through the decay
    `~qrules.topology.Topology` starting from the initial state up some
    :code:`rotated_state_id`. Each rotation operates on the spin state and is
    therefore formulated as a `~sympy.physics.quantum.spin.WignerD` function
    (see :func:`.formulate_helicity_rotation`). See
    {doc}`/usage/helicity/spin-alignment` for more info.
    """
    topology = transition.topology
    rotated_state = transition.states[rotated_state_id]
    spin_magnitude = rotated_state.particle.spin
    idx_root_counter = 0
    idx_suffix = get_helicity_suffix(transition.topology, rotated_state_id)

    def get_helicity_rotation(state_id: int) -> Generator[PoolSum, None, None]:
        parent_id = get_parent_id(topology, state_id)
        if parent_id is None:
            return
        # pylint: disable=stop-iteration-return
        nonlocal idx_root_counter
        idx_root = __GREEK_INDEX_NAMES[idx_root_counter]
        next_idx_root = __GREEK_INDEX_NAMES[idx_root_counter + 1]
        idx_root_counter += 1
        if is_opposite_helicity_state(topology, state_id):
            state_id = get_sibling_state_id(topology, state_id)
        phi_label, theta_theta = get_helicity_angle_label(topology, state_id)
        phi = sp.Symbol(phi_label, real=True)
        theta = sp.Symbol(theta_theta, real=True)
        no_zero_spin = transition.states[rotated_state_id].particle.mass == 0.0
        yield formulate_helicity_rotation(
            spin_magnitude,
            spin_projection=sp.Symbol(
                f"{next_idx_root}{idx_suffix}", rational=True
            ),
            m_prime=sp.Symbol(f"{idx_root}{idx_suffix}", rational=True),
            alpha=phi,
            beta=theta,
            gamma=0,
            no_zero_spin=no_zero_spin,
        )
        yield from get_helicity_rotation(parent_id)

    rotations = get_helicity_rotation(rotated_state_id)
    summation = __multiply_pool_sums(list(rotations))
    if len(summation.indices) == 1:
        idx_root = __GREEK_INDEX_NAMES[idx_root_counter]
        dangling_idx = sp.Symbol(f"{idx_root}{idx_suffix}", rational=True)
        return summation.subs(dangling_idx, helicity_symbol)
    return summation


def __multiply_pool_sums(sum_expressions: Sequence[PoolSum]) -> PoolSum:
    if len(sum_expressions) == 0:
        raise ValueError(f"Product needs at least one {PoolSum.__name__}")
    product = sp.Mul(*[pool_sum.expression for pool_sum in sum_expressions])
    combined_indices = []
    for pool_sum in sum_expressions:
        combined_indices.extend(pool_sum.indices)
    return PoolSum(product, *combined_indices)


def formulate_wigner_rotation(
    transition: StateTransition,
    rotated_state_id: int,
    helicity_symbol: sp.Symbol,
    m_prime: sp.Symbol,
) -> PoolSum:
    """Formulate the spin rotation matrices for a Wigner rotation.

    A **Wigner rotation** is the 'average' rotation that results form a chain
    of Lorentz boosts to a new reference frame with regard to a direct boost.
    See :cite:`marangottoHelicityAmplitudesGeneric2020`, p.6, especially
    Eq.(36).

    Args:
        transition: The `~qrules.transition.StateTransition` in which you
            want to rotate one of the spin states.
        rotated_state_id: The state ID of a spin `~qrules.transition.State`
            that you want to rotate.
        helicity_symbol: Optional `~sympy.core.symbol.Symbol` for :math:`m` in
            :math:`D^s_{mm'}. Falls back to the value of
            `~qrules.transition.State.spin_projection` embedded in the provided
            :code:`transition`.
        m_prime: The summation symbol :math:`m'` that should be used when
            summing over the Wigner-:math:`D` functions for this rotation.
    """
    state = transition.states[rotated_state_id]
    no_zero_spin = state.particle.mass == 0.0
    suffix = get_helicity_suffix(transition.topology, rotated_state_id)
    if helicity_symbol is None:
        spin_projection = state.spin_projection
    else:
        spin_projection = helicity_symbol
    return formulate_helicity_rotation(
        spin_magnitude=state.particle.spin,
        spin_projection=spin_projection,
        m_prime=m_prime,
        alpha=sp.Symbol(f"alpha{suffix}", real=True),
        beta=sp.Symbol(f"beta{suffix}", real=True),
        gamma=sp.Symbol(f"gamma{suffix}", real=True),
        no_zero_spin=no_zero_spin,
    )


def formulate_helicity_rotation(
    spin_magnitude: Union[float, sp.Symbol],
    spin_projection: Union[float, sp.Symbol],
    m_prime: sp.Symbol,
    alpha: sp.Symbol,
    beta: sp.Symbol,
    gamma: sp.Symbol,
    no_zero_spin: bool = False,
) -> PoolSum:
    r"""Formulate action of an Euler rotation on a spin state.

    When rotation a spin state :math:`\left|s,m\right\rangle` over `Euler
    angles <https://en.wikipedia.org/wiki/Euler_angles>`_
    :math:`\alpha,\beta,\gamma`, the new state can be expressed in terms of
    other spin states :math:`\left|s,m'\right\rangle` with the help of
    Wigner-:math:`D` expansion coefficients:

    .. math::
        :label: formulate_helicity_rotation

        R(\alpha,\beta,\gamma)\left|s,m\right\rangle
        = \sum^s_{m'=-s} D^s_{m',m}\left(\alpha,\beta,\gamma\right)
        \left|s,m'\right\rangle

    See :cite:`marangottoHelicityAmplitudesGeneric2020`, Eq.(B.5).

    This function gives the summation over these Wigner-:math:`D` functions and
    can be used for spin alignment following
    :cite:`marangottoHelicityAmplitudesGeneric2020`, Eq.(45).

    Args:
        spin_magnitude: Spin magnitude :math:`s` of spin state that is being
            rotated.
        spin_projection: Spin projection component :math:`m` of the spin state
            that is being rotated.
        m_prime: A index `~sympy.core.symbol.Symbol` or
            `~sympy.core.symbol.Dummy` that represents :math:`m'` helicities in
            Eq. :eq:`formulate_helicity_rotation`.

        alpha: First Euler angle.
        beta: Second Euler angle.
        gamma: Third Euler angle.
        no_zero_spin: Skip value :code:`0.0` in the generated spin projection
            range. Useful for massless particles.

    Example
    -------
    >>> a, b, c, i = sp.symbols("a b c i")
    >>> formulate_helicity_rotation(0, 0, i, a, b, c)
    PoolSum(WignerD(0, 0, i, a, b, c), (i, (0,)))
    >>> formulate_helicity_rotation(1/2, -1/2, i, a, b, c)
    PoolSum(WignerD(1/2, -1/2, i, a, b, c), (i, (-1/2, 1/2)))
    """
    from sympy.physics.quantum.spin import Rotation as Wigner

    helicities = map(
        sp.Rational, _create_spin_range(spin_magnitude, no_zero_spin)
    )
    return PoolSum(
        Wigner.D(
            j=__rationalize(spin_magnitude),
            m=__rationalize(spin_projection),
            mp=m_prime,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        ),
        (m_prime, list(helicities)),
    )


@overload
def __rationalize(value: float) -> sp.Rational:
    ...


@overload
def __rationalize(value: sp.Symbol) -> sp.Symbol:
    ...


def __rationalize(value):  # type:ignore[no-untyped-def]
    if isinstance(value, sp.Symbol):
        return value
    return sp.Rational(value)


def _create_spin_range(
    spin_magnitude: float, no_zero_spin: bool = False
) -> List[float]:
    """Create a list of allowed spin projections.

    >>> _create_spin_range(0)
    [0.0]
    >>> _create_spin_range(0.5)
    [-0.5, 0.5]
    >>> _create_spin_range(1)
    [-1.0, 0.0, 1.0]
    >>> _create_spin_range(1, no_zero_spin=True)
    [-1.0, 1.0]
    >>> projections = _create_spin_range(5)
    >>> list(map(int, projections))
    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    """
    spin_projections = []
    projection = Decimal(-spin_magnitude)
    while projection <= spin_magnitude:
        spin_projections.append(float(projection))
        projection += 1
    if no_zero_spin and len(spin_projections) > 1:
        spin_projections.remove(0.0)
    return spin_projections


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
    angular_momentum: Optional[int] = decay.interaction.l_magnitude
    if angular_momentum is None:
        if decay.parent.particle.spin.is_integer():
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
