# pylint: disable=too-many-lines

"""Functions to solve a particle reaction problem.

This module is responsible for solving a particle reaction problem stated by
a `.StateTransitionGraph` and corresponding `.GraphSettings`. The Solver classes
(e.g. :class:`.CSPSolver`) generate new quantum numbers (for example belonging
to an intermediate state) and use the implemented conservation rules of
:mod:`.conservation_rules`.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

from constraint import (
    BacktrackingSolver,
    Constraint,
    Problem,
    Unassigned,
    Variable,
)

from expertsystem.data import Parity, ParticleWithSpin, Spin
from expertsystem.nested_dicts import (
    InteractionQuantumNumberNames,
    Labels,
    ParticleDecayPropertyNames,
    ParticlePropertyNames,
    QNClassConverterMapping,
    QNNameClassMapping,
    StateQuantumNumberNames,
    _convert_edges_to_dict,
    edge_qn_to_enum,
)
from expertsystem.solving.conservation_rules import Rule
from expertsystem.state.properties import (
    get_interaction_property,
    get_particle_candidates_for_state,
    get_particle_property,
    initialize_graphs_with_particles,
)
from expertsystem.topology import StateTransitionGraph, Topology


class _GraphElementTypes(Enum):
    """Types of graph elements in the form of an enumerate."""

    node = auto()
    edge = auto()


class InteractionTypes(Enum):
    """Types of interactions in the form of an enumerate."""

    Strong = auto()
    EM = auto()
    Weak = auto()


@dataclass
class EdgeSettings:
    """Solver settings for a specific edge of a graph."""

    conservation_rules: Set[Rule] = field(default_factory=set)
    rule_priorities: Dict[Type[Rule], int] = field(default_factory=dict)
    qn_domains: Dict[Any, Any] = field(default_factory=dict)


@dataclass
class NodeSettings:
    """Container class for the interaction settings.

    This class can be assigned to each node of a state transition graph. Hence,
    these settings contain the complete configuration information which is
    required for the solution finding, e.g:

      - set of conservation rules
      - mapping of rules to priorities (optional)
      - mapping of quantum numbers to their domains
      - strength scale parameter (higher value means stronger force)
    """

    conservation_rules: Set[Rule] = field(default_factory=set)
    rule_priorities: Dict[Type[Rule], int] = field(default_factory=dict)
    qn_domains: Dict[Any, Any] = field(default_factory=dict)
    interaction_strength: float = 1.0


@dataclass
class GraphSettings:
    edge_settings: Dict[int, EdgeSettings]
    node_settings: Dict[int, NodeSettings]


def create_interaction_node_settings(
    graph: Topology, interaction_settings: NodeSettings
) -> Dict[int, NodeSettings]:
    return {node_id: interaction_settings for node_id in graph.nodes}


class Result:
    def __init__(
        self,
        solutions: List[StateTransitionGraph[dict]],
        not_executed_rules: Optional[Dict[int, Set[Rule]]] = None,
        violated_rules: Optional[Dict[int, Set[Tuple[Rule]]]] = None,
    ) -> None:
        if solutions and violated_rules:
            raise ValueError(
                "Invalid Result! Found solutions, but also violated rules."
            )
        self.__solutions = solutions
        self.__not_executed_rules: Dict[int, Set[Rule]] = defaultdict(set)
        if not_executed_rules is not None:
            self.__not_executed_rules = not_executed_rules

        # a tuple of rules defines a group which together violates all possible
        # combinations that were processed
        self.__violated_rules: Dict[int, Set[Tuple[Rule]]] = defaultdict(set)
        if violated_rules is not None:
            self.__violated_rules = violated_rules

    @property
    def solutions(self) -> List[StateTransitionGraph[dict]]:
        return self.__solutions

    @property
    def not_executed_rules(self) -> Dict[int, Set[Rule]]:
        return self.__not_executed_rules

    @property
    def violated_rules(self) -> Dict[int, Set[Tuple[Rule]]]:
        return self.__violated_rules

    def extend(
        self, other_result: "Result", intersect_violations: bool = False
    ) -> None:
        if self.solutions or other_result.solutions:
            self.__solutions.extend(other_result.solutions)
            self.__not_executed_rules.clear()
            self.__violated_rules.clear()
        else:
            for key, rules in other_result.not_executed_rules.items():
                self.__not_executed_rules[key].update(rules)

            for key, rules2 in other_result.violated_rules.items():
                if intersect_violations:
                    self.__violated_rules[key] &= rules2
                else:
                    self.__violated_rules[key].update(rules2)


class Solver(ABC):
    """Interface of a Solver."""

    @abstractmethod
    def find_solutions(
        self,
        graph: StateTransitionGraph[ParticleWithSpin],
        graph_settings: GraphSettings,
    ) -> Result:
        """Find solutions for the given input.

        It is expected that this function determines and returns all of the
        found solutions. In case no solutions are found a partial list of
        violated rules has to be given. This list of violated rules does not
        have to be complete.

        Args:
          graph: a `.StateTransitionGraph` which contains all of the known
            facts quantum numbers of the problem.
          edge_settings: mapping of edge id's to `EdgeSettings`, that
            assigns specific rules and variable domains to an edge of the graph.
          node_settings: mapping of node id's to `NodeSettings`, that
            assigns specific rules and variable domains to a node of the graph.

        Returns:
          Result: contains possible solutions, violated rules and not executed
          rules due to requirement issues.
        """


def _is_optional(class_field: Any) -> bool:
    if "__args__" in class_field.__dict__:
        if type(None) in class_field.__args__:
            return True
    return False


def _init_class(class_type: Type, props: Dict[Enum, Any]) -> object:
    return class_type(
        **{
            class_field.name: _extract_value(props, class_field.type)
            for class_field in fields(class_type)
            if not _is_optional(class_field.type)
            or edge_qn_to_enum[class_field.type.__args__[0].__name__] in props
        }
    )


def _extract_value(props: Dict[Enum, Any], obj_type: Any) -> Any:
    if _is_optional(obj_type):
        obj_type = obj_type.__args__[0]

    qn_name = obj_type.__name__
    value = props[edge_qn_to_enum[qn_name]]
    if "projection" in qn_name:
        value = value.projection
    elif "magnitude" in qn_name:
        value = value.magnitude

    if (
        "__supertype__" in obj_type.__dict__
        and obj_type.__supertype__ == Parity
    ):
        return obj_type.__supertype__(value)
    return obj_type(value)


def _check_arg_requirements(
    class_type: Type[Any], props: Dict[Enum, Any]
) -> bool:
    if "__dataclass_fields__" in class_type.__dict__:
        return all(
            [
                bool(edge_qn_to_enum[class_field.type.__name__] in props)
                for class_field in fields(class_type)
                if not _is_optional(class_field.type)
            ]
        )

    return edge_qn_to_enum[class_type.__name__] in props


def _check_requirements(
    rule: Rule,
    in_edge_props: List[Dict[Enum, Any]],
    out_edge_props: List[Dict[Enum, Any]],
    node_props: dict,
) -> bool:
    if not hasattr(rule.__class__.__call__, "__annotations__"):
        raise TypeError(
            f"missing type annotations for __call__ of rule {rule.__class__.__name__}"
        )
    arg_counter = 1
    rule_annotations = _remove_return_annotation(
        list(rule.__class__.__call__.__annotations__.values())
    )
    for arg_type, props in zip(
        rule_annotations,
        (in_edge_props, out_edge_props, node_props),
    ):
        if arg_counter == 3:
            if not _check_arg_requirements(arg_type, props):  # type: ignore
                return False
        else:
            if not all(
                [
                    _check_arg_requirements(arg_type.__args__[0], x)
                    for x in props
                ]
            ):
                return False
        arg_counter += 1

    return True


def _create_rule_edge_arg(
    input_type: Type[Any], edge_props: List[Dict[Enum, Any]]
) -> List[Any]:
    # pylint: disable=unidiomatic-typecheck
    if not isinstance(edge_props, (list, tuple)):
        raise TypeError("edge_props are incompatible...")
    if not (type(input_type) is list or type(input_type) is tuple):
        raise TypeError("input type is incompatible...")
    in_list_type = input_type[0]

    if "__dataclass_fields__" in in_list_type.__dict__:
        # its a composite type -> create new class type here
        return [_init_class(in_list_type, x) for x in edge_props]
    return [_extract_value(x, in_list_type) for x in edge_props]


def _create_rule_node_arg(
    input_type: Type[Any], node_props: Dict[Enum, Any]
) -> Any:
    # pylint: disable=unidiomatic-typecheck
    if isinstance(node_props, (list, tuple)):
        raise TypeError("node_props is incompatible...")
    if type(input_type) is list or type(input_type) is tuple:
        raise TypeError("input type is incompatible...")

    if "__dataclass_fields__" in input_type.__dict__:
        # its a composite type -> create new class type here
        return _init_class(input_type, node_props)
    return _extract_value(node_props, input_type)


def _create_rule_args(
    rule: Rule,
    in_edge_props: List[Dict[Enum, Any]],
    out_edge_props: List[Dict[Enum, Any]],
    node_props: Dict[Enum, Any],
) -> list:
    if not hasattr(rule.__class__.__call__, "__annotations__"):
        raise TypeError(
            f"missing type annotations for __call__ of rule {str(rule)}"
        )
    args = []
    arg_counter = 0
    rule_annotations = list(rule.__class__.__call__.__annotations__.values())
    rule_annotations = _remove_return_annotation(rule_annotations)

    ordered_props = (in_edge_props, out_edge_props, node_props)
    for arg_type in rule_annotations:
        if arg_counter == 2:
            args.append(
                _create_rule_node_arg(arg_type, ordered_props[arg_counter])  # type: ignore
            )
        else:
            args.append(
                _create_rule_edge_arg(
                    arg_type.__args__, ordered_props[arg_counter]  # type: ignore
                )
            )
        arg_counter += 1
    if arg_counter == 2:
        # the rule does not use the third argument, just add None
        args.append(None)

    return args


def _remove_return_annotation(
    rule_annotations: List[Type[Any]],
) -> List[Type[Any]]:
    # this assumes that all rules have also the return type defined
    return rule_annotations[:-1]


def _get_required_qn_names(rule: Rule) -> List[Enum]:
    if not hasattr(rule.__class__.__call__, "__annotations__"):
        raise TypeError(
            f"missing type annotations for __call__ of rule {rule.__class__.__name__}"
        )

    qn_set = set()

    rule_annotations = list(rule.__class__.__call__.__annotations__.values())
    rule_annotations = _remove_return_annotation(rule_annotations)

    for input_type in rule_annotations:
        class_type = input_type
        if "__origin__" in input_type.__dict__ and (
            input_type.__origin__ is list or input_type.__origin__ is tuple
        ):
            class_type = input_type.__args__[0]

        if "__dataclass_fields__" in class_type.__dict__:
            for class_field in fields(class_type):
                qn_set.add(
                    edge_qn_to_enum[
                        class_field.type.__args__[0].__name__
                        if _is_optional(class_field.type)
                        else class_field.type.__name__
                    ]
                )
        else:
            qn_set.add(edge_qn_to_enum[class_type.__name__])

    return list(qn_set)


class _VariableInfo:
    """Data container for variable information."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        graph_element_type: _GraphElementTypes,
        element_id: int,
        qn_name: Enum,
    ) -> None:
        self.graph_element_type = graph_element_type
        self.element_id = element_id
        self.qn_name = qn_name


def _decode_variable_name(variable_name: str, delimiter: str) -> _VariableInfo:
    """Decode the variable name.

    Also see `.encode_variable_name`.
    """
    split_name = variable_name.split(delimiter)
    if not len(split_name) == 3:
        raise ValueError(
            "The variable name does not follow the scheme: " + variable_name
        )
    qn_name: Optional[Enum] = None
    graph_element_type = None
    element_id = int(split_name[1])
    if split_name[0] in _GraphElementTypes.node.name:
        qn_name = InteractionQuantumNumberNames[split_name[2]]
        graph_element_type = _GraphElementTypes.node
    else:
        qn_name = StateQuantumNumberNames[split_name[2]]
        graph_element_type = _GraphElementTypes.edge

    return _VariableInfo(graph_element_type, element_id, qn_name)


def _encode_variable_name(variable_info: _VariableInfo, delimiter: str) -> str:
    """Encode variable name.

    The variable names are encoded as a concatenated string of the form graph
    element type + delimiter + element id + delimiter + qn name The variable of
    type :class:`.VariableInfo` and contains:

      - graph_element_type: is either "node" or "edge" (enum)
      - element_id: is the id of that node/edge (as it is defined in the graph)
      - qn_name: the quantum number name (enum)
    """
    if not isinstance(variable_info, _VariableInfo):
        raise TypeError("parameter variable_info must be of type VariableInfo")
    var_name = (
        variable_info.graph_element_type.name
        + delimiter
        + str(variable_info.element_id)
        + delimiter
        + variable_info.qn_name.name
    )
    return var_name


def validate_fully_initialized_graph(
    graph: StateTransitionGraph, rules_per_node: Dict[int, Set[Rule]]
) -> Result:
    logging.debug("validating graph...")

    def _create_node_variables(
        node_id: int, qn_list: list
    ) -> Dict[InteractionQuantumNumberNames, Union[Spin, float]]:
        """Create variables for the quantum numbers of the specified node."""
        variables = {}
        type_label = Labels.Type.name
        if node_id in graph.node_props:
            qns_label = Labels.QuantumNumber.name
            for qn_name in qn_list:
                converter = QNClassConverterMapping[
                    QNNameClassMapping[qn_name]
                ]
                found_prop = None
                for node_qn in graph.node_props[node_id][qns_label]:
                    if node_qn[type_label] == qn_name.name:
                        found_prop = node_qn
                        break
                if found_prop is not None:
                    value = converter.parse_from_dict(found_prop)
                    variables[qn_name] = value
        return variables

    def _create_edge_variables(
        edge_ids: Sequence[int],
        qn_list: List[
            Union[
                ParticleDecayPropertyNames,
                ParticlePropertyNames,
                StateQuantumNumberNames,
            ]
        ],
    ) -> List[dict]:
        """Create variables for the quantum numbers of the specified edges.

        Initial and final state edges just get a single domain value.
        Intermediate edges are initialized with the default domains of that
        quantum number.
        """
        variables = []
        for edge_id in edge_ids:
            if edge_id in graph.edge_props:
                edge_vars = {}
                edge_props = graph.edge_props[edge_id]
                for qn_name in qn_list:
                    value = get_particle_property(edge_props, qn_name)
                    if value is not None:
                        edge_vars[qn_name] = value
                variables.append(edge_vars)
        return variables

    def _prepare_qns(
        qn_names: Sequence[Enum], type_to_filter: Union[Type, Tuple[Type, ...]]
    ) -> List[Enum]:
        return [x for x in qn_names if isinstance(x, type_to_filter)]

    def _create_variable_containers(
        node_id: int, cons_law: Rule
    ) -> Tuple[List[dict], List[dict], dict]:
        in_edges = graph.get_edges_ingoing_to_node(node_id)
        out_edges = graph.get_edges_outgoing_from_node(node_id)

        qn_names = _get_required_qn_names(cons_law)
        qn_list = _prepare_qns(
            qn_names,
            (
                StateQuantumNumberNames,
                ParticlePropertyNames,
                ParticleDecayPropertyNames,
            ),
        )
        in_edges_vars = _create_edge_variables(in_edges, qn_list)  # type: ignore
        out_edges_vars = _create_edge_variables(out_edges, qn_list)  # type: ignore

        node_vars = _create_node_variables(
            node_id,
            _prepare_qns(qn_names, InteractionQuantumNumberNames),
        )

        return (in_edges_vars, out_edges_vars, node_vars)

    node_violated_rules: Dict[int, Set[Tuple[Rule]]] = defaultdict(set)
    node_not_executed_rules: Dict[int, Set[Rule]] = defaultdict(set)
    for node_id, rules in rules_per_node.items():
        for rule in rules:
            # get the needed qns for this conservation law
            # for all edges and the node
            var_containers = _create_variable_containers(node_id, rule)
            # check the requirements
            if _check_requirements(
                rule,
                var_containers[0],
                var_containers[1],
                var_containers[2],
            ):
                # and run the rule check
                if not rule(
                    *_create_rule_args(
                        rule,
                        var_containers[0],
                        var_containers[1],
                        var_containers[2],
                    )
                ):
                    node_violated_rules[node_id].add((rule,))
            else:
                node_not_executed_rules[node_id].add(rule)
    if node_violated_rules or node_not_executed_rules:
        return Result([], node_not_executed_rules, node_violated_rules)
    return Result([graph])


class CSPSolver(Solver):
    """Solver reducing the task to a Constraint Satisfaction Problem.

    Solving this done with the python-constraint module.

    The variables are the quantum numbers of particles/edges, but also some
    composite quantum numbers which are attributed to the interaction nodes
    (such as angular momentum :math:`L`). The conservation rules serve as the
    constraints and a special wrapper class serves as an adapter.
    """

    def __init__(self, allowed_intermediate_particles: List[dict]):
        self.__graph = StateTransitionGraph[dict]()
        self.__variable_set: Set[str] = set()
        self.__constraints: Dict[
            int, Set[_ConservationRuleConstraintWrapper]
        ] = defaultdict(set)
        self.__non_executable_constraints: Dict[
            int, Set[_ConservationRuleConstraintWrapper]
        ] = defaultdict(set)
        self.__problem = Problem(BacktrackingSolver(True))
        self.__particle_variable_delimiter = "-*-"
        self.__allowed_intermediate_particles = allowed_intermediate_particles

    def find_solutions(
        self,
        graph: StateTransitionGraph[ParticleWithSpin],
        graph_settings: GraphSettings,
    ) -> Result:
        _convert_edges_to_dict([graph])
        dict_graph: StateTransitionGraph[dict] = graph  # type: ignore
        csp_result = self.__run_csp(dict_graph, graph_settings)

        # insert particle instances
        if self.__constraints:
            full_particle_graphs = initialize_graphs_with_particles(
                csp_result.solutions, self.__allowed_intermediate_particles
            )
        else:
            full_particle_graphs = [dict_graph]

        if full_particle_graphs and csp_result.not_executed_rules:
            # rerun solver on these graphs using not executed rules
            # and combine results
            result = Result([])
            for full_particle_graph in full_particle_graphs:
                result.extend(
                    validate_fully_initialized_graph(
                        full_particle_graph, csp_result.not_executed_rules
                    )
                )
            return result

        return Result(
            full_particle_graphs,
            csp_result.not_executed_rules,
            csp_result.violated_rules,
        )

    def __run_csp(
        self, graph: StateTransitionGraph[dict], graph_settings: GraphSettings
    ) -> Result:
        self.__graph = graph
        self.__initialize_constraints(graph_settings)
        solutions = self.__problem.getSolutions()

        solutions = self.__apply_solutions_to_graph(solutions)

        not_executed_rules: Dict[int, Set[Rule]] = {
            node_id: set(x.rule for x in constraints)
            for node_id, constraints in self.__non_executable_constraints.items()
        }
        not_satisfied_rules: Dict[int, Set[Tuple[Rule]]] = defaultdict(set)
        for node_id, constraints in self.__constraints.items():
            for constraint in constraints:
                if (
                    constraint.conditions_never_met
                    or sum(constraint.scenario_results) == 0
                ):
                    not_executed_rules[node_id].add(constraint.rule)
                if (
                    sum(constraint.scenario_results) > 0
                    and constraint.scenario_results[1] == 0
                ):
                    not_satisfied_rules[node_id].add((constraint.rule,))

        return Result(solutions, not_executed_rules, not_satisfied_rules)

    def __clear(self) -> None:
        self.__variable_set = set()
        self.__constraints = defaultdict(set)
        self.__problem = Problem(BacktrackingSolver(True))

    def __initialize_constraints(
        self, graph_settings: GraphSettings
    ) -> None:  # pylint: disable=too-many-locals
        """Initialize all of the constraints for this graph.

        For each interaction node a set of independent constraints/conservation
        laws are created. For each conservation law a new CSP wrapper is
        created. This wrapper needs all of the qn numbers/variables which enter
        or exit the node and play a role for this conservation law. Hence
        variables are also created within this method.
        """
        self.__clear()

        def get_rules_by_priority(
            node_settings: NodeSettings,
        ) -> List[Rule]:
            # first add priorities to the entries
            priority_list = [
                (x, node_settings.rule_priorities[type(x)])
                if type(x) in node_settings.rule_priorities
                else (x, 1)
                for x in node_settings.conservation_rules
            ]
            # then sort according to priority
            sorted_list = sorted(
                priority_list, key=lambda x: x[1], reverse=True
            )
            # and strip away the priorities again
            return [x[0] for x in sorted_list]

        for node_id in self.__graph.nodes:
            # currently we only have rules related to graph nodes
            # later on rules that are directly connected to edge can also be
            # defined (GellmannNishijimaRule can be changed to that)
            for cons_law in get_rules_by_priority(
                graph_settings.node_settings[node_id]
            ):
                variable_mapping: Dict[str, Any] = {}
                # from cons law and graph determine needed var lists
                qn_names = _get_required_qn_names(cons_law)

                in_edges = self.__graph.get_edges_ingoing_to_node(node_id)
                in_edge_vars = self.__create_edge_variables(
                    in_edges, qn_names, graph_settings.edge_settings
                )
                variable_mapping["ingoing"] = in_edge_vars[0]
                variable_mapping["ingoing-fixed"] = in_edge_vars[1]
                var_list = list(variable_mapping["ingoing"])

                out_edges = self.__graph.get_edges_outgoing_from_node(node_id)
                out_edge_vars = self.__create_edge_variables(
                    out_edges, qn_names, graph_settings.edge_settings
                )
                variable_mapping["outgoing"] = out_edge_vars[0]
                variable_mapping["outgoing-fixed"] = out_edge_vars[1]
                var_list.extend(list(variable_mapping["outgoing"]))

                # now create variables for node/interaction qns
                int_node_vars = self.__create_node_variables(
                    node_id, qn_names, graph_settings.node_settings
                )
                variable_mapping["interaction"] = int_node_vars[0]
                variable_mapping["interaction-fixed"] = int_node_vars[1]
                var_list.extend(list(variable_mapping["interaction"]))

                constraint = _ConservationRuleConstraintWrapper(
                    cons_law,
                    variable_mapping,
                    self.__particle_variable_delimiter,
                )
                if var_list:
                    self.__constraints[node_id].add(constraint)
                    self.__problem.addConstraint(constraint, var_list)
                else:
                    self.__non_executable_constraints[node_id].add(constraint)

    def __create_node_variables(
        self,
        node_id: int,
        qn_names: Sequence[Enum],
        node_settings: Dict[int, NodeSettings],
    ) -> Tuple[
        Set[str],
        Set[Tuple[InteractionQuantumNumberNames, Union[Spin, float]]],
    ]:
        """Create variables for the quantum numbers of the specified node.

        If a quantum number is already defined for a node, then a fixed
        variable is created, which cannot be changed by the csp solver.
        Otherwise the node is initialized with the specified domain of that
        quantum number.
        """
        variables: Tuple[
            Set[str],
            Set[Tuple[InteractionQuantumNumberNames, Union[Spin, float]]],
        ] = (
            set(),
            set(),
        )

        if node_id in self.__graph.node_props:
            node_props = self.__graph.node_props[node_id]
            for interaction_qn in [
                x
                for x in qn_names
                if isinstance(x, InteractionQuantumNumberNames)
            ]:
                value = get_interaction_property(node_props, interaction_qn)
                if value is not None:
                    variables[1].add((interaction_qn, value))
        else:
            for qn_name in qn_names:
                var_info = _VariableInfo(
                    _GraphElementTypes.node, node_id, qn_name
                )
                if qn_name in node_settings[node_id].qn_domains:
                    qn_domain = node_settings[node_id].qn_domains[qn_name]
                    key = self.__add_variable(var_info, qn_domain)
                    variables[0].add(key)
        return variables

    def __create_edge_variables(
        self,
        edge_ids: Sequence[int],
        qn_names: Sequence[Enum],
        edge_settings: Dict[int, EdgeSettings],
    ) -> Tuple[Set[str], Dict[int, List[Tuple[Any, Any]]]]:
        """Create variables for the quantum numbers of the specified edges.

        If a quantum number is already defined for an edge, then a fixed
        variable is created, which cannot be changed by the csp solver. This is
        the case for initial and final state edges. Otherwise the edges are
        initialized with the specified domains of that quantum number.
        """
        variables: Tuple[Set[str], Dict[int, List[Tuple[Any, Any]]]] = (
            set(),
            dict(),
        )
        for edge_id in edge_ids:
            variables[1][edge_id] = []
            if edge_id in self.__graph.edge_props:
                edge_props = self.__graph.edge_props[edge_id]
                for qn_name in [
                    x
                    for x in qn_names
                    if not isinstance(x, InteractionQuantumNumberNames)
                ]:
                    value = get_particle_property(edge_props, qn_name)  # type: ignore
                    if value is not None:
                        variables[1][edge_id].append((qn_name, value))
            else:
                for qn_name in qn_names:
                    var_info = _VariableInfo(
                        _GraphElementTypes.edge, edge_id, qn_name
                    )
                    if qn_name in edge_settings[edge_id].qn_domains:
                        qn_domain = edge_settings[edge_id].qn_domains[qn_name]
                        key = self.__add_variable(var_info, qn_domain)
                        variables[0].add(key)
        return variables

    def __add_variable(
        self, var_info: _VariableInfo, domain: List[Any]
    ) -> str:
        key = _encode_variable_name(
            var_info, self.__particle_variable_delimiter
        )
        if key not in self.__variable_set:
            self.__variable_set.add(key)
            self.__problem.addVariable(key, domain)
        return key

    def __apply_solutions_to_graph(
        self, solutions: List[Dict[str, Any]]
    ) -> List[StateTransitionGraph[dict]]:
        # pylint: disable=too-many-locals
        """Apply the CSP solutions to the graph instance.

        In other words attach the solution quantum numbers as properties to the
        edges. Also the solutions are filtered using the allowed intermediate
        particle list, to avoid large memory consumption.

        Args:
            solutions: list of solutions of the csp solver

        Returns:
            solution graphs ([:class:`.StateTransitionGraph`])
        """

        def add_qn_to_graph_element(
            graph: StateTransitionGraph[dict],
            var_info: _VariableInfo,
            value: Any,
        ) -> None:
            if value is None:
                return
            qns_label = Labels.QuantumNumber.name

            element_id = var_info.element_id
            qn_name = var_info.qn_name
            graph_prop_dict = graph.edge_props
            if var_info.graph_element_type is _GraphElementTypes.node:
                graph_prop_dict = graph.node_props

            converter = QNClassConverterMapping[QNNameClassMapping[qn_name]]

            if element_id not in graph_prop_dict:
                graph_prop_dict[element_id] = {qns_label: []}

            graph_prop_dict[element_id][qns_label].append(
                converter.convert_to_dict(qn_name, value)
            )

        solution_graphs = []
        initial_edges = self.__graph.get_initial_state_edges()
        final_edges = self.__graph.get_final_state_edges()

        found_jps = set()

        for solution in solutions:
            graph_copy = deepcopy(self.__graph)
            for var_name, value in solution.items():
                var_info = _decode_variable_name(
                    var_name, self.__particle_variable_delimiter
                )
                ele_id = var_info.element_id

                if var_info.graph_element_type is _GraphElementTypes.edge:
                    if ele_id in initial_edges or ele_id in final_edges:
                        # skip if its an initial or final state edge
                        continue

                add_qn_to_graph_element(graph_copy, var_info, value)

            solution_valid = True
            if self.__allowed_intermediate_particles:
                for int_edge_id in graph_copy.get_intermediate_state_edges():
                    # for documentation in case of failure
                    spin = get_particle_property(
                        graph_copy.edge_props[int_edge_id],
                        StateQuantumNumberNames.Spin,
                    )
                    parity = get_particle_property(
                        graph_copy.edge_props[int_edge_id],
                        StateQuantumNumberNames.Parity,
                    )
                    found_jps.add(
                        str(spin.magnitude)  # type: ignore
                        + ("-" if parity in (-1, -1.0) else "+")
                    )
                    # now do actual candidate finding
                    candidates = get_particle_candidates_for_state(
                        graph_copy.edge_props[int_edge_id],
                        self.__allowed_intermediate_particles,
                    )
                    if not candidates:
                        solution_valid = False
                        break
            if solution_valid:
                solution_graphs.append(graph_copy)

        if solutions and not solution_graphs:
            logging.warning(
                "No intermediate state particles match the found %d solutions!",
                len(solutions),
            )
            logging.warning("solution inter. state J^P: %s", str(found_jps))
        return solution_graphs


class _ConservationRuleConstraintWrapper(Constraint):
    """Wrapper class of the python-constraint Constraint class.

    This allows a customized definition of conservation rules, and hence a
    cleaner user interface.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, rule: Rule, variable_mapping: Dict[str, Any], name_delimiter: str
    ) -> None:
        if not isinstance(rule, Rule):
            raise TypeError("rule has to be of type Rule!")
        self.rule = rule
        self.in_variable_set = variable_mapping["ingoing"]
        self.fixed_in_variables = variable_mapping["ingoing-fixed"]
        self.out_variable_set = variable_mapping["outgoing"]
        self.fixed_out_variables = variable_mapping["outgoing-fixed"]
        self.interaction_variable_set = variable_mapping["interaction"]
        self.fixed_interaction_variable_set = variable_mapping[
            "interaction-fixed"
        ]
        self.name_delimiter = name_delimiter
        self.part_in: List[Dict[Enum, Any]] = []
        self.part_out: List[Dict[Enum, Any]] = []
        self.interaction_qns: Dict[Enum, Any] = {}
        self.variable_name_decoding_map: Dict[str, Tuple[int, Enum]] = {}

        self.initialize_particle_lists()

        self.conditions_never_met = False
        self.scenario_results = [0, 0]

    def initialize_particle_lists(self) -> None:
        """Fill the name decoding map.

        Also initialize the in and out particle lists. The variable names
        follow the scheme edge_id(delimiter)qn_name. This method creates a dict
        linking the var name to a list that consists of the particle list index
        and the qn name.
        """
        self.initialize_particle_list(
            self.in_variable_set, self.fixed_in_variables, self.part_in
        )
        self.initialize_particle_list(
            self.out_variable_set, self.fixed_out_variables, self.part_out
        )
        # and now interaction node variables
        for var_name in self.interaction_variable_set:
            var_info = _decode_variable_name(var_name, self.name_delimiter)
            self.interaction_qns[var_info.qn_name] = {}
            self.variable_name_decoding_map[var_name] = (0, var_info.qn_name)
        for qn_name, value in self.fixed_interaction_variable_set:
            self.interaction_qns[qn_name] = value

    def initialize_particle_list(
        self,
        variable_set: Sequence[str],
        fixed_variables: Dict[int, list],
        list_to_init: List[dict],
    ) -> None:
        temp_var_dict: Dict[int, Any] = {}
        for var_name in variable_set:
            var_info = _decode_variable_name(var_name, self.name_delimiter)
            if var_info.element_id not in temp_var_dict:
                temp_var_dict[var_info.element_id] = {
                    "vars": {var_name: var_info.qn_name}
                }
            else:
                temp_var_dict[var_info.element_id]["vars"][
                    var_name
                ] = var_info.qn_name

        for edge_id, varlist in fixed_variables.items():
            if edge_id not in temp_var_dict:
                temp_var_dict[edge_id] = {"fixed-vars": varlist}
            else:
                if "fixed-vars" not in temp_var_dict[edge_id]:
                    temp_var_dict[edge_id]["fixed-vars"] = varlist

        for value in temp_var_dict.values():
            index = len(list_to_init)
            list_to_init.append({})
            if "vars" in value:
                for var_name, qn_name in value["vars"].items():
                    self.variable_name_decoding_map[var_name] = (
                        index,
                        qn_name,
                    )
            if "fixed-vars" in value:
                for item in value["fixed-vars"]:
                    list_to_init[-1][item[0]] = item[1]

    def __call__(
        self,
        variables: Set[str],
        domains: dict,
        assignments: dict,
        forwardcheck: bool = False,
        _unassigned: Variable = Unassigned,
    ) -> bool:
        """Perform the constraint checking.

        If the forwardcheck parameter is not false, besides telling if the
        constraint is currently broken or not, the constraint implementation
        may choose to hide values from the domains of unassigned variables to
        prevent them from being used, and thus prune the search space.

        Args:
            variables: Variables affected by that constraint, in the same order
                provided by the user.

            domains (dict): Dictionary mapping variables to their domains.

            assignments (dict): Dictionary mapping assigned variables to their
                current assumed value.

            forwardcheck (bool): Boolean value stating whether forward checking
                should be performed or not.

            _unassigned: Can be left empty

        Return:
            bool:
                Boolean value stating if this constraint is currently broken
                or not.
        """
        if self.conditions_never_met:
            return True
        params = [(x, assignments.get(x, _unassigned)) for x in variables]
        missing = [name for (name, val) in params if val is _unassigned]
        if missing:
            return True

        self.update_variable_lists(params)
        if not _check_requirements(
            self.rule, self.part_in, self.part_out, self.interaction_qns
        ):
            self.conditions_never_met = True
            return True

        passed = self.rule(
            *_create_rule_args(
                self.rule, self.part_in, self.part_out, self.interaction_qns
            )
        )

        # before returning gather statistics about the rule
        if passed:
            self.scenario_results[1] += 1
        else:
            self.scenario_results[0] += 1
        return passed

    def update_variable_lists(
        self, parameters: List[Tuple[str, float]]
    ) -> None:
        for [var_name, value] in parameters:
            (index, qn_name) = self.variable_name_decoding_map[var_name]
            if var_name in self.in_variable_set:
                self.part_in[index][qn_name] = value
            elif var_name in self.out_variable_set:
                self.part_out[index][qn_name] = value
            elif var_name in self.interaction_variable_set:
                self.interaction_qns[qn_name] = value
            else:
                raise ValueError(
                    "The variable with name "
                    + var_name
                    + "does not appear in the variable mapping!"
                )
