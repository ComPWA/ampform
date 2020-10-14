# pylint: disable=too-many-lines

"""Functions to solve a particle reaction problem.

This module is responsible for solving a particle reaction problem stated by a
`.StateTransitionGraph` and corresponding `.GraphSettings`. The `.Solver`
classes (e.g. :class:`.CSPSolver`) generate new quantum numbers (for example
belonging to an intermediate state) and validate the decay processes with the
rules formulated by the :mod:`.conservation_rules` module.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

import attr
from constraint import (
    BacktrackingSolver,
    Constraint,
    Problem,
    Unassigned,
    Variable,
)

from expertsystem.particle import Parity, Particle, ParticleCollection, Spin

from .conservation_rules import IsoSpinValidity, Rule
from .quantum_numbers import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumber,
    ParticleWithSpin,
)
from .topology import StateTransitionGraph

Scalar = Union[int, float]


class InteractionTypes(Enum):
    """Types of interactions in the form of an enumerate."""

    Strong = auto()
    EM = auto()
    Weak = auto()


@attr.s
class EdgeSettings:
    """Solver settings for a specific edge of a graph."""

    conservation_rules: Set[Rule] = attr.ib(factory=set)
    rule_priorities: Dict[Type[Rule], int] = attr.ib(factory=dict)
    qn_domains: Dict[Any, Any] = attr.ib(factory=dict)


@attr.s
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

    conservation_rules: Set[Rule] = attr.ib(factory=set)
    rule_priorities: Dict[Type[Rule], int] = attr.ib(factory=dict)
    qn_domains: Dict[Any, Any] = attr.ib(factory=dict)
    interaction_strength: float = 1.0


@attr.s
class GraphSettings:
    edge_settings: Dict[int, EdgeSettings] = attr.ib(factory=dict)
    node_settings: Dict[int, NodeSettings] = attr.ib(factory=dict)


class Result:
    def __init__(
        self,
        solutions: Optional[
            List[StateTransitionGraph[ParticleWithSpin]]
        ] = None,
        not_executed_rules: Optional[Dict[int, Set[Rule]]] = None,
        violated_rules: Optional[Dict[int, Set[Tuple[Rule]]]] = None,
    ) -> None:
        if solutions and violated_rules:
            raise ValueError(
                "Invalid Result! Found solutions, but also violated rules."
            )

        self.__solutions: List[StateTransitionGraph[ParticleWithSpin]] = list()
        if solutions is not None:
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
    def solutions(self) -> List[StateTransitionGraph[ParticleWithSpin]]:
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


@attr.s(frozen=True)
class _QuantumNumberSolution:
    node_quantum_numbers: Dict[
        int, Dict[Type[NodeQuantumNumber], Scalar]
    ] = attr.field(factory=lambda: defaultdict(dict))
    edge_quantum_numbers: Dict[
        int, Dict[Type[EdgeQuantumNumber], Scalar]
    ] = attr.field(factory=lambda: defaultdict(dict))


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


def _is_optional(
    field_type: Union[EdgeQuantumNumber, NodeQuantumNumber]
) -> bool:
    if (
        hasattr(field_type, "__origin__")
        and getattr(field_type, "__origin__") is Union
        and type(None) in getattr(field_type, "__args__")
    ):
        return True
    return False


_GraphEdgePropertyMap = Dict[Type[EdgeQuantumNumber], Any]
_GraphNodePropertyMap = Dict[Type[NodeQuantumNumber], Any]
_GraphElementPropertyMap = Union[_GraphEdgePropertyMap, _GraphNodePropertyMap]


def _init_class(
    class_type: Type,
    props: _GraphElementPropertyMap,
) -> object:
    return class_type(
        **{
            class_field.name: _extract_value(props, class_field.type)
            for class_field in attr.fields(class_type)
        }
    )


def _extract_value(
    props: _GraphElementPropertyMap,
    obj_type: Any,
) -> Any:
    if _is_optional(obj_type):
        obj_type = obj_type.__args__[0]
        if obj_type in props:
            value = props[obj_type]
        else:
            return None
    else:
        value = props[obj_type]

    if (
        "__supertype__" in obj_type.__dict__
        and obj_type.__supertype__ == Parity
    ):
        return obj_type.__supertype__(value)
    return obj_type(value)


def _check_arg_requirements(
    class_type: type,
    props: _GraphElementPropertyMap,
) -> bool:
    if attr.has(class_type):
        return all(
            [
                bool(class_field.type in props)
                for class_field in attr.fields(class_type)
                if not _is_optional(class_field.type)  # type: ignore
            ]
        )

    return class_type in props


def _check_requirements(
    rule: Rule,
    in_edge_props: Sequence[_GraphEdgePropertyMap],
    out_edge_props: Sequence[_GraphEdgePropertyMap],
    node_props: _GraphNodePropertyMap,
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
                    for x in props  # type: ignore
                ]
            ):
                return False
        arg_counter += 1

    return True


def _create_rule_edge_arg(
    input_type: Type[Any],
    edge_props: Sequence[_GraphEdgePropertyMap],
) -> List[Any]:
    # pylint: disable=unidiomatic-typecheck
    if not isinstance(edge_props, (list, tuple)):
        raise TypeError("edge_props are incompatible...")
    if not (type(input_type) is list or type(input_type) is tuple):
        raise TypeError("input type is incompatible...")
    in_list_type = input_type[0]

    if attr.has(in_list_type):
        # its a composite type -> create new class type here
        return [_init_class(in_list_type, x) for x in edge_props if x]
    return [_extract_value(x, in_list_type) for x in edge_props if x]


def _create_rule_node_arg(
    input_type: Type[Any],
    node_props: _GraphNodePropertyMap,
) -> Any:
    # pylint: disable=unidiomatic-typecheck
    if isinstance(node_props, (list, tuple)):
        raise TypeError("node_props is incompatible...")
    if type(input_type) is list or type(input_type) is tuple:
        raise TypeError("input type is incompatible...")

    if attr.has(input_type):
        # its a composite type -> create new class type here
        return _init_class(input_type, node_props)
    return _extract_value(node_props, input_type)


def _create_rule_args(
    rule: Rule,
    in_edge_props: Sequence[_GraphEdgePropertyMap],
    out_edge_props: Sequence[_GraphEdgePropertyMap],
    node_props: _GraphNodePropertyMap,
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
                _create_rule_node_arg(
                    arg_type,
                    ordered_props[arg_counter],  # type: ignore
                )
            )
        else:
            args.append(
                _create_rule_edge_arg(
                    arg_type.__args__,
                    ordered_props[arg_counter],  # type: ignore
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


def _get_required_qns(
    rule: Rule,
) -> Tuple[Set[Type[EdgeQuantumNumber]], Set[Type[NodeQuantumNumber]]]:
    if not hasattr(rule.__class__.__call__, "__annotations__"):
        raise TypeError(
            f"missing type annotations for __call__ of rule {rule.__class__.__name__}"
        )

    required_edge_qns: Set[Type[EdgeQuantumNumber]] = set()
    required_node_qns: Set[Type[NodeQuantumNumber]] = set()

    rule_annotations = list(rule.__class__.__call__.__annotations__.values())
    rule_annotations = _remove_return_annotation(rule_annotations)

    arg_counter = 0
    for input_type in rule_annotations:
        qn_set = required_edge_qns
        if arg_counter == 2:
            qn_set = required_node_qns  # type: ignore

        class_type = input_type
        if "__origin__" in input_type.__dict__ and (
            input_type.__origin__ is list
            or input_type.__origin__ is tuple
            or input_type.__origin__ is List
            or input_type.__origin__ is Tuple
        ):
            class_type = input_type.__args__[0]

        if attr.has(class_type):
            for class_field in attr.fields(class_type):
                qn_set.add(
                    class_field.type.__args__[0]  # type: ignore
                    if _is_optional(class_field.type)  # type: ignore
                    else class_field.type
                )
        else:
            qn_set.add(class_type)
        arg_counter += 1

    return (required_edge_qns, required_node_qns)


def _merge_solutions_with_graph(
    solutions: List[_QuantumNumberSolution],
    graph: StateTransitionGraph[ParticleWithSpin],
    allowed_particles: ParticleCollection,
) -> List[StateTransitionGraph[ParticleWithSpin]]:
    initialized_graphs = []

    logging.debug("merging solutions with graph...")
    intermediate_edges = graph.get_intermediate_state_edges()
    for solution in solutions:
        temp_graph = copy(graph)
        for node_id in temp_graph.nodes:
            if node_id in solution.node_quantum_numbers:
                temp_graph.node_props[
                    node_id
                ] = _create_interaction_properties(
                    solution.node_quantum_numbers[node_id]
                )

        current_new_graphs = [temp_graph]
        for int_edge_id in intermediate_edges:
            particle_edges = __get_particle_candidates_for_state(
                solution.edge_quantum_numbers[int_edge_id],
                allowed_particles,
            )
            if len(particle_edges) == 0:
                logging.debug("Did not find any particle candidates for")
                logging.debug("edge id: %d", int_edge_id)
                logging.debug("edge properties:")
                logging.debug(solution.edge_quantum_numbers[int_edge_id])
            new_graphs_temp = []
            for current_new_graph in current_new_graphs:
                for particle_edge in particle_edges:
                    temp_graph = copy(current_new_graph)
                    temp_graph.edge_props[int_edge_id] = particle_edge
                    new_graphs_temp.append(temp_graph)
            current_new_graphs = new_graphs_temp

        initialized_graphs.extend(current_new_graphs)

    return initialized_graphs


def __get_particle_candidates_for_state(
    state: Dict[Type[EdgeQuantumNumber], Scalar],
    allowed_particles: ParticleCollection,
) -> List[ParticleWithSpin]:
    particle_edges: List[ParticleWithSpin] = []

    for allowed_particle in allowed_particles:
        if __check_qns_equal(state, allowed_particle):
            particle_edges.append(
                (allowed_particle, state[EdgeQuantumNumbers.spin_projection])
            )

    return particle_edges


def __check_qns_equal(
    state: Dict[Type[EdgeQuantumNumber], Scalar], particle: Particle
) -> bool:
    # This function assumes the attribute names of Particle and the quantum
    # numbers defined by new type match
    changes_dict: Dict[str, Union[int, float, Parity, Spin]] = {
        edge_qn.__name__: value
        for edge_qn, value in state.items()
        if "magnitude" not in edge_qn.__name__
        and "projection" not in edge_qn.__name__
    }

    if EdgeQuantumNumbers.isospin_magnitude in state:
        changes_dict["isospin"] = Spin(
            state[EdgeQuantumNumbers.isospin_magnitude],
            state[EdgeQuantumNumbers.isospin_projection],
        )
    if EdgeQuantumNumbers.spin_magnitude in state:
        changes_dict["spin"] = state[EdgeQuantumNumbers.spin_magnitude]
    return attr.evolve(particle, **changes_dict) == particle


def validate_fully_initialized_graph(
    graph: StateTransitionGraph[ParticleWithSpin],
    rules_per_node: Dict[int, Set[Rule]],
) -> Result:
    logging.debug("validating graph...")

    def _create_node_variables(
        node_id: int, qn_list: Set[Type[NodeQuantumNumber]]
    ) -> Dict[Type[NodeQuantumNumber], Scalar]:
        """Create variables for the quantum numbers of the specified node."""
        variables = {}
        if node_id in graph.node_props:
            for qn_type in qn_list:
                value = _get_node_quantum_number(
                    qn_type, graph.node_props[node_id]
                )
                if value is not None:
                    variables[qn_type] = value
        return variables

    def _create_edge_variables(
        edge_ids: Sequence[int],
        qn_list: Set[Type[EdgeQuantumNumber]],
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
                for qn_type in qn_list:
                    value = _get_particle_property(edge_props, qn_type)
                    if value is not None:
                        edge_vars[qn_type] = value
                variables.append(edge_vars)
        return variables

    def _create_variable_containers(
        node_id: int, cons_law: Rule
    ) -> Tuple[List[dict], List[dict], dict]:
        in_edges = graph.get_edges_ingoing_to_node(node_id)
        out_edges = graph.get_edges_outgoing_from_node(node_id)

        edge_qns, node_qns = _get_required_qns(cons_law)
        in_edges_vars = _create_edge_variables(in_edges, edge_qns)  # type: ignore
        out_edges_vars = _create_edge_variables(out_edges, edge_qns)  # type: ignore

        node_vars = _create_node_variables(node_id, node_qns)

        return (in_edges_vars, out_edges_vars, node_vars)

    node_violated_rules: Dict[int, Set[Tuple[Rule]]] = defaultdict(set)
    node_not_executed_rules: Dict[int, Set[Rule]] = defaultdict(set)
    for node_id, rules in rules_per_node.items():
        for rule in rules:
            # get the needed qns for this conservation law
            # for all edges and the node
            var_containers = _create_variable_containers(node_id, rule)
            # check the requirements
            if isinstance(rule, IsoSpinValidity) or _check_requirements(
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


_EdgeVariableInfo = Tuple[int, Type[EdgeQuantumNumber]]
_NodeVariableInfo = Tuple[int, Type[NodeQuantumNumber]]


def _create_variable_string(
    element_id: int,
    qn_type: Union[Type[EdgeQuantumNumber], Type[NodeQuantumNumber]],
) -> str:
    return str(element_id) + "-" + qn_type.__name__


@attr.s
class _VariableContainer:
    ingoing_edge_variables: Set[_EdgeVariableInfo] = attr.ib(factory=set)
    fixed_ingoing_edge_variables: Dict[
        int, Dict[Type[EdgeQuantumNumber], Scalar]
    ] = attr.ib(factory=dict)
    outgoing_edge_variables: Set[_EdgeVariableInfo] = attr.ib(factory=set)
    fixed_outgoing_edge_variables: Dict[
        int, Dict[Type[EdgeQuantumNumber], Scalar]
    ] = attr.ib(factory=dict)
    node_variables: Set[_NodeVariableInfo] = attr.ib(factory=set)
    fixed_node_variables: Dict[Type[NodeQuantumNumber], Scalar] = attr.ib(
        factory=dict
    )


class CSPSolver(Solver):
    """Solver reducing the task to a Constraint Satisfaction Problem.

    Solving this done with the python-constraint module.

    The variables are the quantum numbers of particles/edges, but also some
    composite quantum numbers which are attributed to the interaction nodes
    (such as angular momentum :math:`L`). The conservation rules serve as the
    constraints and a special wrapper class serves as an adapter.
    """

    def __init__(self, allowed_intermediate_particles: ParticleCollection):
        self.__graph = StateTransitionGraph[ParticleWithSpin]()
        self.__variables: Set[
            Union[_EdgeVariableInfo, _NodeVariableInfo]
        ] = set()
        self.__var_string_to_data: Dict[
            str, Union[_EdgeVariableInfo, _NodeVariableInfo]
        ] = {}
        self.__constraints: Dict[
            int, Set[_ConservationRuleConstraintWrapper]
        ] = defaultdict(set)
        self.__non_executable_constraints: Dict[
            int, Set[_ConservationRuleConstraintWrapper]
        ] = defaultdict(set)
        self.__problem = Problem(BacktrackingSolver(True))
        self.__allowed_intermediate_particles = allowed_intermediate_particles

    def find_solutions(
        self,
        graph: StateTransitionGraph[ParticleWithSpin],
        graph_settings: GraphSettings,
    ) -> Result:
        self.__graph = graph
        self.__initialize_constraints(graph_settings)
        solutions = self.__problem.getSolutions()

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

        # insert particle instances
        solutions = self.__convert_solution_keys(solutions)
        if self.__constraints:
            full_particle_graphs = _merge_solutions_with_graph(
                solutions, graph, self.__allowed_intermediate_particles
            )
        else:
            full_particle_graphs = [graph]

        if full_particle_graphs and not_executed_rules:
            # rerun solver on these graphs using not executed rules
            # and combine results
            result = Result()
            for full_particle_graph in full_particle_graphs:
                result.extend(
                    validate_fully_initialized_graph(
                        full_particle_graph, not_executed_rules
                    )
                )
            return result

        return Result(
            full_particle_graphs, not_executed_rules, not_satisfied_rules
        )

    def __clear(self) -> None:
        self.__variables = set()
        self.__var_string_to_data = {}
        self.__constraints = defaultdict(set)
        self.__problem = Problem(BacktrackingSolver(True))

    def __initialize_constraints(self, graph_settings: GraphSettings) -> None:
        """Initialize all of the constraints for this graph.

        For each interaction node a set of independent constraints/conservation
        laws are created. For each conservation law a new CSP wrapper is
        created. This wrapper needs all of the qn numbers/variables which enter
        or exit the node and play a role for this conservation law. Hence
        variables are also created within this method.
        """
        # pylint: disable=too-many-locals

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
                variable_mapping = _VariableContainer()
                # from cons law and graph determine needed var lists
                edge_qns, node_qns = _get_required_qns(cons_law)

                in_edges = self.__graph.get_edges_ingoing_to_node(node_id)
                in_edge_vars = self.__create_edge_variables(
                    in_edges, edge_qns, graph_settings.edge_settings
                )
                variable_mapping.ingoing_edge_variables = in_edge_vars[0]
                variable_mapping.fixed_ingoing_edge_variables = in_edge_vars[1]
                var_list: List[
                    Union[_EdgeVariableInfo, _NodeVariableInfo]
                ] = list(variable_mapping.ingoing_edge_variables)

                out_edges = self.__graph.get_edges_outgoing_from_node(node_id)
                out_edge_vars = self.__create_edge_variables(
                    out_edges, edge_qns, graph_settings.edge_settings
                )
                variable_mapping.outgoing_edge_variables = out_edge_vars[0]
                variable_mapping.fixed_outgoing_edge_variables = out_edge_vars[
                    1
                ]
                var_list.extend(list(variable_mapping.outgoing_edge_variables))

                # now create variables for node/interaction qns
                int_node_vars = self.__create_node_variables(
                    node_id,
                    node_qns,
                    graph_settings.node_settings,
                )
                variable_mapping.node_variables = int_node_vars[0]
                variable_mapping.fixed_node_variables = int_node_vars[1]
                var_list.extend(list(variable_mapping.node_variables))

                constraint = _ConservationRuleConstraintWrapper(
                    cons_law,
                    variable_mapping,
                )
                if var_list:
                    var_strings = [
                        _create_variable_string(*x) for x in var_list
                    ]
                    self.__constraints[node_id].add(constraint)
                    self.__problem.addConstraint(constraint, var_strings)
                else:
                    self.__non_executable_constraints[node_id].add(constraint)

    def __create_node_variables(
        self,
        node_id: int,
        qn_list: Set[Type[NodeQuantumNumber]],
        node_settings: Dict[int, NodeSettings],
    ) -> Tuple[Set[_NodeVariableInfo], Dict[Type[NodeQuantumNumber], Scalar],]:
        """Create variables for the quantum numbers of the specified node.

        If a quantum number is already defined for a node, then a fixed
        variable is created, which cannot be changed by the csp solver.
        Otherwise the node is initialized with the specified domain of that
        quantum number.
        """
        variables: Tuple[
            Set[_NodeVariableInfo],
            Dict[Type[NodeQuantumNumber], Scalar],
        ] = (
            set(),
            dict(),
        )

        if node_id in self.__graph.node_props:
            node_props = self.__graph.node_props[node_id]
            for qn_type in qn_list:
                value = _get_node_quantum_number(qn_type, node_props)
                if value is not None:
                    variables[1].update({qn_type: value})
        else:
            for qn_type in qn_list:
                var_info = (node_id, qn_type)
                if qn_type in node_settings[node_id].qn_domains:
                    qn_domain = node_settings[node_id].qn_domains[qn_type]
                    self.__add_variable(var_info, qn_domain)
                    variables[0].add(var_info)
        return variables

    def __create_edge_variables(
        self,
        edge_ids: Sequence[int],
        qn_list: Set[Type[EdgeQuantumNumber]],
        edge_settings: Dict[int, EdgeSettings],
    ) -> Tuple[
        Set[_EdgeVariableInfo],
        Dict[int, Dict[Type[EdgeQuantumNumber], Scalar]],
    ]:
        """Create variables for the quantum numbers of the specified edges.

        If a quantum number is already defined for an edge, then a fixed
        variable is created, which cannot be changed by the csp solver. This is
        the case for initial and final state edges. Otherwise the edges are
        initialized with the specified domains of that quantum number.
        """
        variables: Tuple[
            Set[_EdgeVariableInfo],
            Dict[int, Dict[Type[EdgeQuantumNumber], Scalar]],
        ] = (
            set(),
            dict(),
        )
        for edge_id in edge_ids:
            variables[1][edge_id] = {}
            if edge_id in self.__graph.edge_props:
                edge_props = self.__graph.edge_props[edge_id]
                for qn_type in qn_list:
                    value = _get_particle_property(edge_props, qn_type)
                    if value is not None:
                        variables[1][edge_id].update({qn_type: value})
            else:
                for qn_type in qn_list:
                    var_info = (edge_id, qn_type)
                    if qn_type in edge_settings[edge_id].qn_domains:
                        qn_domain = edge_settings[edge_id].qn_domains[qn_type]
                        self.__add_variable(var_info, qn_domain)
                        variables[0].add(var_info)
        return variables

    def __add_variable(
        self,
        var_info: Union[_EdgeVariableInfo, _NodeVariableInfo],
        domain: List[Any],
    ) -> None:
        if var_info not in self.__variables:
            self.__variables.add(var_info)
            var_string = _create_variable_string(*var_info)
            self.__var_string_to_data[var_string] = var_info
            self.__problem.addVariable(var_string, domain)

    def __convert_solution_keys(
        self,
        solutions: List[Dict[str, Scalar]],
    ) -> List[_QuantumNumberSolution]:
        """Convert keys of CSP solutions from string to quantum number types."""
        initial_edges = self.__graph.get_initial_state_edges()
        final_edges = self.__graph.get_final_state_edges()

        converted_solutions = list()
        for solution in solutions:
            qn_solution = _QuantumNumberSolution()
            for var_string, value in solution.items():
                ele_id, qn_type = self.__var_string_to_data[var_string]

                if qn_type in getattr(EdgeQuantumNumber, "__args__"):
                    if ele_id in initial_edges or ele_id in final_edges:
                        # skip if its an initial or final state edge
                        continue
                    qn_solution.edge_quantum_numbers[ele_id].update(
                        {qn_type: value}  # type: ignore
                    )
                else:
                    qn_solution.node_quantum_numbers[ele_id].update(
                        {qn_type: value}  # type: ignore
                    )
            converted_solutions.append(qn_solution)

        return converted_solutions


class _ConservationRuleConstraintWrapper(Constraint):
    """Wrapper class of the python-constraint Constraint class.

    This allows a customized definition of conservation rules, and hence a
    cleaner user interface.
    """

    def __init__(self, rule: Rule, variables: _VariableContainer) -> None:
        if not isinstance(rule, Rule):
            raise TypeError("rule has to be of type Rule!")
        self.__rule = rule
        self.__var_string_to_data: Dict[
            str,
            Union[_EdgeVariableInfo, _NodeVariableInfo],
        ] = {}
        self.__in_edges_qns: Dict[int, _GraphEdgePropertyMap] = {}
        self.__out_edges_qns: Dict[int, _GraphEdgePropertyMap] = {}
        self.__node_qns: _GraphNodePropertyMap = {}

        self.conditions_never_met = False
        self.scenario_results = [0, 0]

        self.__initialize_variable_containers(variables)

    @property
    def rule(self) -> Rule:
        return self.__rule

    def __initialize_variable_containers(
        self, variables: _VariableContainer
    ) -> None:
        """Fill the name decoding map.

        Also initialize the in and out particle lists. The variable names
        follow the scheme edge_id(delimiter)qn_name. This method creates a dict
        linking the var name to a list that consists of the particle list index
        and the qn name.
        """

        def _initialize_edge_container(
            variable_set: Set[_EdgeVariableInfo],
            fixed_variables: Dict[int, Dict[Type[EdgeQuantumNumber], Scalar]],
            container: Dict[int, _GraphEdgePropertyMap],
        ) -> None:
            container.update(fixed_variables)
            for element_id, qn_type in variable_set:
                self.__var_string_to_data[
                    _create_variable_string(element_id, qn_type)
                ] = (element_id, qn_type)
                if element_id not in container:
                    container[element_id] = {}
                container[element_id].update({qn_type: None})

        _initialize_edge_container(
            variables.ingoing_edge_variables,
            variables.fixed_ingoing_edge_variables,
            self.__in_edges_qns,
        )
        _initialize_edge_container(
            variables.outgoing_edge_variables,
            variables.fixed_outgoing_edge_variables,
            self.__out_edges_qns,
        )
        # and now interaction node variables
        for var_info in variables.node_variables:
            self.__node_qns[var_info[1]] = None
            self.__var_string_to_data[
                _create_variable_string(*var_info)
            ] = var_info
        self.__node_qns.update(variables.fixed_node_variables)

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

        self.__update_variable_lists(params)
        if not isinstance(self.rule, IsoSpinValidity):
            if not _check_requirements(
                self.__rule,
                list(self.__in_edges_qns.values()),
                list(self.__out_edges_qns.values()),
                self.__node_qns,
            ):
                self.conditions_never_met = True
                return True

        passed = self.__rule(
            *_create_rule_args(
                self.__rule,
                list(self.__in_edges_qns.values()),
                list(self.__out_edges_qns.values()),
                self.__node_qns,
            )
        )

        # before returning gather statistics about the rule
        if passed:
            self.scenario_results[1] += 1
        else:
            self.scenario_results[0] += 1
        return passed

    def __update_variable_lists(
        self,
        parameters: List[Tuple[str, Any]],
    ) -> None:
        for var_string, value in parameters:
            index, qn_type = self.__var_string_to_data[var_string]
            if (
                index in self.__in_edges_qns
                and qn_type in self.__in_edges_qns[index]
            ):
                self.__in_edges_qns[index][qn_type] = value  # type: ignore
            elif (
                index in self.__out_edges_qns
                and qn_type in self.__out_edges_qns[index]
            ):
                self.__out_edges_qns[index][qn_type] = value  # type: ignore
            elif qn_type in self.__node_qns:
                self.__node_qns[qn_type] = value  # type: ignore
            else:
                raise ValueError(
                    "The variable with name "
                    + qn_type.__name__
                    + "does not appear in the variable mapping!"
                )


def _get_particle_property(
    edge_property: ParticleWithSpin, qn_type: Type[EdgeQuantumNumber]
) -> Optional[Union[float, int]]:
    """Convert a data member of `.Particle` into one of `.EdgeQuantumNumbers`.

    The `.reaction` model requires a list of 'flat' values, such as `int` and
    `float`. It cannot handle `.Spin` (which contains `~.Spin.magnitude` and
    `~.Spin.projection`). The `.reaction` module also works with spin
    projection, which a general `.Particle` instance does not carry.
    """
    particle, spin_projection = edge_property
    value = None
    if hasattr(particle, qn_type.__name__):
        value = getattr(particle, qn_type.__name__)
    else:
        if qn_type is EdgeQuantumNumbers.spin_magnitude:
            value = particle.spin
        elif qn_type is EdgeQuantumNumbers.spin_projection:
            value = spin_projection
        if particle.isospin is not None:
            if qn_type is EdgeQuantumNumbers.isospin_magnitude:
                value = particle.isospin.magnitude
            elif qn_type is EdgeQuantumNumbers.isospin_projection:
                value = particle.isospin.projection

    if isinstance(value, Parity):
        return int(value)
    return value


def _get_node_quantum_number(
    qn_type: Type[NodeQuantumNumber], node_props: InteractionProperties
) -> Optional[Scalar]:
    return getattr(node_props, qn_type.__name__)


def _create_interaction_properties(
    qn_solution: Dict[Type[NodeQuantumNumber], Scalar]
) -> InteractionProperties:
    converted_solution = {k.__name__: v for k, v in qn_solution.items()}
    kw_args = {
        x.name: converted_solution[x.name]
        for x in attr.fields(InteractionProperties)
        if x.name in converted_solution
    }

    return attr.evolve(InteractionProperties(), **kw_args)
