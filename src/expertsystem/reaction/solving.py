# pylint: disable=too-many-lines

"""Functions to solve a particle reaction problem.

This module is responsible for solving a particle reaction problem stated by a
`.StateTransitionGraph` and corresponding `.GraphSettings`. The `.Solver`
classes (e.g. :class:`.CSPSolver`) generate new quantum numbers (for example
belonging to an intermediate state) and validate the decay processes with the
rules formulated by the :mod:`.conservation_rules` module.
"""


import inspect
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr
from constraint import (
    BacktrackingSolver,
    Constraint,
    Problem,
    Unassigned,
    Variable,
)

from .argument_handling import (
    GraphEdgePropertyMap,
    GraphElementRule,
    GraphNodePropertyMap,
    Rule,
    RuleArgumentHandler,
    Scalar,
    get_required_qns,
)
from .quantum_numbers import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    NodeQuantumNumber,
)
from .topology import Topology


@attr.s
class EdgeSettings:
    """Solver settings for a specific edge of a graph."""

    conservation_rules: Set[GraphElementRule] = attr.ib(factory=set)
    rule_priorities: Dict[GraphElementRule, int] = attr.ib(factory=dict)
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
    rule_priorities: Dict[Rule, int] = attr.ib(factory=dict)
    qn_domains: Dict[Any, Any] = attr.ib(factory=dict)
    interaction_strength: float = 1.0


@attr.s
class GraphSettings:
    edge_settings: Dict[int, EdgeSettings] = attr.ib(factory=dict)
    node_settings: Dict[int, NodeSettings] = attr.ib(factory=dict)


@attr.s
class GraphElementProperties:
    edge_props: Dict[int, GraphEdgePropertyMap] = attr.ib(factory=dict)
    node_props: Dict[int, GraphNodePropertyMap] = attr.ib(factory=dict)


@attr.s(frozen=True)
class QNProblemSet:
    """Particle reaction problem set, defined as a graph like data structure.

    Args:
      topology (`.Topology`): a topology that represent the structure of the
        reaction
      initial_facts (`.GraphElementProperties`): all of the known facts quantum
        numbers of the problem
      solving_settings (`.GraphSettings`): solving specific settings such as
        the specific rules and variable domains for nodes and edges of the
        topology
    """

    topology: Topology = attr.ib()
    initial_facts: GraphElementProperties = attr.ib()
    solving_settings: GraphSettings = attr.ib()


@attr.s(frozen=True)
class QuantumNumberSolution:
    node_quantum_numbers: Dict[int, GraphNodePropertyMap] = attr.ib()
    edge_quantum_numbers: Dict[int, GraphEdgePropertyMap] = attr.ib()


def _convert_violated_rules_to_names(
    rules: Union[
        Dict[int, Set[Rule]],
        Dict[int, Set[GraphElementRule]],
    ]
) -> Dict[int, Set[str]]:
    def get_name(rule: Any) -> str:
        if inspect.isfunction(rule):
            return rule.__name__
        if isinstance(rule, str):
            return rule
        return rule.__class__.__name__

    converted_dict = defaultdict(set)
    for node_id, rule_set in rules.items():
        converted_dict[node_id] = {get_name(rule) for rule in rule_set}

    return converted_dict


def _convert_non_executed_rules_to_names(
    rules: Union[
        Dict[int, Set[Rule]],
        Dict[int, Set[GraphElementRule]],
    ]
) -> Dict[int, Set[str]]:
    def get_name(rule: Any) -> str:
        if inspect.isfunction(rule):
            return rule.__name__
        if isinstance(rule, str):
            return rule
        return rule.__class__.__name__

    converted_dict = defaultdict(set)
    for node_id, rule_set in rules.items():
        rule_name_set = set()
        for rule_tuple in rule_set:
            rule_name_set.add(get_name(rule_tuple))

        converted_dict[node_id] = rule_name_set

    return converted_dict


@attr.s(on_setattr=attr.setters.frozen)
class QNResult:
    """Defines a result to a problem set processed by the solving code."""

    solutions: List[QuantumNumberSolution] = attr.ib(factory=list)
    not_executed_node_rules: Dict[int, Set[str]] = attr.ib(
        factory=lambda: defaultdict(set)
    )
    violated_node_rules: Dict[int, Set[str]] = attr.ib(
        factory=lambda: defaultdict(set)
    )
    not_executed_edge_rules: Dict[int, Set[str]] = attr.ib(
        factory=lambda: defaultdict(set)
    )
    violated_edge_rules: Dict[int, Set[str]] = attr.ib(
        factory=lambda: defaultdict(set)
    )

    def __attrs_post_init__(self) -> None:
        if self.solutions and (
            self.violated_node_rules or self.violated_edge_rules
        ):
            raise ValueError(
                f"Invalid {self.__class__.__name__}!"
                f" Found {len(self.solutions)} solutions, but also violated rules.",
                self.violated_node_rules,
                self.violated_edge_rules,
            )

    def extend(self, other_result: "QNResult") -> None:
        if self.solutions or other_result.solutions:
            self.solutions.extend(other_result.solutions)
            self.not_executed_node_rules.clear()
            self.violated_node_rules.clear()
            self.not_executed_edge_rules.clear()
            self.violated_edge_rules.clear()
        else:
            for key, rules in other_result.not_executed_node_rules.items():
                self.not_executed_node_rules[key].update(rules)

            for key, rules in other_result.not_executed_edge_rules.items():
                self.not_executed_edge_rules[key].update(rules)

            for key, rules2 in other_result.violated_node_rules.items():
                self.violated_node_rules[key].update(rules2)

            for key, rules2 in other_result.violated_edge_rules.items():
                self.violated_edge_rules[key].update(rules2)


class Solver(ABC):
    """Interface of a Solver."""

    @abstractmethod
    def find_solutions(self, problem_set: QNProblemSet) -> QNResult:
        """Find solutions for the given input.

        It is expected that this function determines and returns all of the
        found solutions. In case no solutions are found a partial list of
        violated rules has to be given. This list of violated rules does not
        have to be complete.

        Args:
          problem_set (`.QNProblemSet`): states a problem set

        Returns:
          QNResult: contains possible solutions, violated rules and not executed
          rules due to requirement issues.
        """


def _merge_particle_candidates_with_solutions(
    solutions: List[QuantumNumberSolution],
    topology: Topology,
    allowed_particles: List[GraphEdgePropertyMap],
) -> List[QuantumNumberSolution]:
    merged_solutions = []

    logging.debug("merging solutions with graph...")
    intermediate_edges = topology.intermediate_edge_ids
    for solution in solutions:
        current_new_solutions = [solution]
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
            new_solutions_temp = []
            for current_new_solution in current_new_solutions:
                for particle_edge in particle_edges:
                    # a "shallow" copy of the nested dicts is needed
                    new_edge_qns = {
                        k: copy(v)
                        for k, v in current_new_solution.edge_quantum_numbers.items()
                    }
                    new_edge_qns[int_edge_id].update(particle_edge)
                    temp_solution = attr.evolve(
                        current_new_solution,
                        edge_quantum_numbers=new_edge_qns,
                    )
                    new_solutions_temp.append(temp_solution)
            current_new_solutions = new_solutions_temp

        merged_solutions.extend(current_new_solutions)

    return merged_solutions


def __get_particle_candidates_for_state(
    state: GraphEdgePropertyMap,
    allowed_particles: List[GraphEdgePropertyMap],
) -> List[GraphEdgePropertyMap]:
    particle_edges = []

    for particle_qns in allowed_particles:
        if __is_sub_mapping(state, particle_qns):
            particle_edges.append(particle_qns)

    return particle_edges


def __is_sub_mapping(
    qn_state: GraphEdgePropertyMap, reference_qn_state: GraphEdgePropertyMap
) -> bool:
    for qn_type, qn_value in qn_state.items():
        if qn_type is EdgeQuantumNumbers.spin_projection:
            continue
        if qn_type not in reference_qn_state:
            return False
        if qn_value != reference_qn_state[qn_type]:
            return False

    return True


def validate_full_solution(problem_set: QNProblemSet) -> QNResult:
    # pylint: disable=too-many-locals
    logging.debug("validating graph...")

    rule_argument_handler = RuleArgumentHandler()

    def _create_node_variables(
        node_id: int, qn_list: Set[Type[NodeQuantumNumber]]
    ) -> Dict[Type[NodeQuantumNumber], Scalar]:
        """Create variables for the quantum numbers of the specified node."""
        variables = {}
        if node_id in problem_set.initial_facts.node_props:
            node_props = problem_set.initial_facts.node_props[node_id]
            variables = node_props
            for qn_type in qn_list:
                if qn_type in node_props:
                    variables[qn_type] = node_props[qn_type]
        return variables

    def _create_edge_variables(
        edge_ids: Iterable[int],
        qn_list: Set[Type[EdgeQuantumNumber]],
    ) -> List[dict]:
        """Create variables for the quantum numbers of the specified edges.

        Initial and final state edges just get a single domain value.
        Intermediate edges are initialized with the default domains of that
        quantum number.
        """
        variables = []
        for edge_id in edge_ids:
            if edge_id in problem_set.initial_facts.edge_props:
                edge_props = problem_set.initial_facts.edge_props[edge_id]
                edge_vars = {}
                for qn_type in qn_list:
                    if qn_type in edge_props:
                        edge_vars[qn_type] = edge_props[qn_type]
                variables.append(edge_vars)
        return variables

    def _create_variable_containers(
        node_id: int, cons_law: Rule
    ) -> Tuple[List[dict], List[dict], dict]:
        in_edges = problem_set.topology.get_edge_ids_ingoing_to_node(node_id)
        out_edges = problem_set.topology.get_edge_ids_outgoing_from_node(
            node_id
        )

        edge_qns, node_qns = get_required_qns(cons_law)
        in_edges_vars = _create_edge_variables(in_edges, edge_qns)
        out_edges_vars = _create_edge_variables(out_edges, edge_qns)

        node_vars = _create_node_variables(node_id, node_qns)

        return (in_edges_vars, out_edges_vars, node_vars)

    edge_violated_rules: Dict[int, Set[GraphElementRule]] = defaultdict(set)
    edge_not_executed_rules: Dict[int, Set[GraphElementRule]] = defaultdict(
        set
    )
    node_violated_rules: Dict[int, Set[Rule]] = defaultdict(set)
    node_not_executed_rules: Dict[int, Set[Rule]] = defaultdict(set)
    for (
        edge_id,
        edge_settings,
    ) in problem_set.solving_settings.edge_settings.items():
        edge_rules = edge_settings.conservation_rules
        for edge_rule in edge_rules:
            # get the needed qns for this conservation law
            # for all edges and the node
            (
                check_requirements,
                create_rule_args,
            ) = rule_argument_handler.register_rule(edge_rule)

            edge_qns, _ = get_required_qns(edge_rule)
            edge_variables = _create_edge_variables([edge_id], edge_qns)[0]
            if check_requirements(
                edge_variables,
            ):
                if not edge_rule(
                    *create_rule_args(
                        edge_variables,
                    )
                ):
                    edge_violated_rules[edge_id].add(edge_rule)
            else:
                edge_not_executed_rules[edge_id].add(edge_rule)

    for (
        node_id,
        node_settings,
    ) in problem_set.solving_settings.node_settings.items():
        node_rules = node_settings.conservation_rules
        for rule in node_rules:
            # get the needed qns for this conservation law
            # for all edges and the node
            (
                check_requirements,
                create_rule_args,
            ) = rule_argument_handler.register_rule(rule)

            var_containers = _create_variable_containers(node_id, rule)
            if check_requirements(
                var_containers[0],
                var_containers[1],
                var_containers[2],
            ):
                if not rule(
                    *create_rule_args(
                        var_containers[0],
                        var_containers[1],
                        var_containers[2],
                    )
                ):
                    node_violated_rules[node_id].add(rule)
            else:
                node_not_executed_rules[node_id].add(rule)
    if node_violated_rules or node_not_executed_rules:
        return QNResult(
            [],
            _convert_non_executed_rules_to_names(node_not_executed_rules),
            _convert_violated_rules_to_names(node_violated_rules),
            _convert_non_executed_rules_to_names(edge_not_executed_rules),
            _convert_violated_rules_to_names(edge_violated_rules),
        )
    return QNResult(
        [
            QuantumNumberSolution(
                edge_quantum_numbers=problem_set.initial_facts.edge_props,
                node_quantum_numbers=problem_set.initial_facts.node_props,
            )
        ],
    )


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
    fixed_ingoing_edge_variables: Dict[int, GraphEdgePropertyMap] = attr.ib(
        factory=dict
    )
    outgoing_edge_variables: Set[_EdgeVariableInfo] = attr.ib(factory=set)
    fixed_outgoing_edge_variables: Dict[int, GraphEdgePropertyMap] = attr.ib(
        factory=dict
    )
    node_variables: Set[_NodeVariableInfo] = attr.ib(factory=set)
    fixed_node_variables: GraphNodePropertyMap = attr.ib(factory=dict)


class CSPSolver(Solver):
    """Solver reducing the task to a Constraint Satisfaction Problem.

    Solving this done with the python-constraint module.

    The variables are the quantum numbers of particles/edges, but also some
    composite quantum numbers which are attributed to the interaction nodes
    (such as angular momentum :math:`L`). The conservation rules serve as the
    constraints and a special wrapper class serves as an adapter.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self, allowed_intermediate_particles: List[GraphEdgePropertyMap]
    ):
        self.__variables: Set[
            Union[_EdgeVariableInfo, _NodeVariableInfo]
        ] = set()
        self.__var_string_to_data: Dict[
            str, Union[_EdgeVariableInfo, _NodeVariableInfo]
        ] = {}
        self.__node_rules: Dict[int, Set[Rule]] = defaultdict(set)
        self.__non_executable_node_rules: Dict[int, Set[Rule]] = defaultdict(
            set
        )
        self.__edge_rules: Dict[int, Set[GraphElementRule]] = defaultdict(set)
        self.__non_executable_edge_rules: Dict[
            int, Set[GraphElementRule]
        ] = defaultdict(set)
        self.__problem = Problem(BacktrackingSolver(True))
        self.__allowed_intermediate_particles = allowed_intermediate_particles
        self.__scoresheet = Scoresheet()

    def find_solutions(self, problem_set: QNProblemSet) -> QNResult:
        # pylint: disable=too-many-locals
        self.__initialize_constraints(problem_set)
        solutions = self.__problem.getSolutions()

        node_not_executed_rules = self.__non_executable_node_rules
        node_not_satisfied_rules: Dict[int, Set[Rule]] = defaultdict(set)
        edge_not_executed_rules = self.__non_executable_edge_rules
        edge_not_satisfied_rules: Dict[
            int, Set[GraphElementRule]
        ] = defaultdict(set)
        for node_id, rules in self.__node_rules.items():
            for rule in rules:
                if self.__scoresheet.rule_calls[(node_id, rule)] == 0:
                    node_not_executed_rules[node_id].add(rule)
                elif self.__scoresheet.rule_passes[(node_id, rule)] == 0:
                    node_not_satisfied_rules[node_id].add(rule)

        for edge_id, edge_rules in self.__edge_rules.items():
            for rule in edge_rules:
                if self.__scoresheet.rule_calls[(edge_id, rule)] == 0:
                    edge_not_executed_rules[edge_id].add(rule)
                elif self.__scoresheet.rule_passes[(edge_id, rule)] == 0:
                    edge_not_satisfied_rules[edge_id].add(rule)

        solutions = self.__convert_solution_keys(solutions)

        # insert particle instances
        if self.__node_rules or self.__edge_rules:
            full_particle_solutions = (
                _merge_particle_candidates_with_solutions(
                    solutions,
                    problem_set.topology,
                    self.__allowed_intermediate_particles,
                )
            )
        else:
            full_particle_solutions = [
                QuantumNumberSolution(
                    node_quantum_numbers=problem_set.initial_facts.node_props,
                    edge_quantum_numbers=problem_set.initial_facts.edge_props,
                )
            ]

        if full_particle_solutions and (
            node_not_executed_rules or edge_not_executed_rules
        ):
            # rerun solver on these graphs using not executed rules
            # and combine results
            result = QNResult()
            for full_particle_solution in full_particle_solutions:
                node_props = full_particle_solution.node_quantum_numbers
                edge_props = full_particle_solution.edge_quantum_numbers
                node_props.update(problem_set.initial_facts.node_props)
                edge_props.update(problem_set.initial_facts.edge_props)
                result.extend(
                    validate_full_solution(
                        QNProblemSet(
                            topology=problem_set.topology,
                            initial_facts=GraphElementProperties(
                                node_props=node_props,
                                edge_props=edge_props,
                            ),
                            solving_settings=GraphSettings(
                                node_settings={
                                    i: NodeSettings(conservation_rules=rules)
                                    for i, rules in node_not_executed_rules.items()
                                },
                                edge_settings={
                                    i: EdgeSettings(conservation_rules=rules)
                                    for i, rules in edge_not_executed_rules.items()
                                },
                            ),
                        )
                    )
                )
            return result

        return QNResult(
            full_particle_solutions,
            _convert_non_executed_rules_to_names(node_not_executed_rules),
            _convert_violated_rules_to_names(node_not_satisfied_rules),
            _convert_non_executed_rules_to_names(edge_not_executed_rules),
            _convert_violated_rules_to_names(edge_not_satisfied_rules),
        )

    def __clear(self) -> None:
        self.__variables = set()
        self.__var_string_to_data = {}
        self.__node_rules = defaultdict(set)
        self.__edge_rules = defaultdict(set)
        self.__problem = Problem(BacktrackingSolver(True))
        self.__scoresheet = Scoresheet()

    def __initialize_constraints(self, problem_set: QNProblemSet) -> None:
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
            graph_element_settings: Union[
                NodeSettings,
                EdgeSettings,
            ]
        ) -> List[Rule]:
            # first add priorities to the entries
            priority_list = [
                (x, graph_element_settings.rule_priorities[type(x)])
                if type(x) in graph_element_settings.rule_priorities
                else (x, 1)
                for x in graph_element_settings.conservation_rules
            ]
            # then sort according to priority
            sorted_list = sorted(
                priority_list, key=lambda x: x[1], reverse=True
            )
            # and strip away the priorities again
            return [x[0] for x in sorted_list]

        arg_handler = RuleArgumentHandler()

        for edge_id in problem_set.topology.edges:
            edge_settings = problem_set.solving_settings.edge_settings[edge_id]
            for rule in get_rules_by_priority(edge_settings):
                variable_mapping = _VariableContainer()
                # from cons law and graph determine needed var lists
                edge_qns, node_qns = get_required_qns(rule)

                edge_vars, fixed_edge_vars = self.__create_edge_variables(
                    [
                        edge_id,
                    ],
                    edge_qns,
                    problem_set,
                )

                score_callback = self.__scoresheet.register_rule(edge_id, rule)
                constraint = _GraphElementConstraint[EdgeQuantumNumber](
                    rule,  # type: ignore
                    edge_vars,
                    fixed_edge_vars,
                    arg_handler,
                    score_callback,
                )

                if edge_vars:
                    var_strings = [
                        _create_variable_string(*x) for x in edge_vars
                    ]
                    self.__edge_rules[edge_id].add(rule)  # type: ignore
                    self.__problem.addConstraint(constraint, var_strings)
                else:
                    self.__non_executable_edge_rules[edge_id].add(
                        rule  # type: ignore
                    )

        for node_id in problem_set.topology.nodes:
            for rule in get_rules_by_priority(
                problem_set.solving_settings.node_settings[node_id]
            ):
                variable_mapping = _VariableContainer()
                # from cons law and graph determine needed var lists
                edge_qns, node_qns = get_required_qns(rule)

                in_edges = problem_set.topology.get_edge_ids_ingoing_to_node(
                    node_id
                )
                in_edge_vars = self.__create_edge_variables(
                    in_edges, edge_qns, problem_set
                )
                variable_mapping.ingoing_edge_variables = in_edge_vars[0]
                variable_mapping.fixed_ingoing_edge_variables = in_edge_vars[1]
                var_list: List[
                    Union[_EdgeVariableInfo, _NodeVariableInfo]
                ] = list(variable_mapping.ingoing_edge_variables)

                out_edges = (
                    problem_set.topology.get_edge_ids_outgoing_from_node(
                        node_id
                    )
                )
                out_edge_vars = self.__create_edge_variables(
                    out_edges, edge_qns, problem_set
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
                    problem_set,
                )
                variable_mapping.node_variables = int_node_vars[0]
                variable_mapping.fixed_node_variables = int_node_vars[1]
                var_list.extend(list(variable_mapping.node_variables))

                score_callback = self.__scoresheet.register_rule(node_id, rule)
                if len(inspect.signature(rule).parameters) == 1:
                    constraint = _GraphElementConstraint[NodeQuantumNumber](
                        rule,  # type: ignore
                        int_node_vars[0],
                        {node_id: int_node_vars[1]},
                        arg_handler,
                        score_callback,
                    )
                else:
                    constraint = _ConservationRuleConstraintWrapper(
                        rule, variable_mapping, arg_handler, score_callback
                    )
                if var_list:
                    var_strings = [
                        _create_variable_string(*x) for x in var_list
                    ]
                    self.__node_rules[node_id].add(rule)
                    self.__problem.addConstraint(constraint, var_strings)
                else:
                    self.__non_executable_node_rules[node_id].add(rule)

    def __create_node_variables(
        self,
        node_id: int,
        qn_list: Set[Type[NodeQuantumNumber]],
        problem_set: QNProblemSet,
    ) -> Tuple[Set[_NodeVariableInfo], GraphNodePropertyMap]:
        """Create variables for the quantum numbers of the specified node.

        If a quantum number is already defined for a node, then a fixed
        variable is created, which cannot be changed by the csp solver.
        Otherwise the node is initialized with the specified domain of that
        quantum number.
        """
        variables: Tuple[Set[_NodeVariableInfo], GraphNodePropertyMap] = (
            set(),
            dict(),
        )

        if node_id in problem_set.initial_facts.node_props:
            node_props = problem_set.initial_facts.node_props[node_id]
            for qn_type in qn_list:
                if qn_type in node_props:
                    variables[1].update({qn_type: node_props[qn_type]})
        else:
            node_settings = problem_set.solving_settings.node_settings[node_id]
            for qn_type in qn_list:
                var_info = (node_id, qn_type)
                if qn_type in node_settings.qn_domains:
                    qn_domain = node_settings.qn_domains[qn_type]
                    self.__add_variable(var_info, qn_domain)
                    variables[0].add(var_info)
        return variables

    def __create_edge_variables(
        self,
        edge_ids: Iterable[int],
        qn_list: Set[Type[EdgeQuantumNumber]],
        problem_set: QNProblemSet,
    ) -> Tuple[Set[_EdgeVariableInfo], Dict[int, GraphEdgePropertyMap]]:
        """Create variables for the quantum numbers of the specified edges.

        If a quantum number is already defined for an edge, then a fixed
        variable is created, which cannot be changed by the csp solver. This is
        the case for initial and final state edges. Otherwise the edges are
        initialized with the specified domains of that quantum number.
        """
        variables: Tuple[
            Set[_EdgeVariableInfo],
            Dict[int, GraphEdgePropertyMap],
        ] = (
            set(),
            dict(),
        )

        for edge_id in edge_ids:
            variables[1][edge_id] = {}
            if edge_id in problem_set.initial_facts.edge_props:
                edge_props = problem_set.initial_facts.edge_props[edge_id]
                for qn_type in qn_list:
                    if qn_type in edge_props:
                        variables[1][edge_id].update(
                            {qn_type: edge_props[qn_type]}
                        )
            else:
                edge_settings = problem_set.solving_settings.edge_settings[
                    edge_id
                ]
                for qn_type in qn_list:
                    var_info = (edge_id, qn_type)
                    if qn_type in edge_settings.qn_domains:
                        qn_domain = edge_settings.qn_domains[qn_type]
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
    ) -> List[QuantumNumberSolution]:
        """Convert keys of CSP solutions from string to quantum number types."""
        converted_solutions = list()
        for solution in solutions:
            edge_quantum_numbers: Dict[
                int, GraphEdgePropertyMap
            ] = defaultdict(dict)
            node_quantum_numbers: Dict[
                int, GraphNodePropertyMap
            ] = defaultdict(dict)
            for var_string, value in solution.items():
                ele_id, qn_type = self.__var_string_to_data[var_string]

                if qn_type in getattr(  # noqa: B009
                    EdgeQuantumNumber, "__args__"
                ):
                    edge_quantum_numbers[ele_id].update(
                        {qn_type: value}  # type: ignore
                    )
                else:
                    node_quantum_numbers[ele_id].update(
                        {qn_type: value}  # type: ignore
                    )
            converted_solutions.append(
                QuantumNumberSolution(
                    node_quantum_numbers, edge_quantum_numbers
                )
            )

        return converted_solutions


class Scoresheet:
    def __init__(self) -> None:
        self.__rule_calls: Dict[Tuple[int, Rule], int] = {}
        self.__rule_passes: Dict[Tuple[int, Rule], int] = {}

    def register_rule(
        self, graph_element_id: int, rule: Rule
    ) -> Callable[[bool], None]:
        self.__rule_calls[(graph_element_id, rule)] = 0
        self.__rule_passes[(graph_element_id, rule)] = 0

        return self.__create_callback(graph_element_id, rule)

    def __create_callback(
        self, graph_element_id: int, rule: Rule
    ) -> Callable[[bool], None]:
        def passed_callback(passed: bool) -> None:
            if passed:
                self.__rule_passes[(graph_element_id, rule)] += 1
            self.__rule_calls[(graph_element_id, rule)] += 1

        return passed_callback

    @property
    def rule_calls(self) -> Dict[Tuple[int, Rule], int]:
        return self.__rule_calls

    @property
    def rule_passes(self) -> Dict[Tuple[int, Rule], int]:
        return self.__rule_passes


_QNType = TypeVar("_QNType", EdgeQuantumNumber, NodeQuantumNumber)


class _GraphElementConstraint(Generic[_QNType], Constraint):
    """Wrapper class of the python-constraint Constraint class.

    This allows a customized definition of conservation rules, and hence a
    cleaner user interface.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        rule: GraphElementRule,
        variables: Set[Tuple[int, Type[_QNType]]],
        fixed_variables: Dict[int, Dict[Type[_QNType], Scalar]],
        argument_handler: RuleArgumentHandler,
        scoresheet: Callable[[bool], None],
    ) -> None:
        if not callable(rule):
            raise TypeError("rule has to be a callable!")
        self.__rule = rule
        (
            self.__check_rule_requirements,
            self.__create_rule_args,
        ) = argument_handler.register_rule(rule)
        self.__score_callback = scoresheet

        self.__var_string_to_data: Dict[str, Type[_QNType]] = {}
        self.__qns: Dict[Type[_QNType], Optional[Scalar]] = {}

        self.__initialize_variable_containers(variables, fixed_variables)

    @property
    def rule(self) -> Rule:
        return self.__rule

    def __initialize_variable_containers(
        self,
        variables: Set[Tuple[int, Type[_QNType]]],
        fixed_variables: Dict[int, Dict[Type[_QNType], Scalar]],
    ) -> None:
        """Fill the name decoding map.

        Also initialize the in and out particle lists. The variable names
        follow the scheme edge_id(delimiter)qn_name. This method creates a dict
        linking the var name to a list that consists of the particle list index
        and the qn name.
        """
        self.__qns.update(list(fixed_variables.values())[0])  # type: ignore
        for element_id, qn_type in variables:
            self.__var_string_to_data[
                _create_variable_string(element_id, qn_type)
            ] = qn_type
            self.__qns.update({qn_type: None})

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
        params = [(x, assignments.get(x, _unassigned)) for x in variables]
        missing = [name for (name, val) in params if val is _unassigned]
        if missing:
            return True

        self.__update_variable_lists(params)

        if not self.__check_rule_requirements(
            self.__qns,
        ):
            return True

        passed = self.__rule(*self.__create_rule_args(self.__qns))

        self.__score_callback(passed)

        return passed

    def __update_variable_lists(
        self,
        parameters: List[Tuple[str, Any]],
    ) -> None:
        for var_string, value in parameters:
            qn_type = self.__var_string_to_data[var_string]
            if qn_type in self.__qns:
                self.__qns[qn_type] = value  # type: ignore
            else:
                raise ValueError(
                    "The variable with name "
                    + qn_type.__name__
                    + "does not appear in the variable mapping!"
                )


class _ConservationRuleConstraintWrapper(Constraint):
    """Wrapper class of the python-constraint Constraint class.

    This allows a customized definition of conservation rules, and hence a
    cleaner user interface.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        rule: Rule,
        variables: _VariableContainer,
        argument_handler: RuleArgumentHandler,
        score_callback: Callable[[bool], None],
    ) -> None:
        if not callable(rule):
            raise TypeError("rule has to be a callable!")
        self.__rule = rule
        (
            self.__check_rule_requirements,
            self.__create_rule_args,
        ) = argument_handler.register_rule(rule)
        self.__score_callback = score_callback

        self.__var_string_to_data: Dict[
            str,
            Union[_EdgeVariableInfo, _NodeVariableInfo],
        ] = {}
        self.__in_edges_qns: Dict[int, GraphEdgePropertyMap] = {}
        self.__out_edges_qns: Dict[int, GraphEdgePropertyMap] = {}
        self.__node_qns: GraphNodePropertyMap = {}

        self.__initialize_variable_containers(variables)

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
            container: Dict[int, GraphEdgePropertyMap],
        ) -> None:
            container.update(fixed_variables)  # type: ignore
            for element_id, qn_type in variable_set:
                self.__var_string_to_data[
                    _create_variable_string(element_id, qn_type)
                ] = (element_id, qn_type)
                if element_id not in container:
                    container[element_id] = {}
                container[element_id].update({qn_type: None})  # type: ignore

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
            self.__node_qns[var_info[1]] = None  # type: ignore
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
        params = [(x, assignments.get(x, _unassigned)) for x in variables]
        missing = [name for (name, val) in params if val is _unassigned]
        if missing:
            return True

        self.__update_variable_lists(params)

        if not self.__check_rule_requirements(
            list(self.__in_edges_qns.values()),
            list(self.__out_edges_qns.values()),
            self.__node_qns,
        ):
            return True

        passed = self.__rule(
            *self.__create_rule_args(
                list(self.__in_edges_qns.values()),
                list(self.__out_edges_qns.values()),
                self.__node_qns,
            )
        )
        self.__score_callback(passed)
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
                    f"The variable with name {qn_type.__name__} and a graph "
                    f"element index of {index} does not appear in the variable "
                    f"mapping!"
                )
