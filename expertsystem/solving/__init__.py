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
from dataclasses import dataclass, field, fields, replace
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from constraint import (
    BacktrackingSolver,
    Constraint,
    Problem,
    Unassigned,
    Variable,
)

from expertsystem.data import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    NodeQuantumNumber,
    NodeQuantumNumbers,
    Parity,
    Particle,
    ParticleCollection,
    ParticleWithSpin,
    Spin,
)
from expertsystem.nested_dicts import (
    InteractionQuantumNumberNames,
    StateQuantumNumberNames,
    edge_qn_to_enum,
)
from expertsystem.solving.conservation_rules import Rule
from expertsystem.state.properties import get_particle_property
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


@dataclass(frozen=True)
class QuantumNumberSolution:
    node_quantum_numbers: Dict[
        int, Dict[Type[NodeQuantumNumber], Union[int, float]]
    ] = field(default_factory=lambda: defaultdict(dict))
    edge_quantum_numbers: Dict[
        int, Dict[Type[EdgeQuantumNumber], Union[int, float]]
    ] = field(default_factory=lambda: defaultdict(dict))


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


def _init_class(
    class_type: Type, props: Dict[Enum, Any], key_mapping: Optional[dict]
) -> object:
    return class_type(
        **{
            class_field.name: _extract_value(
                props, class_field.type, key_mapping
            )
            for class_field in fields(class_type)
            if not _is_optional(class_field.type)
            or (
                key_mapping
                and key_mapping[class_field.type.__args__[0]] in props
            )
            or class_field.type.__args__[0] in props
        }
    )


def _extract_value(
    props: Dict[Enum, Any], obj_type: Any, key_mapping: Optional[dict]
) -> Any:
    if _is_optional(obj_type):
        obj_type = obj_type.__args__[0]

    qn_name = obj_type.__name__
    if key_mapping and key_mapping[obj_type] in props:
        value = props[key_mapping[obj_type]]
    else:
        value = props[obj_type]
    if "projection" in qn_name and isinstance(value, Spin):
        value = value.projection
    elif "magnitude" in qn_name and isinstance(value, Spin):
        value = value.magnitude

    if (
        "__supertype__" in obj_type.__dict__
        and obj_type.__supertype__ == Parity
    ):
        return obj_type.__supertype__(value)
    return obj_type(value)


def _check_arg_requirements(
    class_type: Type[Any], props: Dict[Enum, Any], key_mapping: Optional[dict]
) -> bool:
    if "__dataclass_fields__" in class_type.__dict__:
        return all(
            [
                bool(
                    key_mapping[class_field.type] in props
                    if key_mapping and class_field.type not in props
                    else class_field.type in props
                )
                for class_field in fields(class_type)
                if not _is_optional(class_field.type)
            ]
        )

    if key_mapping:
        return key_mapping[class_type] in props or class_type in props

    return class_type in props


def _check_requirements(
    rule: Rule,
    in_edge_props: List[Dict[Enum, Any]],
    out_edge_props: List[Dict[Enum, Any]],
    node_props: dict,
    key_mapping: Optional[dict] = None,
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
            if not _check_arg_requirements(arg_type, props, key_mapping):  # type: ignore
                return False
        else:
            if not all(
                [
                    _check_arg_requirements(
                        arg_type.__args__[0], x, key_mapping
                    )
                    for x in props
                ]
            ):
                return False
        arg_counter += 1

    return True


def _create_rule_edge_arg(
    input_type: Type[Any],
    edge_props: List[Dict[Enum, Any]],
    key_mapping: Optional[dict],
) -> List[Any]:
    # pylint: disable=unidiomatic-typecheck
    if not isinstance(edge_props, (list, tuple)):
        raise TypeError("edge_props are incompatible...")
    if not (type(input_type) is list or type(input_type) is tuple):
        raise TypeError("input type is incompatible...")
    in_list_type = input_type[0]

    if "__dataclass_fields__" in in_list_type.__dict__:
        # its a composite type -> create new class type here
        return [_init_class(in_list_type, x, key_mapping) for x in edge_props]
    return [_extract_value(x, in_list_type, key_mapping) for x in edge_props]


def _create_rule_node_arg(
    input_type: Type[Any],
    node_props: Dict[Enum, Any],
    key_mapping: Optional[dict],
) -> Any:
    # pylint: disable=unidiomatic-typecheck
    if isinstance(node_props, (list, tuple)):
        raise TypeError("node_props is incompatible...")
    if type(input_type) is list or type(input_type) is tuple:
        raise TypeError("input type is incompatible...")

    if "__dataclass_fields__" in input_type.__dict__:
        # its a composite type -> create new class type here
        return _init_class(input_type, node_props, key_mapping)
    return _extract_value(node_props, input_type, key_mapping)


def _create_rule_args(
    rule: Rule,
    in_edge_props: List[Dict[Enum, Any]],
    out_edge_props: List[Dict[Enum, Any]],
    node_props: Dict[Enum, Any],
    key_mapping: Optional[dict] = None,
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
                    key_mapping,
                )
            )
        else:
            args.append(
                _create_rule_edge_arg(
                    arg_type.__args__,
                    ordered_props[arg_counter],  # type: ignore
                    key_mapping,
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
            input_type.__origin__ is list or input_type.__origin__ is tuple
        ):
            class_type = input_type.__args__[0]

        if "__dataclass_fields__" in class_type.__dict__:
            for class_field in fields(class_type):
                qn_set.add(
                    class_field.type.__args__[0]
                    if _is_optional(class_field.type)
                    else class_field.type
                )
        else:
            qn_set.add(class_type)
        arg_counter += 1

    return (required_edge_qns, required_node_qns)


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


def _merge_solutions_with_graph(
    solutions: List[QuantumNumberSolution],
    graph: StateTransitionGraph[ParticleWithSpin],
    allowed_particles: ParticleCollection,
) -> List[StateTransitionGraph[ParticleWithSpin]]:
    initialized_graphs = []

    logging.debug("merging solutions with graph...")
    intermediate_edges = graph.get_intermediate_state_edges()
    for solution in solutions:
        temp_graph = deepcopy(graph)
        for node_id in temp_graph.nodes:
            if node_id in solution.node_quantum_numbers:
                temp_graph.node_props[node_id] = solution.node_quantum_numbers[
                    node_id
                ]

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
                    temp_graph = deepcopy(current_new_graph)
                    temp_graph.edge_props[int_edge_id] = particle_edge
                    new_graphs_temp.append(temp_graph)
            current_new_graphs = new_graphs_temp

        initialized_graphs.extend(current_new_graphs)

    return initialized_graphs


def __get_particle_candidates_for_state(
    state: Dict[Type[EdgeQuantumNumber], Union[int, float]],
    allowed_particles: ParticleCollection,
) -> List[ParticleWithSpin]:
    particle_edges: List[ParticleWithSpin] = []

    for allowed_particle in allowed_particles:
        if __check_qns_equal(state, allowed_particle):
            # temp_particle = deepcopy(allowed_particle)
            particle_edges.append(
                (allowed_particle, state[EdgeQuantumNumbers.spin_projection])
            )

    return particle_edges


def __check_qns_equal(
    state: Dict[Type[EdgeQuantumNumber], Union[int, float]], particle: Particle
) -> bool:
    # This function assumes the attribute names of Particle and the quantum
    # numbers defined by new type match
    changes_dict: Dict[str, Union[int, float, Parity, Spin]] = {
        getattr(edge_qn, "__name__"): value
        for edge_qn, value in state.items()
        if "magnitude" not in getattr(edge_qn, "__name__")
        and "projection" not in getattr(edge_qn, "__name__")
    }

    if EdgeQuantumNumbers.isospin_magnitude in state:
        changes_dict["isospin"] = Spin(
            state[EdgeQuantumNumbers.isospin_magnitude],
            state[EdgeQuantumNumbers.isospin_projection],
        )
    if EdgeQuantumNumbers.spin_magnitude in state:
        changes_dict["spin"] = state[EdgeQuantumNumbers.spin_magnitude]
    return replace(particle, **changes_dict) == particle


def validate_fully_initialized_graph(
    graph: StateTransitionGraph[ParticleWithSpin],
    rules_per_node: Dict[int, Set[Rule]],
) -> Result:
    logging.debug("validating graph...")

    def _create_node_variables(
        node_id: int, qn_list: Set[Type[NodeQuantumNumber]]
    ) -> Dict[Type[NodeQuantumNumber], Union[int, float]]:
        """Create variables for the quantum numbers of the specified node."""
        variables = {}
        if node_id in graph.node_props:
            for qn_type in qn_list:
                if qn_type in graph.node_props[node_id]:
                    variables[qn_type] = graph.node_props[node_id][qn_type]
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
                    value = get_particle_property(edge_props, qn_type)
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

    def __init__(self, allowed_intermediate_particles: ParticleCollection):
        self.__graph = StateTransitionGraph[ParticleWithSpin]()
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
                edge_qns, node_qns = _get_required_qns(cons_law)

                in_edges = self.__graph.get_edges_ingoing_to_node(node_id)
                in_edge_vars = self.__create_edge_variables(
                    in_edges, edge_qns, graph_settings.edge_settings
                )
                variable_mapping["ingoing"] = in_edge_vars[0]
                variable_mapping["ingoing-fixed"] = in_edge_vars[1]
                var_list = list(variable_mapping["ingoing"])

                out_edges = self.__graph.get_edges_outgoing_from_node(node_id)
                out_edge_vars = self.__create_edge_variables(
                    out_edges, edge_qns, graph_settings.edge_settings
                )
                variable_mapping["outgoing"] = out_edge_vars[0]
                variable_mapping["outgoing-fixed"] = out_edge_vars[1]
                var_list.extend(list(variable_mapping["outgoing"]))

                # now create variables for node/interaction qns
                int_node_vars = self.__create_node_variables(
                    node_id,
                    node_qns,
                    graph_settings.node_settings,
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
        qn_list: Set[Type[NodeQuantumNumber]],
        node_settings: Dict[int, NodeSettings],
    ) -> Tuple[Set[str], Dict[Type[NodeQuantumNumber], Union[int, float]],]:
        """Create variables for the quantum numbers of the specified node.

        If a quantum number is already defined for a node, then a fixed
        variable is created, which cannot be changed by the csp solver.
        Otherwise the node is initialized with the specified domain of that
        quantum number.
        """
        variables: Tuple[
            Set[str],
            Dict[Type[NodeQuantumNumber], Union[int, float]],
        ] = (
            set(),
            dict(),
        )

        if node_id in self.__graph.node_props:
            node_props = self.__graph.node_props[node_id]
            for qn_type in qn_list:
                if qn_type in node_props and node_props[qn_type] is not None:
                    variables[1].update({qn_type: node_props[qn_type]})
        else:
            for qn_type in qn_list:
                qn_name = edge_qn_to_enum[qn_type]
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
        qn_list: Set[Type[EdgeQuantumNumber]],
        edge_settings: Dict[int, EdgeSettings],
    ) -> Tuple[
        Set[str], Dict[int, Dict[Type[EdgeQuantumNumber], Union[int, float]]]
    ]:
        """Create variables for the quantum numbers of the specified edges.

        If a quantum number is already defined for an edge, then a fixed
        variable is created, which cannot be changed by the csp solver. This is
        the case for initial and final state edges. Otherwise the edges are
        initialized with the specified domains of that quantum number.
        """
        variables: Tuple[
            Set[str],
            Dict[int, Dict[Type[EdgeQuantumNumber], Union[int, float]]],
        ] = (
            set(),
            dict(),
        )
        for edge_id in edge_ids:
            variables[1][edge_id] = {}
            if edge_id in self.__graph.edge_props:
                edge_props = self.__graph.edge_props[edge_id]
                for qn_type in qn_list:
                    value = get_particle_property(edge_props, qn_type)
                    if value is not None:
                        variables[1][edge_id].update({qn_type: value})
            else:
                for qn_type in qn_list:
                    qn_name = edge_qn_to_enum[qn_type]
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

    def __convert_solution_keys(
        self, solutions: List[Dict[str, Union[int, float]]]
    ) -> List[QuantumNumberSolution]:
        """Convert keys of CSP solutions from string to quantum number types."""
        initial_edges = self.__graph.get_initial_state_edges()
        final_edges = self.__graph.get_final_state_edges()

        def get_qn(
            quantum_number_enum: Enum,
            value: Any,
        ) -> Dict[Any, Any]:
            for qn_type, qn_enum in edge_qn_to_enum.items():
                if qn_enum == quantum_number_enum:
                    if quantum_number_enum is StateQuantumNumberNames.Spin:
                        return {
                            EdgeQuantumNumbers.spin_magnitude: value.magnitude,
                            EdgeQuantumNumbers.spin_projection: value.projection,
                        }
                    if quantum_number_enum is StateQuantumNumberNames.IsoSpin:
                        return {
                            EdgeQuantumNumbers.isospin_magnitude: value.magnitude,
                            EdgeQuantumNumbers.isospin_projection: value.projection,
                        }
                    if quantum_number_enum is InteractionQuantumNumberNames.S:
                        return {
                            NodeQuantumNumbers.s_magnitude: value.magnitude,
                            NodeQuantumNumbers.s_projection: value.projection,
                        }
                    if quantum_number_enum is InteractionQuantumNumberNames.L:
                        return {
                            NodeQuantumNumbers.l_magnitude: value.magnitude,
                            NodeQuantumNumbers.l_projection: value.projection,
                        }
                    return {qn_type: value}
            raise ValueError(
                f"Enum {quantum_number_enum} not found in mapping"
            )

        converted_solutions = list()
        for solution in solutions:
            qn_solution = QuantumNumberSolution()
            for var_name, value in solution.items():
                var_info = _decode_variable_name(
                    var_name, self.__particle_variable_delimiter
                )
                ele_id = var_info.element_id

                if var_info.graph_element_type is _GraphElementTypes.edge:
                    if ele_id in initial_edges or ele_id in final_edges:
                        # skip if its an initial or final state edge
                        continue
                    qn_solution.edge_quantum_numbers[ele_id].update(
                        get_qn(var_info.qn_name, value)
                    )
                else:
                    qn_solution.node_quantum_numbers[ele_id].update(
                        get_qn(var_info.qn_name, value)
                    )
            converted_solutions.append(qn_solution)

        return converted_solutions


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
        self.fixed_interaction_variable_dict = variable_mapping[
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
        self.interaction_qns.update(
            {
                edge_qn_to_enum[k]: v
                for k, v in self.fixed_interaction_variable_dict.items()
            }
        )

    def initialize_particle_list(
        self,
        variable_set: Sequence[str],
        fixed_variables: Dict[
            int, Dict[Type[EdgeQuantumNumber], Union[int, float]]
        ],
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

        for edge_id, var_dict in fixed_variables.items():
            if edge_id not in temp_var_dict:
                temp_var_dict[edge_id] = {"fixed-vars": var_dict}
            else:
                if "fixed-vars" not in temp_var_dict[edge_id]:
                    temp_var_dict[edge_id]["fixed-vars"] = var_dict

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
                for qn_type, qn_value in value["fixed-vars"].items():
                    list_to_init[-1][qn_type] = qn_value

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
            self.rule,
            self.part_in,
            self.part_out,
            self.interaction_qns,
            edge_qn_to_enum,
        ):
            self.conditions_never_met = True
            return True

        passed = self.rule(
            *_create_rule_args(
                self.rule,
                self.part_in,
                self.part_out,
                self.interaction_qns,
                edge_qn_to_enum,
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
