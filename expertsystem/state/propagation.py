# pylint: disable=too-many-lines

"""Functions to propagate quantum numbers through a `.StateTransitionGraph`.

This module is responsible for propagating the quantum numbers of the initial
and final state particles through a graphs (Propagator classes). Hence it finds
the allowed quantum numbers of the intermediate states. The propagator classes
(e.g. :class:`.CSPPropagator`) use the implemented conservation rules of
:mod:`.conservation_rules`.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import fields
from enum import Enum, auto

from expertsystem.data import EdgeQuantumNumbers, NodeQuantumNumbers, Parity
from expertsystem.solvers.constraint import (
    BacktrackingSolver,
    Constraint,
    Problem,
    Unassigned,
)
from expertsystem.state import particle
from expertsystem.state.conservation_rules import Rule
from expertsystem.state.particle import (
    InteractionQuantumNumberNames,
    ParticleDecayPropertyNames,
    ParticlePropertyNames,
    QNClassConverterMapping,
    QNNameClassMapping,
    StateQuantumNumberNames,
    get_interaction_property,
    get_particle_candidates_for_state,
    get_particle_property,
    initialize_graphs_with_particles,
)


class GraphElementTypes(Enum):
    """Types of graph elements in the form of an enumerate."""

    node = auto()
    edge = auto()


class InteractionTypes(Enum):
    """Types of interactions in the form of an enumerate."""

    Strong = auto()
    EM = auto()
    Weak = auto()


class InteractionNodeSettings:
    """Container class for the interaction settings.

    This class can be assigned to each node of a state transition graph. Hence,
    these settings contain the complete configuration information which is
    required for the solution finding, e.g:

      - list of conservation laws
      - list of quantum number domains
      - strength scale parameter (higher value means stronger force)
    """

    def __init__(self):
        self.conservation_laws = []
        self.qn_domains = {}
        self.interaction_strength = 1.0

    def __repr__(self):
        return_string = "conservation laws:\n" + str(self.conservation_laws)
        return_string += "\nquantum number domains:\n" + str(self.qn_domains)
        return_string += (
            "\ninteraction strength: " + str(self.interaction_strength) + "\n"
        )

        return return_string


class AbstractPropagator(ABC):
    """Abstract interface of a propagator."""

    def __init__(self, graph):
        self.node_settings = {}
        self.node_non_satisfied_laws = defaultdict(list)
        self.node_postponed_conservation_laws = defaultdict(list)
        self.graph = graph

    @abstractmethod
    def find_solutions(self):
        pass

    def get_non_satisfied_conservation_laws(self):
        return self.node_non_satisfied_laws

    def assign_settings_to_all_nodes(self, interaction_settings):
        for node_id in self.graph.nodes:
            self.assign_settings_to_node(node_id, interaction_settings)

    def assign_settings_to_node(self, node_id, interaction_settings):
        self.node_settings[node_id] = interaction_settings


class FullPropagator:
    """Hander that combines all propagator rules."""

    def __init__(
        self, graph, allowed_intermediate_particles, propagation_mode="fast"
    ):
        self.propagator = CSPPropagator(graph, allowed_intermediate_particles)
        self.propagation_mode = propagation_mode
        logging.debug("using CSP propagator")

        self.node_non_satisfied_laws = defaultdict(list)

    def assign_settings_to_all_nodes(self, interaction_settings):
        for node_id in self.propagator.graph.nodes:
            self.assign_settings_to_node(node_id, interaction_settings)

    def assign_settings_to_node(self, node_id, interaction_settings):
        if isinstance(self.propagator, ParticleStateTransitionGraphValidator):
            self.propagator.assign_settings_to_node(
                node_id, interaction_settings.conservation_laws
            )
        else:
            self.propagator.assign_settings_to_node(
                node_id, interaction_settings
            )

    def get_non_satisfied_conservation_laws(self):
        return self.node_non_satisfied_laws

    def find_solutions(self):
        # pylint: disable=too-many-branches, too-many-locals
        run_validation = False
        solutions = self.propagator.find_solutions()
        logging.debug(
            "Number of solutions after propagator: %s", len(solutions)
        )
        if solutions:
            if self.propagator.node_postponed_conservation_laws:
                run_validation = True
        else:
            self.node_non_satisfied_laws = deepcopy(
                self.propagator.node_non_satisfied_laws
            )
            # special case: no solutions were found but propagation mode is set
            # to "full". Then just rerun with postponed rules
            if (
                self.propagator.node_postponed_conservation_laws
                and self.propagation_mode == "full"
            ):
                # rerun the solution finding with postponed rules
                last_postponed_rules = []
                while set(
                    self.propagator.node_postponed_conservation_laws
                ) != set(last_postponed_rules):
                    graph = self.propagator.graph
                    allowed_intermediate_particles = (
                        self.propagator.allowed_intermediate_particles
                    )
                    last_postponed_rules = (
                        self.propagator.node_postponed_conservation_laws
                    )
                    interaction_settings = self.propagator.node_settings
                    for node_id, cons_laws in last_postponed_rules.items():
                        interaction_settings[
                            node_id
                        ].conservation_laws = cons_laws
                    self.propagator = CSPPropagator(
                        graph, allowed_intermediate_particles
                    )
                    for node_id, int_settings in interaction_settings.items():
                        self.assign_settings_to_node(node_id, int_settings)
                    self.propagator.find_solutions()
                    for (
                        key,
                        value,
                    ) in self.propagator.node_non_satisfied_laws.items():
                        self.node_non_satisfied_laws[key].extend(value)
            if self.propagator.node_postponed_conservation_laws:
                run_validation = True

        full_particle_graphs = initialize_graphs_with_particles(
            solutions, self.propagator.allowed_intermediate_particles
        )
        logging.debug(
            "Number of fully initialized graphs: %d", len(full_particle_graphs)
        )

        if run_validation:
            if not full_particle_graphs:
                full_particle_graphs = [
                    self.propagator.graph,
                ]
            temp_solution_graphs = full_particle_graphs
            full_particle_graphs = []
            additional_violated_laws = defaultdict(list)
            validation_failed = True
            for graph in temp_solution_graphs:
                validator = ParticleStateTransitionGraphValidator(graph)
                postponed_rules = (
                    self.propagator.node_postponed_conservation_laws
                )
                for node_id, cons_laws in postponed_rules.items():
                    validator.assign_settings_to_node(node_id, cons_laws)
                if not self.node_non_satisfied_laws:
                    full_particle_graphs.extend(validator.find_solutions())
                else:
                    validator.find_solutions()
                if (
                    not validator.node_non_satisfied_laws
                    and not self.node_non_satisfied_laws
                ):
                    validation_failed = False
                for (key, value) in validator.node_non_satisfied_laws.items():
                    additional_violated_laws[key].extend(value)
            if validation_failed:
                for (key, value) in additional_violated_laws.items():
                    self.node_non_satisfied_laws[key].extend(value)

        logging.debug(
            "Number of solutions after full propagator: %d",
            len(full_particle_graphs),
        )
        if len(full_particle_graphs) == 0:
            logging.debug(
                "violated rules: %s", str(self.node_non_satisfied_laws)
            )

        return full_particle_graphs


def _is_optional(class_field):
    if "__args__" in class_field.__dict__:
        if type(None) in class_field.__args__:
            return True
    return False


_qn_mapping = {
    EdgeQuantumNumbers.pid.__name__: ParticlePropertyNames.Pid,
    EdgeQuantumNumbers.mass.__name__: ParticlePropertyNames.Mass,
    EdgeQuantumNumbers.width.__name__: ParticleDecayPropertyNames.Width,
    EdgeQuantumNumbers.spin_magnitude.__name__: StateQuantumNumberNames.Spin,
    EdgeQuantumNumbers.spin_projection.__name__: StateQuantumNumberNames.Spin,
    EdgeQuantumNumbers.charge.__name__: StateQuantumNumberNames.Charge,
    EdgeQuantumNumbers.isospin_magnitude.__name__: StateQuantumNumberNames.IsoSpin,
    EdgeQuantumNumbers.isospin_projection.__name__: StateQuantumNumberNames.IsoSpin,
    EdgeQuantumNumbers.strangeness.__name__: StateQuantumNumberNames.Strangeness,
    EdgeQuantumNumbers.charmness.__name__: StateQuantumNumberNames.Charmness,
    EdgeQuantumNumbers.bottomness.__name__: StateQuantumNumberNames.Bottomness,
    EdgeQuantumNumbers.topness.__name__: StateQuantumNumberNames.Topness,
    EdgeQuantumNumbers.baryon_number.__name__: StateQuantumNumberNames.BaryonNumber,
    EdgeQuantumNumbers.electron_lepton_number.__name__: StateQuantumNumberNames.ElectronLN,
    EdgeQuantumNumbers.muon_lepton_number.__name__: StateQuantumNumberNames.MuonLN,
    EdgeQuantumNumbers.tau_lepton_number.__name__: StateQuantumNumberNames.TauLN,
    EdgeQuantumNumbers.parity.__name__: StateQuantumNumberNames.Parity,
    EdgeQuantumNumbers.c_parity.__name__: StateQuantumNumberNames.CParity,
    EdgeQuantumNumbers.g_parity.__name__: StateQuantumNumberNames.GParity,
    NodeQuantumNumbers.l_magnitude.__name__: InteractionQuantumNumberNames.L,
    NodeQuantumNumbers.l_projection.__name__: InteractionQuantumNumberNames.L,
    NodeQuantumNumbers.s_magnitude.__name__: InteractionQuantumNumberNames.S,
    NodeQuantumNumbers.s_projection.__name__: InteractionQuantumNumberNames.S,
    NodeQuantumNumbers.parity_prefactor.__name__: InteractionQuantumNumberNames.ParityPrefactor,
}


def _init_class(class_type, props):
    return class_type(
        **{
            class_field.name: _extract_value(props, class_field.type)
            for class_field in fields(class_type)
            if not _is_optional(class_field.type)
            or _qn_mapping[class_field.type.__args__[0].__name__] in props
        }
    )


def _extract_value(props, obj_type):
    if _is_optional(obj_type):
        obj_type = obj_type.__args__[0]

    qn_name = obj_type.__name__
    value = props[_qn_mapping[qn_name]]
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


def _check_arg_requirements(class_type, props):
    if "__dataclass_fields__" in class_type.__dict__:
        return all(
            [
                bool(_qn_mapping[class_field.type.__name__] in props)
                for class_field in fields(class_type)
                if not _is_optional(class_field.type)
            ]
        )

    return _qn_mapping[class_type.__name__] in props


def _check_requirements(rule, in_edge_props, out_edge_props, node_props):
    if not hasattr(rule.__class__.__call__, "__annotations__"):
        raise TypeError(
            f"missing type annotations for __call__ of rule {rule.__name__}"
        )
    arg_counter = 1
    rule_annotations = _remove_return_annotation(
        list(rule.__class__.__call__.__annotations__.values())
    )
    for arg_type, props in zip(
        rule_annotations, (in_edge_props, out_edge_props, node_props),
    ):
        if arg_counter == 3:
            if not _check_arg_requirements(arg_type, props):
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


def _create_rule_edge_arg(input_type, edge_props):
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


def _create_rule_node_arg(input_type, node_props):
    # pylint: disable=unidiomatic-typecheck
    if isinstance(node_props, (list, tuple)):
        raise TypeError("node_props is incompatible...")
    if type(input_type) is list or type(input_type) is tuple:
        raise TypeError("input type is incompatible...")

    if "__dataclass_fields__" in input_type.__dict__:
        # its a composite type -> create new class type here
        return _init_class(input_type, node_props)
    return _extract_value(node_props, input_type)


def _create_rule_args(rule, in_edge_props, out_edge_props, node_props) -> list:
    if not hasattr(rule.__class__.__call__, "__annotations__"):
        raise TypeError(
            f"missing type annotations for __call__ of rule {rule.__name__}"
        )
    args = []
    arg_counter = 0
    rule_annotations = list(rule.__class__.__call__.__annotations__.values())
    rule_annotations = _remove_return_annotation(rule_annotations)

    ordered_props = (in_edge_props, out_edge_props, node_props)
    for arg_type in rule_annotations:
        if arg_counter == 2:
            args.append(
                _create_rule_node_arg(arg_type, ordered_props[arg_counter])
            )
        else:
            args.append(
                _create_rule_edge_arg(
                    arg_type.__args__, ordered_props[arg_counter]
                )
            )
        arg_counter += 1
    if arg_counter == 2:
        # the rule does not use the third argument, just add None
        args.append(None)

    return args


def _remove_return_annotation(rule_annotations):
    # this assumes that all rules have also the return type defined
    return rule_annotations[:-1]


def _get_required_qn_names(rule):
    if not hasattr(rule.__class__.__call__, "__annotations__"):
        raise TypeError(
            f"missing type annotations for __call__ of rule {rule.__name__}"
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
                    _qn_mapping[
                        class_field.type.__args__[0].__name__
                        if _is_optional(class_field.type)
                        else class_field.type.__name__
                    ]
                )
        else:
            qn_set.add(_qn_mapping[class_type.__name__])

    return list(qn_set)


class ParticleStateTransitionGraphValidator(AbstractPropagator):
    """Validate particle states in a transition graph."""

    def find_solutions(self):
        logging.debug("validating graph...")
        for node_id, cons_laws in self.node_settings.items():
            for cons_law in cons_laws:
                # get the needed qns for this conservation law
                # for all edges and the node
                var_containers = self.__create_variable_containers(
                    node_id, cons_law
                )
                # check the requirements
                if _check_requirements(
                    cons_law,
                    var_containers[0],
                    var_containers[1],
                    var_containers[2],
                ):
                    # and run the rule check
                    if not cons_law(
                        *_create_rule_args(
                            cons_law,
                            var_containers[0],
                            var_containers[1],
                            var_containers[2],
                        )
                    ):
                        self.node_non_satisfied_laws[node_id].append(cons_law)
                else:
                    if node_id not in self.node_postponed_conservation_laws:
                        self.node_postponed_conservation_laws[node_id] = []
                    self.node_postponed_conservation_laws[node_id].append(
                        cons_law
                    )
        if len(self.node_non_satisfied_laws) > 0:
            return []
        if len(self.node_postponed_conservation_laws) > 0:
            return []
        return [self.graph]

    def __create_variable_containers(self, node_id, cons_law):
        in_edges = self.graph.get_edges_ingoing_to_node(node_id)
        out_edges = self.graph.get_edges_outgoing_from_node(node_id)

        qn_names = _get_required_qn_names(cons_law)
        qn_list = self.__prepare_qns(
            qn_names,
            (
                StateQuantumNumberNames,
                ParticlePropertyNames,
                ParticleDecayPropertyNames,
            ),
        )
        in_edges_vars = self.__create_edge_variables(in_edges, qn_list)
        out_edges_vars = self.__create_edge_variables(out_edges, qn_list)

        node_vars = self.__create_node_variables(
            node_id,
            self.__prepare_qns(qn_names, InteractionQuantumNumberNames),
        )

        return (in_edges_vars, out_edges_vars, node_vars)

    @staticmethod
    def __prepare_qns(qn_names, type_to_filter):
        return [x for x in qn_names if isinstance(x, type_to_filter)]

    def __create_node_variables(self, node_id, qn_list):
        """Create variables for the quantum numbers of the specified node."""
        variables = {}
        type_label = particle.Labels.Type.name
        if node_id in self.graph.node_props:
            qns_label = particle.Labels.QuantumNumber.name
            for qn_name in qn_list:
                converter = QNClassConverterMapping[
                    QNNameClassMapping[qn_name]
                ]
                found_prop = None
                for node_qn in self.graph.node_props[node_id][qns_label]:
                    if node_qn[type_label] == qn_name.name:
                        found_prop = node_qn
                        break
                if found_prop is not None:
                    value = converter.parse_from_dict(found_prop)
                    variables[qn_name] = value
        return variables

    def __create_edge_variables(self, edge_ids, qn_list):
        """Create variables for the quantum numbers of the specified edges.

        Initial and final state edges just get a single domain value.
        Intermediate edges are initialized with the default domains of that
        quantum number.
        """
        variables = []
        for edge_id in edge_ids:
            if edge_id in self.graph.edge_props:
                edge_vars = {}
                edge_props = self.graph.edge_props[edge_id]
                for qn_name in qn_list:
                    value = get_particle_property(edge_props, qn_name)
                    if value is not None:
                        edge_vars[qn_name] = value
                variables.append(edge_vars)
        return variables


class VariableInfo:
    """Data container for variable information."""

    # pylint: disable=too-few-public-methods

    def __init__(self, graph_element_type, element_id, qn_name):
        self.graph_element_type = graph_element_type
        self.element_id = element_id
        self.qn_name = qn_name


def decode_variable_name(variable_name, delimiter):
    """Decode the variable name.

    Also see `.encode_variable_name`.
    """
    split_name = variable_name.split(delimiter)
    if not len(split_name) == 3:
        raise ValueError(
            "The variable name does not follow the scheme: " + variable_name
        )
    qn_name = None
    graph_element_type = None
    element_id = int(split_name[1])
    if split_name[0] in GraphElementTypes.node.name:
        qn_name = InteractionQuantumNumberNames[split_name[2]]
        graph_element_type = GraphElementTypes.node
    else:
        qn_name = StateQuantumNumberNames[split_name[2]]
        graph_element_type = GraphElementTypes.edge

    return VariableInfo(graph_element_type, element_id, qn_name)


def encode_variable_name(variable_info, delimiter):
    """Encode variable name.

    The variable names are encoded as a concatenated string of the form graph
    element type + delimiter + element id + delimiter + qn name The variable of
    type :class:`.VariableInfo` and contains:

      - graph_element_type: is either "node" or "edge" (enum)
      - element_id: is the id of that node/edge (as it is defined in the graph)
      - qn_name: the quantum number name (enum)
    """
    if not isinstance(variable_info, VariableInfo):
        raise TypeError("parameter variable_info must be of type VariableInfo")
    var_name = (
        variable_info.graph_element_type.name
        + delimiter
        + str(variable_info.element_id)
        + delimiter
        + variable_info.qn_name.name
    )
    return var_name


class CSPPropagator(AbstractPropagator):
    """Quantum number propagator reducing the problem to a CSP.

    Quantum number propagator reducing the problem to a constraint satisfaction
    problem and solving this with the python-constraint module.

    The variables are the quantum numbers of particles/edges, but also some
    composite quantum numbers which are attributed to the interaction nodes
    (such as angular momentum :math:`L`). The conservation laws serve as the
    constraints and are wrapped with a special class
    :class:`.ConservationLawConstraintWrapper`.
    """

    def __init__(self, graph, allowed_intermediate_particles):
        self.variable_set = set()
        self.constraints = []
        solver = BacktrackingSolver(True)
        self.problem = Problem(solver)
        self.particle_variable_delimiter = "-*-"
        self.allowed_intermediate_particles = allowed_intermediate_particles
        super().__init__(graph)

    def find_solutions(self):
        self.initialize_constraints()
        solutions = self.problem.getSolutions()

        solution_graphs = self.apply_solutions_to_graph(solutions)
        for constraint in self.constraints:
            if (
                constraint.conditions_never_met
                or sum(constraint.scenario_results) == 0
            ):
                self.node_postponed_conservation_laws[
                    constraint.node_id
                ].append(constraint.rule)
            if (
                sum(constraint.scenario_results) > 0
                and constraint.scenario_results[1] == 0
            ):
                self.node_non_satisfied_laws[constraint.node_id].append(
                    constraint.rule
                )
        return solution_graphs

    def initialize_constraints(self):  # pylint: disable=too-many-locals
        """Initialize all of the constraints for this graph.

        For each interaction node a set of independent constraints/conservation
        laws are created. For each conservation law a new CSP wrapper is
        created. This wrapper needs all of the qn numbers/variables which enter
        or exit the node and play a role for this conservation law. Hence
        variables are also created within this method.
        """
        for node_id, interaction_settings in self.node_settings.items():
            new_cons_laws = interaction_settings.conservation_laws
            for cons_law in new_cons_laws:
                variable_mapping = {}
                # from cons law and graph determine needed var lists
                qn_names = _get_required_qn_names(cons_law)

                # create needed variables for edges state qns
                part_qn_dict = self.prepare_qns(
                    qn_names,
                    interaction_settings.qn_domains,
                    (
                        StateQuantumNumberNames,
                        ParticlePropertyNames,
                        ParticleDecayPropertyNames,
                    ),
                )
                in_edges = self.graph.get_edges_ingoing_to_node(node_id)

                in_edge_vars = self.create_edge_variables(
                    in_edges, part_qn_dict
                )
                variable_mapping["ingoing"] = in_edge_vars[0]
                variable_mapping["ingoing-fixed"] = in_edge_vars[1]
                var_list = list(variable_mapping["ingoing"])

                out_edges = self.graph.get_edges_outgoing_from_node(node_id)
                out_edge_vars = self.create_edge_variables(
                    out_edges, part_qn_dict
                )
                variable_mapping["outgoing"] = out_edge_vars[0]
                variable_mapping["outgoing-fixed"] = out_edge_vars[1]
                var_list.extend(list(variable_mapping["outgoing"]))

                # now create variables for node/interaction qns
                int_qn_dict = self.prepare_qns(
                    qn_names,
                    interaction_settings.qn_domains,
                    InteractionQuantumNumberNames,
                )
                int_node_vars = self.create_node_variables(
                    node_id, int_qn_dict
                )
                variable_mapping["interaction"] = int_node_vars[0]
                variable_mapping["interaction-fixed"] = int_node_vars[1]
                var_list.extend(list(variable_mapping["interaction"]))

                constraint = ConservationLawConstraintWrapper(
                    cons_law,
                    variable_mapping,
                    self.particle_variable_delimiter,
                )
                constraint.register_graph_node(node_id)
                self.constraints.append(constraint)
                if var_list:
                    self.problem.addConstraint(constraint, var_list)
                else:
                    self.constraints[-1].conditions_never_met = True

    @staticmethod
    def prepare_qns(qn_names, qn_domains, type_to_filter):
        part_qn_dict = {}
        for qn_name in [x for x in qn_names if isinstance(x, type_to_filter)]:
            if qn_name in qn_domains:
                part_qn_dict[qn_name] = qn_domains[qn_name]
            else:
                part_qn_dict[qn_name] = []
        return part_qn_dict

    def create_node_variables(self, node_id, qn_dict):
        """Create variables for the quantum numbers of the specified node.

        If a quantum number is already defined for a node, then a fixed
        variable is created, which cannot be changed by the csp solver.
        Otherwise the node is initialized with the specified domain of that
        quantum number.
        """
        variables = (set(), set())

        if node_id in self.graph.node_props:
            node_props = self.graph.node_props[node_id]
            for qn_name, qn_domain in qn_dict.items():
                value = get_interaction_property(node_props, qn_name)
                if value is not None:
                    variables[1].add((qn_name, value))
        else:
            for qn_name, qn_domain in qn_dict.items():
                var_info = VariableInfo(
                    GraphElementTypes.node, node_id, qn_name
                )
                # domain_values = self.determine_domain(var_info, [], )
                if qn_domain:
                    key = self.add_variable(var_info, qn_domain)
                    variables[0].add(key)
        return variables

    def create_edge_variables(self, edge_ids, qn_dict):
        """Create variables for the quantum numbers of the specified edges.

        If a quantum number is already defined for an edge, then a fixed
        variable is created, which cannot be changed by the csp solver. This is
        the case for initial and final state edges. Otherwise the edges are
        initialized with the specified domains of that quantum number.
        """
        variables = (set(), {})
        for edge_id in edge_ids:
            variables[1][edge_id] = []
            if edge_id in self.graph.edge_props:
                edge_props = self.graph.edge_props[edge_id]
                for qn_name, qn_domain in qn_dict.items():
                    value = get_particle_property(edge_props, qn_name)
                    if value is not None:
                        variables[1][edge_id].append((qn_name, value))
            else:
                for qn_name, qn_domain in qn_dict.items():
                    var_info = VariableInfo(
                        GraphElementTypes.edge, edge_id, qn_name
                    )
                    if qn_domain:
                        key = self.add_variable(var_info, qn_domain)
                        variables[0].add(key)
        return variables

    def add_variable(self, var_info, domain):
        key = encode_variable_name(var_info, self.particle_variable_delimiter)
        if key not in self.variable_set:
            self.variable_set.add(key)
            self.problem.addVariable(key, domain)
        return key

    def apply_solutions_to_graph(
        self, solutions
    ):  # pylint: disable=too-many-locals
        """Apply the CSP solutions to the graph instance.

        In other words attach the solution quantum numbers as properties to the
        edges. Also the solutions are filtered using the allowed intermediate
        particle list, to avoid large memory consumption.

        Args:
            solutions: list of solutions of the csp solver

        Returns:
            solution graphs ([:class:`.StateTransitionGraph`])
        """
        solution_graphs = []
        initial_edges = self.graph.get_initial_state_edges()
        final_edges = self.graph.get_final_state_edges()

        # logging.info("attempting to filter " + str(len(solutions)) +
        #             " solutions for allowed intermediate particles and"
        #             " create a copy graph")
        # bar = IncrementalBar('Filtering solutions', max=len(solutions))

        found_jps = set()

        for solution in solutions:
            graph_copy = deepcopy(self.graph)
            for var_name, value in solution.items():
                var_info = decode_variable_name(
                    var_name, self.particle_variable_delimiter
                )
                ele_id = var_info.element_id

                if var_info.graph_element_type is GraphElementTypes.edge:
                    if ele_id in initial_edges or ele_id in final_edges:
                        # skip if its an initial or final state edge
                        continue

                add_qn_to_graph_element(graph_copy, var_info, value)

            solution_valid = True
            if self.allowed_intermediate_particles:
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
                        str(spin.magnitude)
                        + ("-" if parity in (-1, -1.0) else "+")
                    )
                    # now do actual candidate finding
                    candidates = get_particle_candidates_for_state(
                        graph_copy.edge_props[int_edge_id],
                        self.allowed_intermediate_particles,
                    )
                    if not candidates:
                        solution_valid = False
                        break
            if solution_valid:
                solution_graphs.append(graph_copy)
            # bar.next()
        # bar.finish()
        if solutions and not solution_graphs:
            logging.warning(
                "No intermediate state particles match the found %d solutions!",
                len(solutions),
            )
            logging.warning("solution inter. state J^P: %s", str(found_jps))
        return solution_graphs


def add_qn_to_graph_element(graph, var_info, value):
    if value is None:
        return
    qns_label = particle.Labels.QuantumNumber.name

    element_id = var_info.element_id
    qn_name = var_info.qn_name
    graph_prop_dict = graph.edge_props
    if var_info.graph_element_type is GraphElementTypes.node:
        graph_prop_dict = graph.node_props

    converter = QNClassConverterMapping[QNNameClassMapping[qn_name]]

    if element_id not in graph_prop_dict:
        graph_prop_dict[element_id] = {qns_label: []}

    graph_prop_dict[element_id][qns_label].append(
        converter.convert_to_dict(qn_name, value)
    )


class ConservationLawConstraintWrapper(Constraint):
    """Wrapper class of the python-constraint Constraint class.

    This allows a customized definition of conservation laws, and hence a
    cleaner user interface.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, rule, variable_mapping, name_delimiter):
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
        self.part_in = []
        self.part_out = []
        self.interaction_qns = {}
        self.variable_name_decoding_map = {}

        self.initialize_particle_lists()

        self.node_id = None
        self.conditions_never_met = False
        self.scenario_results = [0, 0]

    def register_graph_node(self, node_id):
        self.node_id = node_id

    def initialize_particle_lists(self):
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
            var_info = decode_variable_name(var_name, self.name_delimiter)
            self.interaction_qns[var_info.qn_name] = {}
            self.variable_name_decoding_map[var_name] = (0, var_info.qn_name)
        for qn_name, value in self.fixed_interaction_variable_set:
            self.interaction_qns[qn_name] = value

    def initialize_particle_list(
        self, variable_set, fixed_variables, list_to_init
    ):
        temp_var_dict = {}
        for var_name in variable_set:
            var_info = decode_variable_name(var_name, self.name_delimiter)
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
        variables,
        domains,
        assignments,
        forwardcheck=False,
        _unassigned=Unassigned,
    ):
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

    def update_variable_lists(self, parameters):
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
