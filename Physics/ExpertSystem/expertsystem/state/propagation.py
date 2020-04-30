"""
This module is responsible for propagating the quantum numbers of the initial
and final state particles through a graphs (Propagator classes). Hence it
finds the allowed quantum numbers of the intermediate states.
The propagator classes (e.g. :class:`.CSPPropagator`) use the implemented
conservation rules of :mod:`.conservationrules`.
"""
from copy import deepcopy
from collections import defaultdict
from enum import Enum
from abc import ABC, abstractmethod
import logging

from tools.progress.bar import IncrementalBar

from expertsystem.solvers.constraint import (Problem,
                                             Constraint,
                                             Unassigned,
                                             BacktrackingSolver)

from expertsystem.topology.graph import (get_initial_state_edges,
                                         get_final_state_edges,
                                         get_edges_ingoing_to_node,
                                         get_edges_outgoing_to_node,
                                         get_intermediate_state_edges)
from expertsystem.state.conservationrules import AbstractRule
from expertsystem.state.particle import (get_xml_label, XMLLabelConstants,
                                         StateQuantumNumberNames,
                                         InteractionQuantumNumberNames,
                                         ParticlePropertyNames,
                                         get_particle_property,
                                         QNNameClassMapping,
                                         QNClassConverterMapping,
                                         initialize_graphs_with_particles)


graph_element_types = Enum('GraphElementTypes', 'node edge')


InteractionTypes = Enum('InteractionTypes', 'Undefined Strong EM Weak')


class InteractionNodeSettings:
    '''
    Container class for the interaction settings, which can be assigned to each
    node of a state transition graph. Hence these settings contain the complete
    configuration information which is required for the solution finding, e.g:
        - list of strict conservation laws
        - list of non-strict conservation laws
        - list of quantum number domains
        - strength scale parameter (higher value means stronger force)
    '''

    def __init__(self):
        self.strict_conservation_laws = []
        self.nonstrict_conservation_laws = []
        self.qn_domains = []
        self.interaction_strength = 1.0

    def __repr__(self):
        return_string = "strict conservation laws:\n" + \
            str(self.strict_conservation_laws)
        return_string += "\nnon-strict conservation laws:\n" + \
            str(self.nonstrict_conservation_laws)
        return_string += "\nquantum number domains:\n" + \
            str(self.qn_domains)
        return_string += "\ninteraction strength: " + \
            str(self.interaction_strength) + "\n"

        return return_string


class AbstractPropagator(ABC):
    def __init__(self, graph):
        self.node_settings = {}
        self.node_non_satisfied_laws = defaultdict(list)
        self.node_postponed_conservation_laws = defaultdict(list)
        self.allowed_intermediate_particles = []
        self.graph = graph

    @abstractmethod
    def find_solutions(self):
        pass

    def get_non_satisfied_conservation_laws(self):
        return self.node_non_satisfied_laws

    def assign_settings_to_all_nodes(self, interaction_settings):
        for node_id in self.graph.nodes:
            self.node_settings[node_id] = interaction_settings

    def assign_settings_to_node(self, node_id, interaction_settings):
        self.node_settings[node_id] = interaction_settings

    def set_allowed_intermediate_particles(self,
                                           allowed_intermediate_particles):
        self.allowed_intermediate_particles = allowed_intermediate_particles


class FullPropagator():
    def __init__(self, graph):
        if len(get_intermediate_state_edges(graph)) > 0:
            self.propagator = CSPPropagator(graph)
            logging.info("using CSP propagator")
        else:
            self.propagator = ParticleStateTransitionGraphValidator(graph)

    def assign_settings_to_all_nodes(self, interaction_settings):
        self.propagator.assign_settings_to_all_nodes(interaction_settings)

    def assign_settings_to_node(self, node_id, interaction_settings):
        self.propagator.assign_settings_to_node(node_id,
                                                interaction_settings)

    def set_allowed_intermediate_particles(self,
                                           allowed_intermediate_particles):
        self.propagator.allowed_intermediate_particles = allowed_intermediate_particles

    def get_non_satisfied_conservation_laws(self):
        return self.propagator.node_non_satisfied_laws

    def find_solutions(self):
        solutions = self.propagator.find_solutions()
        logging.info("Number of solutions after propagator: " +
                     str(len(solutions)))
        if not solutions:
            violated_rules = self.propagator.node_non_satisfied_laws
            logging.info("violated rules: " + str(violated_rules))
            logging.debug(self.propagator.graph)

        full_particle_graphs = initialize_graphs_with_particles(
            solutions, self.propagator.allowed_intermediate_particles)
        logging.info("Number of fully initialized graphs: " +
                     str(len(full_particle_graphs)))

        if (not isinstance(self.propagator, ParticleStateTransitionGraphValidator)
                and len(self.propagator.node_postponed_conservation_laws) > 0):
            logging.info("validating graphs")
            temp_solution_graphs = full_particle_graphs
            full_particle_graphs = []
            violated_rules = []
            for graph in temp_solution_graphs:
                validator = ParticleStateTransitionGraphValidator(graph)
                postponed_rules = self.propagator.node_postponed_conservation_laws
                for node_id, cons_laws in postponed_rules.items():
                    validator.assign_settings_to_node(
                        node_id, cons_laws)
                full_particle_graphs.extend(validator.find_solutions())
                violated_rules.append(validator.node_non_satisfied_laws)

            logging.info("final number of solutions: " +
                         str(len(full_particle_graphs)))
            if len(full_particle_graphs) == 0:
                logging.info("violated rules: " + str(violated_rules))

        return full_particle_graphs


class ParticleStateTransitionGraphValidator(AbstractPropagator):
    def __init__(self, graph):
        super().__init__(graph)

    def find_solutions(self):
        logging.debug("validating graph...")
        for node_id, cons_laws in self.node_settings.items():
            for (cons_law, is_strict) in cons_laws:
                # get the needed qns for this conservation law
                # for all edges and the node
                var_containers = self.create_variable_containers(
                    node_id, cons_law)
                # check the requirements
                if self.check_rule_requirements(cons_law, var_containers):
                    # and run the rule check
                    if not cons_law.check(var_containers[0],
                                          var_containers[1],
                                          var_containers[2]):
                        self.node_non_satisfied_laws[node_id].append(cons_law)
                else:
                    if node_id not in self.node_postponed_conservation_laws:
                        self.node_postponed_conservation_laws[node_id] = []
                    self.node_postponed_conservation_laws[node_id].append(
                        cons_law)
        if len(self.node_non_satisfied_laws) > 0:
            return []
        return [self.graph]

    def create_variable_containers(self, node_id, cons_law):
        in_edges = get_edges_ingoing_to_node(self.graph, node_id)
        out_edges = get_edges_outgoing_to_node(self.graph, node_id)

        qn_names = cons_law.get_required_qn_names()
        qn_list = self.prepare_qns(qn_names, (StateQuantumNumberNames,
                                              ParticlePropertyNames))
        in_edges_vars = self.create_edge_variables(in_edges, qn_list)
        out_edges_vars = self.create_edge_variables(out_edges, qn_list)

        node_vars = self.create_node_variables(
            node_id, self.prepare_qns(qn_names, InteractionQuantumNumberNames))

        return (in_edges_vars, out_edges_vars, node_vars)

    def prepare_qns(self, qn_names, type_to_filter):
        return [x for x in qn_names if isinstance(x, type_to_filter)]

    def create_node_variables(self, node_id, qn_list):
        """
        Creates variables for the quantum numbers of the specified node.
        """
        variables = {}
        type_label = get_xml_label(XMLLabelConstants.Type)
        if node_id in self.graph.node_props:
            qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
            for qn_name in qn_list:
                converter = QNClassConverterMapping[
                    QNNameClassMapping[qn_name]]
                for node_qn in self.graph.node_props[node_id][qns_label]:
                    if (node_qn[type_label] == qn_name.name):
                        found_prop = node_qn
                        break
                if found_prop is not None:
                    value = converter.parse_from_dict(found_prop)
                    variables[qn_name] = value
        return variables

    def create_edge_variables(self, edge_ids, qn_list):
        """
        Creates variables for the quantum numbers of the specified edges.

        Initial and final state edges just get a single domain value.
        Intermediate edges are initialized with the default domains of that
        quantum number.
        """
        variables = []
        for edge_id in edge_ids:
            edge_vars = {}
            edge_props = self.graph.edge_props[edge_id]
            for qn_name in qn_list:
                value = get_particle_property(edge_props, qn_name)
                if value is not None:
                    edge_vars[qn_name] = value
            variables.append(edge_vars)
        return variables

    def check_rule_requirements(self, rule, var_containers):
        logging.debug("checking conditions for rule " + str(rule.__class__))
        for (qn_name_list, cond_functor) in rule.get_qn_conditions():
            logging.debug(str(cond_functor.__class__))
            logging.debug(qn_name_list)
            logging.debug(var_containers)
            if not cond_functor.check(qn_name_list,
                                      var_containers[0],
                                      var_containers[1],
                                      var_containers[2]):
                logging.debug("not satisfied!")
                return False
        logging.debug("all satisfied")
        return True


class VariableInfo():
    def __init__(self, graph_element_type, element_id, qn_name):
        self.graph_element_type = graph_element_type
        self.element_id = element_id
        self.qn_name = qn_name


def decode_variable_name(variable_name, delimiter):
    """
    Decodes the variable name (see ::func::`.encode_variable_name`)
    """
    split_name = variable_name.split(delimiter)
    if not len(split_name) == 3:
        raise ValueError(
            "The variable name does not follow the scheme: " + variable_name)
    qn_name = None
    graph_element_type = None
    element_id = int(split_name[1])
    if split_name[0] in graph_element_types.node.name:
        qn_name = InteractionQuantumNumberNames[split_name[2]]
        graph_element_type = graph_element_types.node
    else:
        qn_name = StateQuantumNumberNames[split_name[2]]
        graph_element_type = graph_element_types.edge

    return VariableInfo(graph_element_type, element_id, qn_name)


def encode_variable_name(variable_info, delimiter):
    """
    The variable names are encoded as a concatenated string of the form
    graph element type + delimiter + element id + delimiter + qn name
    The variable of type ::class::`.VariableInfo` and contains:
      - graph_element_type: is either "node" or "edge" (enum)
      - element_id: is the id of that node/edge
        (as it is defined in the graph)
      - qn_name: the quantum number name (enum)
    """
    if not isinstance(variable_info, VariableInfo):
        raise TypeError('parameter variable_info must be of type VariableInfo')
    var_name = variable_info.graph_element_type.name \
        + delimiter + str(variable_info.element_id) \
        + delimiter + variable_info.qn_name.name
    return var_name


class CSPPropagator(AbstractPropagator):
    """
    Quantum number propagator reducing the problem to a constraint
    satisfaction problem and solving this with the python-constraint module.

    The variables are the quantum numbers of particles/edges, but also some
    composite quantum numbers which are attributed to the interaction nodes
    (such as angular momentum L).
    The conservation laws serve as the constraints and are wrapped with a
    special class ::class::`.ConservationLawConstraintWrapper`.
    """

    def __init__(self, graph):
        self.variable_set = set()
        self.constraints = []
        solver = BacktrackingSolver(True)
        self.problem = Problem(solver)
        self.particle_variable_delimiter = "-*-"
        super().__init__(graph)

    def find_solutions(self):
        self.initialize_contraints()
        solutions = self.problem.getSolutions()
        solution_graphs = self.apply_solutions_to_graph(solutions)
        for constraint in self.constraints:
            if constraint.conditions_never_met:
                self.node_postponed_conservation_laws[constraint.node_id].append(
                    (constraint.rule, constraint.is_strict))
            if (sum(constraint.scenario_results) > 0 and
                    constraint.scenario_results[1] == 0):
                self.node_non_satisfied_laws[constraint.node_id].append(
                    constraint.rule)
        return solution_graphs

    def initialize_contraints(self):
        """
        loop over all nodes
        each node has a list of conservation laws
        for each conservation law, check which qn are required
        then for each conservation law we create a new contraint wrapper,
        which has to know all needed qn numbers/variables put them into a list
        this wrapper will also get a conservation law and and some more info
        where to chop this list of variables into particle informations groups again
        then this wrapper just passes the grouped variable information down to the
        conservation law check method and passes back the return value from that!
        that should be all!
        """
        for node_id, interaction_settings in self.node_settings.items():
            new_cons_laws = [
                (x, True)
                for x in interaction_settings.strict_conservation_laws]
            new_cons_laws.extend([
                (x, False)
                for x in interaction_settings.nonstrict_conservation_laws])
            for (cons_law, is_strict) in new_cons_laws:
                variable_mapping = {}
                # from cons law and graph determine needed var lists
                qn_names = cons_law.get_required_qn_names()

                # create needed variables for edges state qns
                part_qn_dict = self.prepare_qns(
                    qn_names, interaction_settings.qn_domains,
                    (StateQuantumNumberNames, ParticlePropertyNames)
                )
                in_edges = get_edges_ingoing_to_node(self.graph, node_id)

                in_edge_vars = self.create_edge_variables(
                    in_edges, part_qn_dict)
                variable_mapping["ingoing"] = in_edge_vars[0]
                variable_mapping["ingoing-fixed"] = in_edge_vars[1]
                var_list = [key for key in variable_mapping["ingoing"]]

                out_edges = get_edges_outgoing_to_node(self.graph, node_id)
                out_edge_vars = self.create_edge_variables(
                    out_edges, part_qn_dict)
                variable_mapping["outgoing"] = out_edge_vars[0]
                variable_mapping["outgoing-fixed"] = out_edge_vars[1]
                var_list.extend([key for key in variable_mapping["outgoing"]])

                # now create variables for node/interaction qns
                int_qn_dict = self.prepare_qns(
                    qn_names, interaction_settings.qn_domains,
                    InteractionQuantumNumberNames)
                variable_mapping["interaction"] = self.create_node_variables(
                    node_id, int_qn_dict)
                var_list.extend(
                    [key for key in variable_mapping["interaction"]])

                constraint = ConservationLawConstraintWrapper(
                    cons_law,
                    variable_mapping,
                    self.particle_variable_delimiter)
                constraint.register_graph_node(node_id)
                if is_strict:
                    constraint.set_strict()
                self.constraints.append(constraint)
                if var_list:
                    self.problem.addConstraint(constraint, var_list)
                else:
                    self.constraints[-1].conditions_never_met = True

    def prepare_qns(self, qn_names, qn_domains, type_to_filter):
        part_qn_dict = {}
        for qn_name in [x for x in qn_names if isinstance(x, type_to_filter)]:
            if qn_name in qn_domains:
                part_qn_dict[qn_name] = qn_domains[qn_name]
            else:
                part_qn_dict[qn_name] = []
        return part_qn_dict

    def create_node_variables(self, node_id, qn_dict):
        """
        Creates variables for the quantum numbers of the specified node.
        """
        variables = set()
        for qn_name, qn_domain in qn_dict.items():
            var_info = VariableInfo(graph_element_types.node,
                                    node_id,
                                    qn_name
                                    )
           # domain_values = self.determine_domain(var_info, [], )
            key = self.add_variable(var_info, qn_domain)
            variables.add(key)
        return variables

    def create_edge_variables(self, edge_ids, qn_dict):
        """
        Creates variables for the quantum numbers of the specified edges.

        Initial and final state edges just get a single domain value.
        Intermediate edges are initialized with the default domains of that
        quantum number.
        """
        variables = (set(), {})
        for edge_id in edge_ids:
            variables[1][edge_id] = []
            # if its a initial or final state edge we create a fixed var
            if (edge_id in get_initial_state_edges(self.graph) or
                    edge_id in get_final_state_edges(self.graph)):
                edge_props = self.graph.edge_props[edge_id]
                for qn_name, qn_domain in qn_dict.items():
                    value = get_particle_property(edge_props, qn_name)
                    if value is not None:
                        variables[1][edge_id].append((qn_name, value))
            else:
                for qn_name, qn_domain in qn_dict.items():
                    var_info = VariableInfo(graph_element_types.edge,
                                            edge_id,
                                            qn_name
                                            )
                    if qn_domain:
                        key = self.add_variable(var_info, qn_domain)
                        variables[0].add(key)
        return variables

    def add_variable(self, var_info, domain):
        key = encode_variable_name(var_info,
                                   self.particle_variable_delimiter)
        if key not in self.variable_set:
            self.variable_set.add(key)
            self.problem.addVariable(key, domain)
        return key

    def apply_solutions_to_graph(self, solutions):
        """
        Apply the CSP solutions to the graph instance.
        In other words attach the solution quantum numbers as properties to
        the edges.

        Args:
            solutions ([{constraint variables}]): solutions of the
                constraint (csp solving module).
        Returns:
            solution graphs ([:class:`.StateTransitionGraph`])
        """
        solution_graphs = []
        initial_edges = get_initial_state_edges(self.graph)
        final_edges = get_final_state_edges(self.graph)

        logging.debug("atempting to copy graph for " +
                      str(len(solutions)) + " solutions")
        bar = IncrementalBar('Copying solutions', max=len(solutions))
        for solution in solutions:
            graph_copy = deepcopy(self.graph)
            for var_name, value in solution.items():
                var_info = decode_variable_name(
                    var_name, self.particle_variable_delimiter)
                ele_id = var_info.element_id

                if var_info.graph_element_type is graph_element_types.edge:
                    if ele_id in initial_edges or ele_id in final_edges:
                        # skip if its an initial or final state edge
                        continue

                add_qn_to_graph_element(graph_copy, var_info, value)

            solution_graphs.append(graph_copy)
            bar.next()
        bar.finish()
        logging.debug("atempting to copy graph for " +
                      str(len(solutions)) + " solutions")
        return solution_graphs


def add_qn_to_graph_element(graph, var_info, value):
    # TODO: i guess i have to pack all that stuff in OrderdDicts...
    # because of the xmltodict module
    if value is None:
        return
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)

    element_id = var_info.element_id
    qn_name = var_info.qn_name
    graph_prop_dict = graph.edge_props
    if var_info.graph_element_type is graph_element_types.node:
        graph_prop_dict = graph.node_props

    converter = QNClassConverterMapping[
        QNNameClassMapping[qn_name]]

    if element_id not in graph_prop_dict:
        graph_prop_dict[element_id] = {qns_label: []}

    graph_prop_dict[element_id][qns_label].append(converter.convert_to_dict(
        qn_name, value))


class ConservationLawConstraintWrapper(Constraint):
    """
    Wrapper class of the python-constraint Constraint class, to allow a
    customized definition of conservation laws, and hence a cleaner
    user interface.
    """

    def __init__(self, rule, variable_mapping, name_delimiter):
        if not isinstance(rule, AbstractRule):
            raise TypeError("rule has to be of type AbstractRule!")
        self.rule = rule
        self.in_variable_set = variable_mapping["ingoing"]
        self.fixed_in_variables = variable_mapping["ingoing-fixed"]
        self.out_variable_set = variable_mapping["outgoing"]
        self.fixed_out_variables = variable_mapping["outgoing-fixed"]
        self.interaction_variable_set = variable_mapping["interaction"]
        self.name_delimiter = name_delimiter
        self.part_in = []
        self.part_out = []
        self.interaction_qns = {}
        self.variable_name_decoding_map = {}

        self.initialize_particle_lists()

        self.node_id = None
        self.conditions_never_met = False
        self.scenario_results = [0, 0]
        self.is_strict = False

    def register_graph_node(self, node_id):
        self.node_id = node_id

    def set_strict(self):
        self.is_strict = True

    def initialize_particle_lists(self):
        """
        Fill the name decoding map and initialize the in and out particle
        lists. The variable names follow the scheme edge_id(delimiter)qn_name.
        This method creates a dict linking the var name to a list that consists
        of the particle list index and the qn name
        """
        self.initialize_particle_list(self.in_variable_set,
                                      self.fixed_in_variables,
                                      self.part_in)
        self.initialize_particle_list(self.out_variable_set,
                                      self.fixed_out_variables,
                                      self.part_out)
        # and now interaction node variables
        for var_name in self.interaction_variable_set:
            var_info = decode_variable_name(
                var_name, self.name_delimiter)
            self.interaction_qns[var_info.qn_name] = {}
            self.variable_name_decoding_map[var_name] = (
                0, var_info.qn_name)

    def initialize_particle_list(self, variable_set, fixed_variables, list_to_init):
        temp_var_dict = {}
        for var_name in variable_set:
            var_info = decode_variable_name(
                var_name, self.name_delimiter)
            if var_info.element_id not in temp_var_dict:
                temp_var_dict[var_info.element_id] = {
                    'vars': {var_name: var_info.qn_name}}
            else:
                temp_var_dict[var_info.element_id]['vars'][var_name] = var_info.qn_name

        for edge_id, varlist in fixed_variables.items():
            if edge_id not in temp_var_dict:
                temp_var_dict[edge_id] = {
                    'fixed-vars': varlist}
            else:
                if 'fixed-vars' not in temp_var_dict[edge_id]:
                    temp_var_dict[edge_id]['fixed-vars'] = varlist

        for key, value in temp_var_dict.items():
            index = len(list_to_init)
            list_to_init.append({})
            if 'vars' in value:
                for var_name, qn_name in value['vars'].items():
                    self.variable_name_decoding_map[var_name] = (
                        index, qn_name)
            if 'fixed-vars' in value:
                for item in value['fixed-vars']:
                    list_to_init[-1][item[0]] = item[1]

    def __call__(self, variables, domains, assignments, forwardcheck=False,
                 _unassigned=Unassigned):
        if self.conditions_never_met:
            return True
        params = [(x, assignments.get(x, _unassigned)) for x in variables]
        missing = [name for (name, val) in params if val is _unassigned]
        if missing:
            return True
        self.update_variable_lists(params)
        if not self.verify_rule_conditions():
            self.conditions_never_met = True
            return True
        passed = self.rule.check(self.part_in, self.part_out,
                                 self.interaction_qns)
        if self.is_strict:
            if passed:
                self.scenario_results[1] += 1
            else:
                self.scenario_results[0] += 1
            return passed
        else:
            if not passed:
                self.scenario_results.append(
                    ((self.part_in, self.part_out, self.interaction_qns), False))
            return True

    def verify_rule_conditions(self):
        for (qn_name_list, cond_functor) in self.rule.get_qn_conditions():
            if not cond_functor.check(qn_name_list,
                                      self.part_in,
                                      self.part_out,
                                      self.interaction_qns):
                part_props = [x for x in qn_name_list if isinstance(
                    x, ParticlePropertyNames)]
                if part_props:
                    return False

                raise ValueError("Error: "
                                 "quantum number condition << "
                                 + str(cond_functor.__class__) + " >> "
                                 + "of conservation law "
                                 + type(self.rule).__name__
                                 + " when looking for qns:\n"
                                 + str([x.name for x in qn_name_list])
                                 + "\non node with id " + str(self.node_id))
        return True

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
                raise ValueError("The variable with name " +
                                 var_name +
                                 "does not appear in the variable mapping!")
