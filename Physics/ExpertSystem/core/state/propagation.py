"""
This module is responsible for propagating the quantum numbers of the initial
and final state particles through a graphs (Propagator classes). Hence it
finds the allowed quantum numbers of the intermediate states.
The propagator classes (e.g. :class:`.CSPPropagator`) use the implemented
conservation rules of :mod:`.conservationrules`.
"""
from copy import deepcopy
from enum import Enum
from collections import OrderedDict
from numpy import arange

from constraint import (Problem, Constraint, Unassigned)

from core.topology.graph import (get_initial_state_edges,
                                 get_final_state_edges,
                                 get_edges_ingoing_to_node,
                                 get_edges_outgoing_to_node)
from core.state.conservationrules import AbstractRule
from core.state.particle import (get_xml_label, XMLLabelConstants,
                                 ParticleQuantumNumberNames,
                                 InteractionQuantumNumberNames,
                                 QNNameClassMapping,
                                 QNClassConverterMapping)


graph_element_types = Enum('GraphElementTypes', 'node edge')


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
        qn_name = ParticleQuantumNumberNames[split_name[2]]
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


class CSPPropagator():
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
        self.node_conservation_laws = {}
        self.variable_set = set()
        self.graph = graph
        self.problem = Problem()
        self.particle_variable_delimiter = "-*-"

    def find_solutions(self):
        self.initialize_contraints()
        solutions = self.problem.getSolutions()
        solution_graphs = self.apply_solutions_to_graph(solutions)
        return solution_graphs

    def assign_conservation_laws_to_all_nodes(self, conservation_laws,
                                              quantum_number_domains):
        for node_id in self.graph.nodes:
            self.assign_conservation_laws_to_node(
                node_id, conservation_laws, quantum_number_domains)

    def assign_conservation_laws_to_node(self, node_id, conservation_laws,
                                         quantum_number_domains):
        self.node_conservation_laws[node_id] = (
            conservation_laws, quantum_number_domains)

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
        for node_id, (cons_laws, qn_domains) in self.node_conservation_laws.items():
            new_cons_laws = [(x, True) for x in cons_laws['strict']]
            new_cons_laws.extend([(x, False) for x in cons_laws['non-strict']])
            for (cons_law, is_strict) in new_cons_laws:
                # from cons law and graph determine needed var lists
                qn_names = cons_law.get_required_qn_names()
                # create needed variables for edges/particle qns
                part_qn_dict = self.prepare_qns(
                    qn_names, qn_domains, ParticleQuantumNumberNames)

                variable_mapping = {}
                in_edges = get_edges_ingoing_to_node(self.graph, node_id)
                variable_mapping["ingoing"] = self.create_edge_variables(
                    in_edges, part_qn_dict)
                var_list = [key for key in variable_mapping["ingoing"]]
                out_edges = get_edges_outgoing_to_node(self.graph, node_id)
                variable_mapping["outgoing"] = self.create_edge_variables(
                    out_edges, part_qn_dict)
                var_list.extend([key for key in variable_mapping["outgoing"]])
                # now create variables for node/interaction qns
                int_qn_dict = self.prepare_qns(
                    qn_names, qn_domains, InteractionQuantumNumberNames)
                variable_mapping["interaction"] = self.create_node_variables(
                    node_id, int_qn_dict)
                var_list.extend(
                    [key for key in variable_mapping["interaction"]])
                # create constraint
                constraint = ConservationLawConstraintWrapper(
                    cons_law, variable_mapping,
                    self.particle_variable_delimiter)
                constraint.register_graph_node(node_id)
                if is_strict:
                    constraint.set_strict()
                self.problem.addConstraint(constraint, var_list)

    def prepare_qns(self, qn_names, qn_domains, type_to_filter):
        qn_names_used = qn_names
        if ParticleQuantumNumberNames.All in qn_names:
            qn_names_used = []
            for qnname in ParticleQuantumNumberNames:
                if qnname != ParticleQuantumNumberNames.All:
                    qn_names_used.append(qnname)

        part_qn_dict = {}
        for qn_name in qn_names_used:
            if isinstance(qn_name, type_to_filter):
                part_qn_dict[qn_name] = qn_domains[qn_name]
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
            domain_values = self.determine_domain(var_info, [], qn_domain)
            key = self.add_variable(var_info, domain_values)
            variables.add(key)
        return variables

    def create_edge_variables(self, edge_ids, qn_dict):
        """
        Creates variables for the quantum numbers of the specified edges.

        Initial and final state edges just get a single domain value.
        Intermediate edges are initialized with the default domains of that
        quantum number.
        """
        variables = set()
        qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)

        for edge_id in edge_ids:
            var_list = []
            edge_qns = []
            # fill edge qn list if its a initial or final state variable
            if (edge_id in get_initial_state_edges(self.graph) or
                    edge_id in get_final_state_edges(self.graph)):
                edge_props = self.graph.edge_props[edge_id]
                edge_qns = edge_props[qns_label]

            for qn_name, qn_domain in qn_dict.items():
                var_info = VariableInfo(graph_element_types.edge,
                                        edge_id,
                                        qn_name
                                        )
                domain_values = self.determine_domain(
                    var_info, edge_qns, qn_domain)
                if domain_values:
                    var_list.append((var_info, domain_values))

            # add all variables for this edge/particle
            for x in var_list:
                key = self.add_variable(x[0], x[1])
                variables.add(key)
        return variables

    def determine_domain(self, var_info, edge_qns, default_qn_domain):
        type_label = get_xml_label(XMLLabelConstants.Type)
        domain_values = []

        if edge_qns:
            converter = QNClassConverterMapping[
                QNNameClassMapping[var_info.qn_name]]
            domain_values = [converter.parse_from_dict(x)
                             for x in edge_qns if (
                x[type_label] == var_info.qn_name.name)]
        else:
            domain_values = default_qn_domain

        return domain_values

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
        return solution_graphs


def add_qn_to_graph_element(graph, var_info, value):
    # TODO: i guess i have to pack all that stuff in OrderdDicts...
    # because of the xmltodict module
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
        self.out_variable_set = variable_mapping["outgoing"]
        self.interaction_variable_set = variable_mapping["interaction"]
        self.name_delimiter = name_delimiter
        self.part_in = []
        self.part_out = []
        self.interaction_qns = {}
        self.variable_name_decoding_map = {}
        self.initialize_particle_lists()
        self.node_id = None
        self.non_conserved_scenarios = []
        self.is_strict = False

    def register_graph_node(self, node_id):
        self.node_id = node_id

    def set_strict(self):
        self.is_strict = True

    def __call__(self, variables, domains, assignments, forwardcheck=False,
                 _unassigned=Unassigned):
        params = [(x, assignments.get(x, _unassigned)) for x in variables]
        missing = [name for (name, val) in params if val is _unassigned]
        if missing:
            return True
        self.update_variable_lists(params)
        passed = self.rule.check(self.part_in, self.part_out,
                                 self.interaction_qns)
        if self.is_strict:
            return passed
        else:
            if not passed:
                self.non_conserved_scenarios.append(
                    (self.part_in, self.part_out, self.interaction_qns))
            return True

    def initialize_particle_lists(self):
        """
        Fill the name decoding map and initialize the in and out particle
        lists. The variable names follow the scheme edge_id(delimiter)qn_name.
        This method creates a dict linking the var name to a list that consists
        of the particle list index and the qn name
        """
        self.initialize_particle_list(self.in_variable_set, self.part_in)
        self.initialize_particle_list(self.out_variable_set, self.part_out)
        # and now interaction node variables
        for var_name in self.interaction_variable_set:
            var_info = decode_variable_name(
                var_name, self.name_delimiter)
            self.interaction_qns[var_info.qn_name] = {}
            self.variable_name_decoding_map[var_name] = (
                0, var_info.qn_name)

    def initialize_particle_list(self, variable_set, list_to_init):
        temp_var_dict = {}
        for var_name in variable_set:
            var_info = decode_variable_name(
                var_name, self.name_delimiter)
            if var_info.element_id not in temp_var_dict:
                temp_var_dict[var_info.element_id] = {
                    var_name: var_info.qn_name}
            else:
                temp_var_dict[var_info.element_id][var_name] = var_info.qn_name

        for key, value in temp_var_dict.items():
            index = len(list_to_init)
            list_to_init.append({})
            for var_name, qn_name in value.items():
                self.variable_name_decoding_map[var_name] = (
                    index, qn_name)

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
