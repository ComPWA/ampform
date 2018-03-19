import logging

from core.topology.graph import (get_edges_ingoing_to_node,
                                 get_edges_outgoing_to_node)

from core.state.particle import (get_xml_label, XMLLabelConstants,
                                 StateQuantumNumberNames,
                                 InteractionQuantumNumberNames,
                                 ParticlePropertyNames,
                                 QNNameClassMapping,
                                 QNClassConverterMapping)


class ParticleStateTransitionGraphValidator():
    def __init__(self, graph):
        self.graph = graph
        self.node_conservation_laws = {}
        self.unusable_conservation_laws = {}

    def assign_conservation_laws_to_all_nodes(self, conservation_laws):
        for node_id in self.graph.nodes:
            self.assign_conservation_laws_to_node(
                node_id, conservation_laws)

    def assign_conservation_laws_to_node(self, node_id, conservation_laws):
        if node_id not in self.node_conservation_laws:
            self.node_conservation_laws[node_id] = []
        self.node_conservation_laws[node_id].extend(conservation_laws)

    def validate_graph(self):
        logging.debug("validating graph...")
        for node_id, conservation_laws in self.node_conservation_laws.items():
            for cons_law in conservation_laws:
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
                        logging.debug(
                            "removing graph because conservation law "
                            + str(cons_law.__class__))
                        logging.debug(
                            " failed on node with id " + str(node_id))
                        return False
                else:
                    if node_id not in self.unusable_conservation_laws:
                        self.unusable_conservation_laws[node_id] = []
                    self.unusable_conservation_laws[node_id].append(cons_law)

        return True

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
        qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
        type_label = get_xml_label(XMLLabelConstants.Type)
        value_label = get_xml_label(XMLLabelConstants.Value)

        for edge_id in edge_ids:
            edge_vars = {}
            edge_props = self.graph.edge_props[edge_id]
            edge_qns = edge_props[qns_label]
            for qn_name in qn_list:
                converter = QNClassConverterMapping[
                    QNNameClassMapping[qn_name]]
                found_prop = None
                if isinstance(qn_name, StateQuantumNumberNames):
                    for x in edge_qns:
                        if (x[type_label] == qn_name.name):
                            found_prop = x
                            break
                else:
                    for key, val in edge_props.items():
                        if (key == qn_name.name):
                            found_prop = {value_label: val}
                            break
                        if (key == 'Parameter' and
                                val[type_label] == qn_name.name):
                            # parameters have a seperate value tag
                            tagname = XMLLabelConstants.Value.name
                            found_prop = {value_label: val[tagname]}
                            break

                if found_prop is not None:
                    value = converter.parse_from_dict(found_prop)
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
