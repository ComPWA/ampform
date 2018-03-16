from core.topology.graph import (get_edges_ingoing_to_node,
                                 get_edges_outgoing_to_node)


class ParticleStateTransitionGraphValidator():
    def __init__(self, graph):
        self.graph = graph
        self.node_conservation_laws = {}

    def assign_conservation_laws_to_all_nodes(self, conservation_laws,
                                              quantum_number_domains):
        for node_id in self.graph.nodes:
            self.assign_conservation_laws_to_node(
                node_id, conservation_laws, quantum_number_domains)

    def assign_conservation_laws_to_node(self, node_id, conservation_laws,
                                         quantum_number_domains):
        if node_id not in self.node_conservation_laws:
            self.node_conservation_laws[node_id] = (
                {'strict': [],
                 'non-strict': []
                 },
                {}
            )
        (cl, qnd) = self.node_conservation_laws[node_id]
        if 'strict' in conservation_laws:
            cl['strict'].extend(conservation_laws['strict'])
        if 'non-strict' in conservation_laws:
            cl['non-strict'].extend(conservation_laws['non-strict'])
        qnd.update(quantum_number_domains)

    def validate_graph(self):
        for node_id, conservation_laws in self.node_conservation_laws.items():
            for cons_law in conservation_laws:
                # get the needed qns for this conservation law
                # for all edges and the node
                var_containers = self.create_variable_containers(
                    node_id, cons_law)
                # check the requirements
                self.check_rule_requirements(cons_law, var_containers)
                # and run the rule check
                if not cons_law.check(var_containers[0],
                                      var_containers[1],
                                      var_containers[2]):
                    return False
        return True

    def create_variable_containers(self, node_id, cons_law):
        var_constainers = ()
        in_edges = get_edges_ingoing_to_node(self.graph, node_id)
        out_edges = get_edges_outgoing_to_node(self.graph, node_id)

        return var_constainers


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

        return variables

    def create_edge_variables(self, edge_ids, qn_dict):
        """
        Creates variables for the quantum numbers of the specified edges.

        Initial and final state edges just get a single domain value.
        Intermediate edges are initialized with the default domains of that
        quantum number.
        """
        variables = (set(), {})
        qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
        type_label = get_xml_label(XMLLabelConstants.Type)
        value_label = get_xml_label(XMLLabelConstants.Value)

        for edge_id in edge_ids:
                 edge_props = self.graph.edge_props[edge_id]
                edge_qns = edge_props[qns_label]
                for qn_name, qn_domain in qn_dict.items():
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
                        variables[1][(edge_id, qn_name)] = value
        return variables



    def check_rule_requirements(self, rule, var_containers):
        for (qn_name_list, cond_functor) in rule.get_qn_conditions():
            if not cond_functor.check(qn_name_list,
                                      var_containers[0],
                                      var_containers[1],
                                      var_containers[2]):
                raise ValueError("Error: "
                                 "quantum number condition << "
                                 + str(cond_functor.__class__) + " >> "
                                 + "of conservation law "
                                 + type(rule).__name__
                                 + " when looking for qns:\n"
                                 + str([x.name for x in qn_name_list]))
