from collections import OrderedDict
import json
import logging

import xmltodict

from expertsystem.topology.graph import (StateTransitionGraph,
                                         get_initial_state_edges,
                                         get_final_state_edges,
                                         get_edges_ingoing_to_node,
                                         get_edges_outgoing_to_node)
from expertsystem.state.particle import (
    StateQuantumNumberNames, XMLLabelConstants, get_xml_label)


def group_graphs_same_initial_and_final(graphs):
    '''
    Each graph corresponds to a specific state transition amplitude.
    This function groups together graphs, which have the same initial and
    final state (including spin). This is needed to determine the coherency of
    the individual amplitude parts.

    Args:
        graphs ([:class:`.StateTransitionGraph`])
    Returns:
        graph groups ([[:class:`.StateTransitionGraph`]])
    '''
    graph_groups = dict()
    for graph in graphs:
        ise = get_final_state_edges(graph)
        fse = get_initial_state_edges(graph)
        ifsg = (tuple([json.dumps(graph.edge_props[x]) for x in ise]),
                tuple([json.dumps(graph.edge_props[x]) for x in fse]))
        if ifsg not in graph_groups:
            graph_groups[ifsg] = []
        graph_groups[ifsg].append(graph)

    graph_group_list = [graph_groups[x] for x in graph_groups.keys()]
    return graph_group_list


def get_helicity_from_edge_props(edge_props):
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
    type_label = get_xml_label(XMLLabelConstants.Type)
    spin_label = StateQuantumNumberNames.Spin
    proj_label = get_xml_label(XMLLabelConstants.Projection)
    for qn in edge_props[qns_label]:
        if qn[type_label] == spin_label.name:
            return qn[proj_label]
    print(edge_props[qns_label])
    raise ValueError("Could not find spin projection quantum number!")


def determine_attached_final_state_string(graph, edge_id):
    edge_ids = determine_attached_final_state(graph, edge_id)
    fs_string = ""
    for eid in edge_ids:
        fs_string += " " + str(eid)
    return fs_string[1:]


def determine_attached_final_state(graph, edge_id):
    '''
    Determines all final state particles of a graph, which are attached
    downward (forward in time) for a given edge (resembling the root)

    Args:
        graph (:class:`.StateTransitionGraph`)
        edge_id (int): id of the edge, which is taken as the root
    Returns:
        list of final state edge ids ([int])
    '''
    final_state_edge_ids = []
    all_final_state_edges = get_final_state_edges(graph)
    current_edges = [edge_id]
    while current_edges:
        temp_current_edges = current_edges
        current_edges = []
        for curr_edge in temp_current_edges:
            if curr_edge in all_final_state_edges:
                final_state_edge_ids.append(curr_edge)
            else:
                node_id = graph.edges[curr_edge].ending_node_id
                current_edges.extend(get_edges_outgoing_to_node(graph,
                                                                node_id))
    return final_state_edge_ids


def get_recoil_edge(graph, edge_id):
    '''
    Determines the id of the recoil edge for the specified edge of a graph.

    Args:
        graph (:class:`.StateTransitionGraph`)
        edge_id (int): id of the edge, for which the recoil partner is
            determined
    Returns:
        recoil edge id (int)
    '''
    node_id = graph.edges[edge_id].originating_node_id
    if node_id is None:
        return None
    outgoing_edges = get_edges_outgoing_to_node(graph, node_id)
    outgoing_edges.remove(edge_id)
    if len(outgoing_edges) != 1:
        raise ValueError("The node with id " + str(node_id) +
                         " has more than 2 outgoing edges \n" + str(graph))
    return outgoing_edges[0]


def get_final_state_edge_ids(graph, list_of_particle_names):
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph must be a StateTransitionGraph")
    name_label = get_xml_label(XMLLabelConstants.Name)
    fsp_names = {graph.edge_props[i][name_label]: i
                 for i in get_final_state_edges(graph)}
    edge_list = []
    for particle_name in list_of_particle_names:
        if particle_name in fsp_names:
            edge_list.append(fsp_names[particle_name])
    return edge_list


class HelicityDecayAmplitudeGeneratorXML():
    def __init__(self, graphs):
        self.particle_list = {}
        if len(graphs) <= 0:
            raise ValueError(
                "Number of solution graphs is not larger than zero!")
        self.graphs = graphs
        self.helicity_amplitudes = {}
        self.kinematics = {}
        self.generate_particle_list()
        self.generate_kinematics()
        self.generate_amplitude_info()

    def generate_particle_list(self):
        # create particle entries
        temp_particle_names = []
        particles = []
        for g in self.graphs:
            for edge_props in g.edge_props.values():
                par_name = edge_props[get_xml_label(XMLLabelConstants.Name)]
                if par_name not in temp_particle_names:
                    particles.append(edge_props)
                    temp_particle_names.append(par_name)
        self.particle_list = {'ParticleList': {'Particle': particles}}

    def generate_amplitude_info(self):
        graph_groups = group_graphs_same_initial_and_final(self.graphs)
        logging.debug("There are " + str(len(graph_groups)) + " graph groups")

        class_label = get_xml_label(XMLLabelConstants.Class)
        name_label = get_xml_label(XMLLabelConstants.Name)
        type_label = get_xml_label(XMLLabelConstants.Type)
        parameter_label = get_xml_label(XMLLabelConstants.Parameter)
        
        # for each graph group we create a coherent amplitude
        coherent_amplitudes = []
        for graph_group in graph_groups:
            seq_partial_decays = []
            for graph in graph_group:
                seq_partial_decays.append(
                    self.generate_sequential_decay(graph))

            # in each coherent amplitude we create a product of partial decays
            coherent_amp_name = "coherent_" + \
                str(graph_groups.index(graph_group))
            coherent_amplitudes.append({
                class_label: 'Coherent', name_label: coherent_amp_name,
                parameter_label: {class_label: "Double",
                                  type_label: "Strength",
                                  name_label: "strength_" + coherent_amp_name,
                                  'Value': 1, 'Fix': True},
                'Amplitude': seq_partial_decays
            })

        # now wrap it with an incoherent intensity
        incoherent_amp_name = "incoherent"
        self.helicity_amplitudes = {
            'Intensity': {
                class_label: "Incoherent", name_label: incoherent_amp_name,
                parameter_label: {class_label: "Double",
                                  type_label: "Strength",
                                  name_label: "strength_" +
                                  incoherent_amp_name,
                                  'Value': 1, 'Fix': True},
                'Intensity': coherent_amplitudes
            }
        }

    def generate_sequential_decay(self, graph):
        partial_decays = []
        for node_id in graph.nodes:
            partial_decays.append(self.generate_partial_decay(graph, node_id))

        seq_decay_amp_name = "asdfsadf"
        seq_decay_dict = {
            '@Class': "SequentialPartialAmplitude",
            '@Name': seq_decay_amp_name,
            'Parameter': [{'@Class': "Double", '@Type': "Magnitude",
                           '@Name': "Magnitude_" + seq_decay_amp_name,
                           'Value': 1.0, 'Fix': True},
                          {'@Class': "Double", '@Type': "Phase",
                           '@Name': "Phase_" + seq_decay_amp_name,
                           'Value': 0.0, 'Fix': True}],
            'PartialAmplitude': partial_decays
        }
        return seq_decay_dict

    def generate_partial_decay(self, graph, node_id):
        decay_products = []
        for out_edge_id in get_edges_outgoing_to_node(graph,
                                                      node_id):
            decay_products.append({
                '@Name': graph.edge_props[out_edge_id]['@Name'],
                '@FinalState': determine_attached_final_state_string(
                    graph,
                    out_edge_id),
                '@Helicity': get_helicity_from_edge_props(
                    graph.edge_props[out_edge_id])
            })

        in_edge_ids = get_edges_ingoing_to_node(graph, node_id)
        if len(in_edge_ids) != 1:
            raise ValueError(
                "This node does not represent a two body decay!")
        dec_part = graph.edge_props[in_edge_ids[0]]
        dec_part_name = dec_part['@Name']

        recoil_edge_id = get_recoil_edge(graph, in_edge_ids[0])
        recoil_system_dict = {}
        if recoil_edge_id is not None:
            recoil_system_dict['RecoilSystem'] = {
                '@FinalState':
                determine_attached_final_state_string(graph, recoil_edge_id)
            },

        partial_decay_dict = {
            'Parameter': [{'@Class': "Double", '@Type': "Magnitude",
                           '@Name': "Magnitude_" + dec_part_name,
                           'Value': 1.0, 'Fix': True},
                          {'@Class': "Double", '@Type': "Phase",
                           '@Name': "Phase_" + dec_part_name,
                           'Value': 0.0, 'Fix': True}],
            'DecayParticle': {
                '@Name': dec_part['@Name'],
                '@Helicity': get_helicity_from_edge_props(dec_part)
            },
            'DecayProducts': {'Particle': decay_products}
        }
        partial_decay_dict.update(recoil_system_dict)

        return partial_decay_dict

    def generate_kinematics(self):
        tempdict = {
            # <PhspVolume>0.541493</PhspVolume>
            'InitialState': {'Particle': []}, 'FinalState': {'Particle': []}
        }
        is_edge_ids = get_initial_state_edges(self.graphs[0])
        for x in is_edge_ids:
            tempdict['InitialState']['Particle'].append(
                {'@Name': self.graphs[0].edge_props[x]['@Name'], '@Id': x})
        fs_edge_ids = get_final_state_edges(self.graphs[0])
        for x in fs_edge_ids:
            tempdict['FinalState']['Particle'].append(
                {'@Name': self.graphs[0].edge_props[x]['@Name'], '@Id': x})
        self.kinematics = {'HelicityKinematics': tempdict}

    def write_to_xml(self, output_file):
        with open(output_file, mode='w') as xmlfile:
            full_dict = self.particle_list
            full_dict.update(self.kinematics)
            full_dict.update(self.helicity_amplitudes)
            xmltodict.unparse(OrderedDict(
                {'root': full_dict}), output=xmlfile, pretty=True)
            xmlfile.close()
