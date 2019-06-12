from collections import OrderedDict
import json
import logging
from copy import deepcopy

import xmltodict

from .abstractgenerator import (
    AbstractAmplitudeNameGenerator,
    AbstractAmplitudeGenerator
)

from ..topology.graph import (get_initial_state_edges,
                              get_final_state_edges,
                              get_edges_ingoing_to_node,
                              get_edges_outgoing_to_node)
from ..state.particle import (
    StateQuantumNumberNames, InteractionQuantumNumberNames,
    XMLLabelConstants, get_xml_label, get_interaction_property,
    get_particle_property)


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
        ifsg = (tuple(sorted([json.dumps(graph.edge_props[x]) for x in ise])),
                tuple(sorted([json.dumps(graph.edge_props[x]) for x in fse])))
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
    logging.error(edge_props[qns_label])
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


def get_parent_recoil_edge(graph, edge_id):
    '''
    Determines the id of the recoil edge of the parent edge for the specified
    edge of a graph.

    Args:
        graph (:class:`.StateTransitionGraph`)
        edge_id (int): id of the edge, for which the parents recoil partner is
            determined
    Returns:
        parent recoil edge id (int)
    '''
    node_id = graph.edges[edge_id].originating_node_id
    if node_id is None:
        return None
    ingoing_edges = get_edges_ingoing_to_node(graph, node_id)
    if len(ingoing_edges) != 1:
        raise ValueError("The node with id " + str(node_id) +
                         " does not have a single ingoing edge!\n" +
                         str(graph))
    return get_recoil_edge(graph, ingoing_edges[0])


def get_prefactor(graph):
    '''
    calculates the product of all prefactors defined in this graph as a double
    '''
    prefactor_label = InteractionQuantumNumberNames.ParityPrefactor
    prefactor = None
    for node_id in graph.nodes:
        if node_id in graph.node_props:
            temp_prefactor = get_interaction_property(
                graph.node_props[node_id], prefactor_label)
            if temp_prefactor is not None:
                if prefactor is None:
                    prefactor = temp_prefactor
                else:
                    prefactor *= temp_prefactor
            else:
                prefactor = None
                break
    return prefactor


def generate_kinematics(graphs):
    tempdict = {
        # <PhspVolume>0.541493</PhspVolume>
        'InitialState': {'Particle': []}, 'FinalState': {'Particle': []}
    }
    is_edge_ids = get_initial_state_edges(graphs[0])
    counter = 0
    for x in is_edge_ids:
        tempdict['InitialState']['Particle'].append(
            {'@Name': graphs[0].edge_props[x]['@Name'], '@Id': x,
             '@PositionIndex': counter})
        counter += 1
    fs_edge_ids = get_final_state_edges(graphs[0])
    counter = 0
    for x in fs_edge_ids:
        tempdict['FinalState']['Particle'].append(
            {'@Name': graphs[0].edge_props[x]['@Name'], '@Id': x,
             '@PositionIndex': counter})
        counter += 1
    return {'HelicityKinematics': tempdict}


def generate_particle_list(graphs):
    # create particle entries
    temp_particle_names = []
    particles = []
    for g in graphs:
        for edge_props in g.edge_props.values():
            new_edge_props = remove_spin_projection(edge_props)
            par_name = new_edge_props[get_xml_label(XMLLabelConstants.Name)]
            if par_name not in temp_particle_names:
                particles.append(new_edge_props)
                temp_particle_names.append(par_name)
    return {'ParticleList': {'Particle': particles}}


def remove_spin_projection(edge_props):
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
    type_label = get_xml_label(XMLLabelConstants.Type)
    spin_label = StateQuantumNumberNames.Spin
    proj_label = get_xml_label(XMLLabelConstants.Projection)

    new_edge_props = deepcopy(edge_props)

    for qn_entry in new_edge_props[qns_label]:
        if (StateQuantumNumberNames[qn_entry[type_label]]
                is spin_label):
            del qn_entry[proj_label]
            break
    return new_edge_props


def generate_particles_string(name_hel_dict, use_helicity=True,
                              make_parity_partner=False):
    string = ""
    for name in sorted(name_hel_dict.keys()):
        string += name
        if use_helicity:
            hel = name_hel_dict[name]
            if make_parity_partner:
                string += '_' + str(-1 * hel)
            else:
                string += '_' + str(hel)
        string += '+'
    return string[:-1]


class HelicityAmplitudeNameGenerator(AbstractAmplitudeNameGenerator):
    def __init__(self, use_parity_conservation=False):
        self.partial_amp_coefficient_infos = set()
        self.use_parity_conservation = use_parity_conservation

        # automatically determine parity conservation settings
        if self.use_parity_conservation is None:
            self.use_parity_conservation = True
            logging.debug("Using parity conservation to connect fit "
                          "parameters together with prefactors.")

    def generate_amplitude_coefficient_infos(self, graph):
        '''
        Generates coefficient info for a sequential amplitude graph.

        Generally, each partial amplitude of a sequential amplitude graph
        should check itself if it or a parity partner is already defined. If so
        a coupled coefficient is introduced.
        '''

        seq_par_suffix = ''
        use_prefactor = False
        # loop over decay nodes in time order
        for node_id in graph.nodes:
            (coeff_suffix,
             pp_coeff_suffix) = self.generate_amplitude_coefficient_names(
                graph, node_id)

            if coeff_suffix in self.partial_amp_coefficient_infos:
                seq_par_suffix += coeff_suffix + ';'
            else:
                if (self.use_parity_conservation and
                        pp_coeff_suffix in self.partial_amp_coefficient_infos):
                    seq_par_suffix += pp_coeff_suffix + ';'
                    use_prefactor = True
                else:
                    seq_par_suffix += coeff_suffix + ';'
                    self.partial_amp_coefficient_infos.add(coeff_suffix)

        par_label = get_xml_label(XMLLabelConstants.Parameter)
        amplitude_coefficient_infos = {
            par_label: [{'@Class': "Double", '@Type': "Magnitude",
                         '@Name': "Magnitude_" + seq_par_suffix,
                         'Value': 1.0, 'Fix': False},
                        {'@Class': "Double", '@Type': "Phase",
                            '@Name': "Phase_" + seq_par_suffix,
                            'Value': 0.0, 'Fix': False}]
        }

        # add potential prefactor
        if self.use_parity_conservation and use_prefactor:
            prefactor = get_prefactor(graph)
            if prefactor != 1.0 and prefactor is not None:
                prefactor_label = get_xml_label(XMLLabelConstants.PreFactor)
                amplitude_coefficient_infos[prefactor_label] = {
                    '@Real': prefactor}
        return amplitude_coefficient_infos

    def generate_amplitude_coefficient_names(self, graph, node_id):
        '''
        Generates partial amplitude coefficient name suffixes.
        '''
        (in_hel_info, out_hel_info) = self.retrieve_helicity_info(graph,
                                                                  node_id)
        par_name_suffix = generate_particles_string(
            in_hel_info, False) + '_to_' + \
            generate_particles_string(out_hel_info)

        pp_par_name_suffix = generate_particles_string(
            in_hel_info, False) + '_to_' + \
            generate_particles_string(out_hel_info,
                                      make_parity_partner=True)
        return (par_name_suffix, pp_par_name_suffix)

    def generate_unique_amplitude_name(self, graph, node_id=None):
        '''
        Generates a unique name for the amplitude corresponding to the given
        :py:class:`StateTransitionGraph`. If :py:var:`node_id` is given, it
        generates a unique name for the partial amplitude corresponding to the
        interaction node of the given :py:class:`StateTransitionGraph`.
        '''
        name = ''
        if isinstance(node_id, int):
            nodelist = [node_id]
        else:
            nodelist = graph.nodes
        for node_id in nodelist:
            (in_hel_info, out_hel_info) = self.retrieve_helicity_info(graph,
                                                                      node_id)

            name += generate_particles_string(in_hel_info) + '_to_' + \
                generate_particles_string(out_hel_info) + ';'
        return name

    def retrieve_helicity_info(self, graph, node_id):
        # get ending node of the edge
        # then make name for
        in_edges = get_edges_ingoing_to_node(graph, node_id)
        out_edges = get_edges_outgoing_to_node(graph, node_id)
        name_label = get_xml_label(XMLLabelConstants.Name)
        in_names_hel_dict = {}
        out_names_hel_dict = {}
        for i in in_edges:
            temphel = float(get_helicity_from_edge_props(graph.edge_props[i]))
            # remove .0
            if temphel % 1 == 0:
                temphel = int(temphel)
            in_names_hel_dict[graph.edge_props[i][name_label]] = temphel
        for i in out_edges:
            temphel = float(get_helicity_from_edge_props(graph.edge_props[i]))
            # remove .0
            if temphel % 1 == 0:
                temphel = int(temphel)
            out_names_hel_dict[graph.edge_props[i][name_label]] = temphel

        return (in_names_hel_dict, out_names_hel_dict)


class HelicityAmplitudeGeneratorXML(AbstractAmplitudeGenerator):
    def __init__(self, top_node_no_dynamics=True,
                 name_generator=HelicityAmplitudeNameGenerator(None)):
        self.particle_list = {}
        self.helicity_amplitudes = {}
        self.kinematics = {}
        self.top_node_no_dynamics = top_node_no_dynamics
        self.name_generator = name_generator
        self.fit_parameter_names = set()

    def generate(self, graphs):
        if len(graphs) <= 0:
            raise ValueError(
                "Number of solution graphs is not larger than zero!")

        decay_info = {get_xml_label(XMLLabelConstants.Type): 'nonResonant'}
        decay_info_label = get_xml_label(XMLLabelConstants.DecayInfo)
        for g in graphs:
            if self.top_node_no_dynamics:
                init_edges = get_initial_state_edges(g)
                if len(init_edges) > 1:
                    raise ValueError(
                        "Only a single initial state particle allowed")
                eprops = g.edge_props[init_edges[0]]
                eprops[decay_info_label] = decay_info

        self.particle_list = generate_particle_list(graphs)
        self.kinematics = generate_kinematics(graphs)

        graph_groups = group_graphs_same_initial_and_final(graphs)
        logging.debug("There are " + str(len(graph_groups)) + " graph groups")

        self.fix_parameters_unambiguously()

        self.generate_amplitude_info(graph_groups)

    def fix_parameters_unambiguously(self):
        '''
        Fix parameters, so that the total amplitude is unambiguous, with regard
        to the fit parameters. In other words: all fit parameters per graph,
        except one, will all be fixed. It's fine if they are all already fixed.
        '''
        pass

    def generate_amplitude_info(self, graph_groups):
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
                class_label: 'CoherentIntensity',
                name_label: coherent_amp_name,
                'Amplitude': seq_partial_decays
            })

        # now wrap it with an incoherent intensity
        incoherent_amp_name = "incoherent"
        self.helicity_amplitudes = {
            'Intensity': {
                class_label: "StrengthIntensity",
                name_label: incoherent_amp_name+"_with_strength",
                parameter_label: {class_label: "Double",
                                  type_label: "Strength",
                                  name_label: "strength_" +
                                  incoherent_amp_name,
                                  'Value': 1, 'Fix': True},
                'Intensity': {
                    class_label: "IncoherentIntensity",
                    name_label: incoherent_amp_name,
                    'Intensity': coherent_amplitudes
                }
            }
        }

    def generate_sequential_decay(self, graph):
        class_label = get_xml_label(XMLLabelConstants.Class)
        name_label = get_xml_label(XMLLabelConstants.Name)
        spin_label = StateQuantumNumberNames.Spin
        decay_info_label = get_xml_label(XMLLabelConstants.DecayInfo)
        type_label = get_xml_label(XMLLabelConstants.Type)
        partial_decays = []
        for node_id in graph.nodes:
            # in case a scalar without dynamics decays into daughters with no
            # net helicity, the partial amplitude can be dropped
            # (it is just a constant)
            in_edges = get_edges_ingoing_to_node(graph, node_id)
            out_edges = get_edges_outgoing_to_node(graph, node_id)
            # check mother particle is spin 0
            in_spin = get_particle_property(
                graph.edge_props[in_edges[0]], spin_label)
            out_spins = [get_particle_property(
                graph.edge_props[x], spin_label) for x in out_edges]
            if (in_spin is not None and None not in out_spins
                    and in_spin.magnitude() == 0):
                if abs(out_spins[0].projection()
                       - out_spins[1].projection()) == 0.0:
                    # check if dynamics is non resonsant (constant)
                    if ('NonResonant' ==
                        graph.edge_props[in_edges[0]][
                            decay_info_label][type_label]):
                        continue

            partial_decays.append(self.generate_partial_decay(graph, node_id))

        gen = self.name_generator
        amp_name = gen.generate_unique_amplitude_name(graph)
        amp_coeff_infos = gen.generate_amplitude_coefficient_infos(graph)
        sequential_amplitude_dict = {
            class_label: "SequentialAmplitude",
            name_label: amp_name,
            'Amplitude': partial_decays
        }
        prefactor_label = get_xml_label(XMLLabelConstants.PreFactor)
        if prefactor_label in amp_coeff_infos:
            sequential_amplitude_dict.update(
                {prefactor_label:
                 amp_coeff_infos[prefactor_label]})

        par_label = get_xml_label(XMLLabelConstants.Parameter)
        coefficient_amplitude_dict = {
            class_label: "CoefficientAmplitude",
            name_label: amp_name,
            par_label: amp_coeff_infos[par_label],
            'Amplitude': sequential_amplitude_dict
        }

        self.fit_parameter_names.add(amp_coeff_infos[par_label][0][name_label])
        self.fit_parameter_names.add(amp_coeff_infos[par_label][1][name_label])

        return coefficient_amplitude_dict

    def generate_partial_decay(self, graph, node_id):
        class_label = get_xml_label(XMLLabelConstants.Class)
        name_label = get_xml_label(XMLLabelConstants.Name)
        decay_products = []
        for out_edge_id in get_edges_outgoing_to_node(graph,
                                                      node_id):
            decay_products.append({
                name_label: graph.edge_props[out_edge_id][name_label],
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

        recoil_edge_id = get_recoil_edge(graph, in_edge_ids[0])
        parent_recoil_edge_id = get_parent_recoil_edge(graph, in_edge_ids[0])
        recoil_system_dict = {}
        if recoil_edge_id is not None:
            tempdict = {
                '@RecoilFinalState':
                determine_attached_final_state_string(graph, recoil_edge_id)
            }
            if parent_recoil_edge_id is not None:
                tempdict.update({
                    '@ParentRecoilFinalState':
                    determine_attached_final_state_string(
                        graph, parent_recoil_edge_id)
                })
            recoil_system_dict['RecoilSystem'] = tempdict

        amp_name = self.name_generator.generate_unique_amplitude_name(graph,
                                                                      node_id)
        partial_decay_dict = {
            name_label: amp_name,
            class_label: "HelicityDecay",
            'DecayParticle': {
                name_label: dec_part[name_label],
                '@Helicity': get_helicity_from_edge_props(dec_part)
            },
            'DecayProducts': {'Particle': decay_products}
        }

        partial_decay_dict.update(recoil_system_dict)

        return partial_decay_dict

    def get_fit_parameters(self):
        logging.info("Number of parameters:" + str(
            len(self.fit_parameter_names)))
        return self.fit_parameter_names

    def write_to_file(self, filename):
        with open(filename, mode='w') as xmlfile:
            full_dict = self.particle_list
            full_dict.update(self.kinematics)
            full_dict.update(self.helicity_amplitudes)
            # xmltodict only allows a single xml root
            xmlstring = xmltodict.unparse(OrderedDict(
                {'root': full_dict}), pretty=True)
            # before writing it to file we remove the root tag again
            xmlstring = xmlstring.replace('<root>', '', 1)
            xmlstring = xmlstring[:-10] + \
                xmlstring[-10:].replace('</root>', '', 1)
            xmlfile.write(xmlstring)
            xmlfile.close()
