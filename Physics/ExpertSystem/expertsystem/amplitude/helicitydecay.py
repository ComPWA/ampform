from collections import OrderedDict
import json
import logging
from copy import deepcopy

import xmltodict

from expertsystem.amplitude.abstractgenerator import (
    AbstractAmplitudeNameGenerator,
    AbstractAmplitudeGenerator
)

from expertsystem.topology.graph import (get_initial_state_edges,
                                         get_final_state_edges,
                                         get_edges_ingoing_to_node,
                                         get_edges_outgoing_to_node)
from expertsystem.state.particle import (
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

class HelicityPartialDecayNameGenerator(AbstractAmplitudeNameGenerator):
    def __init__(self, use_parity_conservation):
        self.use_parity_conservation = use_parity_conservation
        self.generated_parameter_names = []

    def generate(self, graph, node_id):
        # get ending node of the edge
        # then make name for
        in_edges = get_edges_ingoing_to_node(graph, node_id)
        out_edges = get_edges_outgoing_to_node(graph, node_id)
        name_label = get_xml_label(XMLLabelConstants.Name)
        names = []
        hel = []
        for i in in_edges + out_edges:
            names.append(graph.edge_props[i][name_label])
            temphel = float(get_helicity_from_edge_props(graph.edge_props[i]))
            # remove .0
            if temphel % 1 == 0:
                temphel = int(temphel)
            hel.append(temphel)

        par_name_suffix = '_to_'
        par_name_suffix += names[1] + '_' + str(hel[1])
        par_name_suffix += '+' + names[2] + '_' + str(hel[2])
        name = names[0] + '_' + str(hel[0]) + par_name_suffix
        par_name_suffix = names[0] + par_name_suffix
        if par_name_suffix not in self.generated_parameter_names:
            append_name = True
            if self.use_parity_conservation:
                # first check if parity partner exists
                pp_par_name_suffix = names[0]
                pp_par_name_suffix += '_to_'
                pp_par_name_suffix += names[1] + '_' + str(-1 * hel[1])
                pp_par_name_suffix += '+' + \
                    names[2] + '_' + str(-1 * hel[2])
                if pp_par_name_suffix in self.generated_parameter_names:
                    par_name_suffix = pp_par_name_suffix
                    append_name = False
            if append_name:
                self.generated_parameter_names.append(par_name_suffix)
        return (name, par_name_suffix)


class HelicityDecayAmplitudeGeneratorXML(AbstractAmplitudeGenerator):
    def __init__(self, top_node_no_dynamics=True,
                 use_parity_conservation=None):
        self.particle_list = {}
        self.helicity_amplitudes = {}
        self.kinematics = {}
        self.use_parity_conservation = use_parity_conservation
        self.top_node_no_dynamics = top_node_no_dynamics

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

        # if use_parity_conservation flag is set to None, use automatic
        # settings. check if the parity prefactor is defined, if so use
        # parity conservation
        if self.use_parity_conservation is None:
            prefactors = [x for x in graphs if get_prefactor(x) is not None]
            self.use_parity_conservation = False
            if prefactors:
                self.use_parity_conservation = True
                logging.info("Using parity conservation to connect fit "
                             "parameters together with prefactors.")
        graph_groups = group_graphs_same_initial_and_final(graphs)
        logging.debug("There are " + str(len(graph_groups)) + " graph groups")
        # At first we need to define the fit paramteres
        name_generator = HelicityPartialDecayNameGenerator(
            self.use_parity_conservation)
        parameter_mapping = self.generate_fit_parameters(graph_groups,
                                                         name_generator)
        self.fix_parameters_unambiguously(parameter_mapping)
        fit_params = set()
        for x in parameter_mapping.values():
            for y in x.values():
                if not y['Magnitude'][1]:
                    fit_params.add('Magnitude_' + y['ParameterNameSuffix'])
                if not y['Phase'][1]:
                    fit_params.add('Phase_' + y['ParameterNameSuffix'])
        logging.info("Number of parameters:" + str(len(fit_params)))
        self.generate_amplitude_info(graph_groups, parameter_mapping)

    def generate_fit_parameters(self, graph_groups, name_generator_functor):
        '''
        Defines fit parameters and their connections. Parameters with the same
        name (all other properties also have to be the same) will automatically
        be treated as the same parameter in the c++ helicity module.
        '''
        parameter_mapping = {}
        for graph_group in graph_groups:
            graph_group_parameters = {}
            for graph in graph_group:
                # loop over decay nodes in time order
                seq_dec_amp_name = ''
                seq_dec_par_suffix = ''
                parameter_props = {}
                for node_id in graph.nodes:
                    (amp_name, par_suffix) = name_generator_functor.generate(
                        graph, node_id)
                    parameter_props.update({node_id: {'Name': amp_name}})
                    seq_dec_amp_name += amp_name + ';'
                    seq_dec_par_suffix += par_suffix + ';'
                parameter_props.update({'AmplitudeName': seq_dec_amp_name,
                                        'ParameterNameSuffix':
                                        seq_dec_par_suffix,
                                        'Magnitude': (1.0, False),
                                        'Phase': (0.0, False)
                                        })
                gi = graph_group.index(graph)
                graph_group_parameters[gi] = parameter_props
            ggi = graph_groups.index(graph_group)
            parameter_mapping[ggi] = graph_group_parameters
        return parameter_mapping

    def fix_parameters_unambiguously(self, parameter_mapping):
        '''
        Fix parameters, so that the total amplitude is unambiguous, with regard
        to the fit parameters. In other words: all fit parameters per graph,
        except one, will all be fixed. It's fine if they are all already fixed.
        '''
        pass

    def generate_magnitude_and_phase(self, parameter_mapping):
        par_label = get_xml_label(XMLLabelConstants.Parameter)
        par_suffix = parameter_mapping['ParameterNameSuffix']
        mag = parameter_mapping['Magnitude']
        phase = parameter_mapping['Phase']
        return {par_label: [{'@Class': "Double", '@Type': "Magnitude",
                             '@Name': "Magnitude_" + par_suffix,
                             'Value': mag[0], 'Fix': mag[1]},
                            {'@Class': "Double", '@Type': "Phase",
                             '@Name': "Phase_" + par_suffix,
                             'Value': phase[0], 'Fix': phase[1]}]}

    def generate_amplitude_info(self, graph_groups, parameter_mapping):
        class_label = get_xml_label(XMLLabelConstants.Class)
        name_label = get_xml_label(XMLLabelConstants.Name)
        type_label = get_xml_label(XMLLabelConstants.Type)
        parameter_label = get_xml_label(XMLLabelConstants.Parameter)

        # for each graph group we create a coherent amplitude
        coherent_amplitudes = []
        for graph_group in graph_groups:
            seq_partial_decays = []
            ggi = graph_groups.index(graph_group)
            for graph in graph_group:
                gi = graph_group.index(graph)
                seq_partial_decays.append(
                    self.generate_sequential_decay(graph,
                                                   parameter_mapping[ggi][gi]))

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

    def generate_sequential_decay(self, graph, parameter_props):
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

            partial_decays.append(self.generate_partial_decay(graph,
                                                              node_id,
                                                              parameter_props)
                                  )

        amp_name = parameter_props['AmplitudeName']
        seq_decay_dict = {
            class_label: "SequentialPartialAmplitude",
            name_label: amp_name,
            'PartialAmplitude': partial_decays
        }
        seq_decay_dict.update(
            self.generate_magnitude_and_phase(parameter_props))
        prefactor = get_prefactor(graph)
        if prefactor != 1.0 and prefactor is not None:
            prefactor_label = get_xml_label(XMLLabelConstants.PreFactor)
            seq_decay_dict[prefactor_label] = {'@Magnitude': prefactor,
                                               '@Phase': 0.0}
        return seq_decay_dict

    def generate_partial_decay(self, graph, node_id, parameter_props):
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

        amp_name = parameter_props[node_id]['Name']
        partial_decay_dict = {
            name_label: amp_name,
            class_label: "HelicityDecay",
            'DecayParticle': {
                name_label: dec_part[name_label],
                '@Helicity': get_helicity_from_edge_props(dec_part)
            },
            'DecayProducts': {'Particle': decay_products}
        }
        # partial_decay_dict.update(self.generate_magnitude_and_phase(amp_name))
        partial_decay_dict.update(recoil_system_dict)

        return partial_decay_dict

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
