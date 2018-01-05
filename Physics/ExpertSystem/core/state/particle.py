""" This module defines a particle as a collection of quantum numbers and
things related to this"""

from enum import Enum

import xmltodict

from core.topology.graph import (
    get_initial_state_edges, get_final_state_edges)


XMLLabelConstants = Enum('XMLLabelConstants',
                         'Name Type Value QuantumNumber \
                          Class Projection Parameter')


def get_xml_label(enum):
    """return the the correctly formatted xml label
    as required by ComPWA and xmltodict"""

    # the xml attribute prefix is needed as the xmltodict module uses that
    attribute_prefix = '@'
    if enum is (XMLLabelConstants.QuantumNumber or
                XMLLabelConstants.Parameter):
        return enum.name
    else:
        return attribute_prefix + enum.name


QuantumNumberClasses = Enum('QuantumNumberClasses', 'Int Spin')

"""definition of quantum number names for particles"""
ParticleQuantumNumberNames = Enum(
    'ParticleQuantumNumbers', 'All Charge Spin IsoSpin Parity Cparity')

"""definition of quantum number names for interaction nodes"""
InteractionQuantumNumberNames = Enum('InteractionQuantumNumbers', 'L')

QNNameClassMapping = {ParticleQuantumNumberNames.Charge: QuantumNumberClasses.Int,
                      ParticleQuantumNumberNames.Spin: QuantumNumberClasses.Spin,
                      ParticleQuantumNumberNames.IsoSpin: QuantumNumberClasses.Spin,
                      ParticleQuantumNumberNames.Parity: QuantumNumberClasses.Int,
                      ParticleQuantumNumberNames.Cparity: QuantumNumberClasses.Int,
                      InteractionQuantumNumberNames.L: QuantumNumberClasses.Spin
                      }

QNAttributeOption = Enum('QNAttributeOption', 'Required Optional')

QNClassAttributes = {QuantumNumberClasses.Spin: [
    (XMLLabelConstants.Projection, QNAttributeOption.Required)]}


def get_attributes_for_qn(qn_name):
    qn_attr = []
    if qn_name in QNNameClassMapping:
        qn_class = QNNameClassMapping[qn_name]
        if qn_class in QNClassAttributes:
            qn_attr = QNClassAttributes[qn_class]
    return qn_attr


particle_list = []


def initialize_graph(graph, initial_state, final_state):
    is_edges = get_initial_state_edges(graph)
    if len(initial_state) != len(is_edges):
        raise ValueError("The graph initial state and the supplied initial"
                         "state are of different size! (" +
                         str(len(is_edges)) + " != " +
                         str(len(initial_state)) + ")")
    initialize_edges(graph, is_edges, initial_state)

    fs_edges = get_final_state_edges(graph)
    if len(final_state) != len(fs_edges):
        raise ValueError("The graph final state and the supplied final"
                         "state are of different size! (" +
                         str(len(fs_edges)) + " != " +
                         str(len(final_state)) + ")")
    initialize_edges(graph, fs_edges, final_state)

    return graph


def initialize_edges(graph, edges, particles):
    for edge in zip(edges, particles):
        # lookup the particle in the list
        name_label = get_xml_label(XMLLabelConstants.Name)
        found_particles = [
            p for p in particle_list if (p[name_label] == edge[1])]
        # and attach quantum numbers
        if found_particles:
            if len(found_particles) > 1:
                raise ValueError(
                    "more than one particle with name "
                    + str(edge[1]) + " found!")
            graph.edge_props[edge[0]] = found_particles[0]
        # now add more quantum numbers given by user (spin_projection)



def load_particle_list_from_xml(file_path):
    with open(file_path, "rb") as xmlfile:
        full_dict = xmltodict.parse(xmlfile)
        for p in full_dict['ParticleList']['Particle']:
            particle_list.append(p)
    """tree = ET.parse(file_path)
    root = tree.getroot()
    # loop over particles
    for particle_xml in root:
        extract_particle(particle_xml)"""


"""
def extract_particle(particle_xml):
    particle = {}
    QN_LIST_LABEL = XMLLabelConstants.QN_LIST_LABEL
    QN_CLASS_LABEL = XMLLabelConstants.QN_CLASS_LABEL
    TYPE_LABEL = XMLLabelConstants.VAR_TYPE_LABEL
    VALUE_LABEL = XMLLabelConstants.VAR_VALUE_LABEL
    QN_PROJECTION_LABEL = XMLLabelConstants.QN_PROJECTION_LABEL

    particle['Name'] = particle_xml.attrib['Name']
    for part_prop in particle_xml:
        if "Parameter" in part_prop.tag:
            if 'Parameters' not in particle:
                particle['Parameters'] = []
            particle['Parameters'].append(
                {TYPE_LABEL: part_prop.attrib[TYPE_LABEL], VALUE_LABEL: part_prop.find(VALUE_LABEL).text})
        elif QN_LIST_LABEL in part_prop.tag:
            if QN_LIST_LABEL not in particle:
                particle[QN_LIST_LABEL] = []
            particle[QN_LIST_LABEL].append(
                {TYPE_LABEL: part_prop.attrib[TYPE_LABEL], VALUE_LABEL: part_prop.attrib[VALUE_LABEL]})
        elif "DecayInfo" in part_prop.tag:
            continue
        else:
            particle[part_prop.tag] = part_prop.text

    particle_list.append(particle)
"""
