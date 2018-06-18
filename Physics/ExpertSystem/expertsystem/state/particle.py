"""
This module defines a particle as a collection of quantum numbers and
things related to this
"""
import logging
from copy import deepcopy
from enum import Enum
from abc import ABC, abstractmethod
from numpy import arange
from itertools import permutations
from json import loads, dumps
from collections import OrderedDict

import xmltodict

from expertsystem.topology.graph import (
    get_initial_state_edges,
    get_final_state_edges,
    get_intermediate_state_edges,
    get_edge_groups_full_attached_node)


XMLLabelConstants = Enum('XMLLabelConstants',
                         'Name Pid Type Value QuantumNumber \
                          Class Projection Parameter PreFactor DecayInfo')

XMLLabelTags = [
    XMLLabelConstants.QuantumNumber,
    XMLLabelConstants.Parameter,
    XMLLabelConstants.PreFactor,
    XMLLabelConstants.DecayInfo
]


def get_xml_label(enum):
    """return the the correctly formatted xml label
    as required by ComPWA and xmltodict"""

    # the xml attribute prefix is needed as the xmltodict module uses that
    attribute_prefix = '@'
    if (enum in XMLLabelTags):
        return enum.name
    else:
        return attribute_prefix + enum.name


class Spin():
    """
    Simple struct-like class defining spin as a magintude plus the projection
    """

    def __init__(self, mag, proj):
        self.__magnitude = float(mag)
        self.__projection = float(proj)
        if self.__magnitude < self.__projection:
            raise ValueError("The spin projection cannot be larger than the"
                             " magnitude " + self.__str__())

    def magnitude(self):
        return self.__magnitude

    def projection(self):
        return self.__projection

    def __str__(self):
        return '(mag: ' + str(self.__magnitude) + ', proj: ' \
            + str(self.__projection) + ')'

    def __repr__(self):
        return '(mag: ' + str(self.__magnitude) + ', proj: ' \
            + str(self.__projection) + ')'

    def __eq__(self, other):
        """
        define the equal operator for the spin class, which is needed for
        equality checks of states in certain rules
        """
        if isinstance(other, Spin):
            return (self.__magnitude == other.magnitude() and
                    self.__projection == other.projection())
        else:
            return NotImplemented

    def __hash__(self):
        return hash(repr(self))


def create_spin_domain(list_of_magnitudes, set_projection_zero=False):
    domain_list = []
    for mag in list_of_magnitudes:
        if set_projection_zero:
            domain_list.append(Spin(mag, 0.0))
        else:
            for proj in arange(-mag, mag + 1, 1.0):
                domain_list.append(Spin(mag, proj))
    return domain_list


QuantumNumberClasses = Enum('QuantumNumberClasses', 'Int Float Spin')

"""definition of quantum number names for states"""
StateQuantumNumberNames = Enum(
    'StateQuantumNumberNames', 'Charge Spin Parity Cparity Gparity IsoSpin\
    Strangeness Charm Bottomness Topness BaryonNumber ElectronLN MuonLN TauLN')

"""definition of properties names of particles"""
ParticlePropertyNames = Enum(
    'ParticlePropertyNames', 'Pid Mass Width')

"""definition of quantum number names for interaction nodes"""
InteractionQuantumNumberNames = Enum('InteractionQuantumNumberNames',
                                     'L S ParityPrefactor')

QNDefaultValues = {
    StateQuantumNumberNames.Charge: 0,
    StateQuantumNumberNames.IsoSpin: Spin(0.0, 0.0),
    StateQuantumNumberNames.Strangeness: 0,
    StateQuantumNumberNames.Charm: 0,
    StateQuantumNumberNames.Bottomness: 0,
    StateQuantumNumberNames.Topness: 0,
    StateQuantumNumberNames.BaryonNumber: 0,
    StateQuantumNumberNames.ElectronLN: 0,
    StateQuantumNumberNames.MuonLN: 0,
    StateQuantumNumberNames.TauLN: 0
}

QNNameClassMapping = {
    StateQuantumNumberNames.Charge: QuantumNumberClasses.Int,
    StateQuantumNumberNames.ElectronLN: QuantumNumberClasses.Int,
    StateQuantumNumberNames.MuonLN: QuantumNumberClasses.Int,
    StateQuantumNumberNames.TauLN: QuantumNumberClasses.Int,
    StateQuantumNumberNames.BaryonNumber: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Spin: QuantumNumberClasses.Spin,
    StateQuantumNumberNames.Parity: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Cparity: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Gparity: QuantumNumberClasses.Int,
    StateQuantumNumberNames.IsoSpin: QuantumNumberClasses.Spin,
    StateQuantumNumberNames.Strangeness: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Charm: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Bottomness: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Topness: QuantumNumberClasses.Int,
    InteractionQuantumNumberNames.L: QuantumNumberClasses.Spin,
    InteractionQuantumNumberNames.S: QuantumNumberClasses.Spin,
    InteractionQuantumNumberNames.ParityPrefactor:
    QuantumNumberClasses.Int,
    ParticlePropertyNames.Pid: QuantumNumberClasses.Int,
    ParticlePropertyNames.Mass: QuantumNumberClasses.Float,
    ParticlePropertyNames.Width: QuantumNumberClasses.Float
}


class AbstractQNConverter(ABC):
    @abstractmethod
    def parse_from_dict(self, data_dict):
        pass

    @abstractmethod
    def convert_to_dict(self, qn_type, qn_value):
        pass


class IntQNConverter(AbstractQNConverter):
    value_label = get_xml_label(XMLLabelConstants.Value)
    type_label = get_xml_label(XMLLabelConstants.Type)
    class_label = get_xml_label(XMLLabelConstants.Class)

    def parse_from_dict(self, data_dict):
        return int(data_dict[self.value_label])

    def convert_to_dict(self, qn_type, qn_value):
        return {self.type_label: qn_type.name,
                self.class_label: QuantumNumberClasses.Int.name,
                self.value_label: str(qn_value)}


class FloatQNConverter(AbstractQNConverter):
    value_label = get_xml_label(XMLLabelConstants.Value)
    type_label = get_xml_label(XMLLabelConstants.Type)
    class_label = get_xml_label(XMLLabelConstants.Class)

    def parse_from_dict(self, data_dict):
        return float(data_dict[self.value_label])

    def convert_to_dict(self, qn_type, qn_value):
        return {self.type_label: qn_type.name,
                self.class_label: QuantumNumberClasses.Float.name,
                self.value_label: str(qn_value)}


class SpinQNConverter(AbstractQNConverter):
    type_label = get_xml_label(XMLLabelConstants.Type)
    class_label = get_xml_label(XMLLabelConstants.Class)
    value_label = get_xml_label(XMLLabelConstants.Value)
    proj_label = get_xml_label(XMLLabelConstants.Projection)

    def __init__(self, parse_projection=True):
        self.parse_projection = parse_projection

    def parse_from_dict(self, data_dict):
        mag = data_dict[self.value_label]
        proj = 0.0
        if self.parse_projection:
            if self.proj_label not in data_dict:
                if float(mag) != 0.0:
                    raise ValueError(
                        "No projection set for spin-like quantum number!")
            else:
                proj = data_dict[self.proj_label]
        return Spin(mag, proj)

    def convert_to_dict(self, qn_type, qn_value):
        return {self.type_label: qn_type.name,
                self.class_label: QuantumNumberClasses.Spin.name,
                self.value_label: str(qn_value.magnitude()),
                self.proj_label: str(qn_value.projection())}


QNClassConverterMapping = {
    QuantumNumberClasses.Int: IntQNConverter(),
    QuantumNumberClasses.Float: FloatQNConverter(),
    QuantumNumberClasses.Spin: SpinQNConverter()
}


def is_boson(qn_dict):
    spin_label = StateQuantumNumberNames.Spin
    return abs(qn_dict[spin_label].magnitude() % 1) < 0.01


particle_list = []


def load_particle_list_from_xml(file_path):
    with open(file_path, "rb") as xmlfile:
        full_dict = xmltodict.parse(xmlfile)
        for p in full_dict['ParticleList']['Particle']:
            particle_list.append(dict(p))


def get_particle_with_name(particle_name):
    name_label = get_xml_label(XMLLabelConstants.Name)
    found_particles = [
        p for p in particle_list if (p[name_label] == particle_name)]
    if len(found_particles) == 0:
        raise ValueError(
            "No particle with name " + str(particle_name) + " found!")
    elif len(found_particles) > 1:
        raise ValueError(
            "more than one particle with name " + str(particle_name)
            + " found!")
    return found_particles[0]


def get_particle_property(particle_properties, qn_name, converter=None):
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
    type_label = get_xml_label(XMLLabelConstants.Type)
    value_label = get_xml_label(XMLLabelConstants.Value)

    found_prop = None
    if isinstance(qn_name, StateQuantumNumberNames):
        particle_qns = particle_properties[qns_label]
        for x in particle_qns:
            if (x[type_label] == qn_name.name):
                found_prop = x
                break
    else:
        for key, val in particle_properties.items():
            if (key == qn_name.name):
                found_prop = {value_label: val}
                break
            if (key == 'Parameter' and
                    val[type_label] == qn_name.name):
                # parameters have a seperate value tag
                tagname = XMLLabelConstants.Value.name
                found_prop = {value_label: val[tagname]}
                break
    # check for default value
    property_value = None
    if found_prop is not None:
        if converter is None:
            converter = QNClassConverterMapping[QNNameClassMapping[qn_name]]
        property_value = converter.parse_from_dict(found_prop)
    else:
        if qn_name in QNDefaultValues:
            property_value = QNDefaultValues[qn_name]
    return property_value


def get_interaction_property(interaction_properties, qn_name, converter=None):
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
    type_label = get_xml_label(XMLLabelConstants.Type)

    found_prop = None
    if isinstance(qn_name, InteractionQuantumNumberNames):
        interaction_qns = interaction_properties[qns_label]
        for x in interaction_qns:
            if (x[type_label] == qn_name.name):
                found_prop = x
                break
    # check for default value
    property_value = None
    if found_prop is not None:
        if converter is None:
            converter = QNClassConverterMapping[QNNameClassMapping[qn_name]]
        property_value = converter.parse_from_dict(found_prop)
    else:
        if qn_name in QNDefaultValues:
            property_value = QNDefaultValues[qn_name]
    return property_value


def compare_graph_element_properties(edge_props1, edge_props2):
    copy_edge_props1 = deepcopy(edge_props1)
    copy_edge_props2 = deepcopy(edge_props2)
    # first clean up the edge qn list, by converting it to a dict
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
    type_label = get_xml_label(XMLLabelConstants.Type)
    for edge_props in copy_edge_props1.values():
        if qns_label in edge_props:
            new_qns = {}
            for x in edge_props[qns_label]:
                temp_qn_dict = dict(x)
                type_name = temp_qn_dict[type_label]
                del temp_qn_dict[type_label]
                new_qns[type_name] = temp_qn_dict
            edge_props[qns_label] = new_qns
    for edge_props in copy_edge_props2.values():
        if qns_label in edge_props:
            new_qns = {}
            for x in edge_props[qns_label]:
                temp_qn_dict = dict(x)
                type_name = temp_qn_dict[type_label]
                del temp_qn_dict[type_label]
                new_qns[type_name] = temp_qn_dict
            edge_props[qns_label] = new_qns
    if (loads(dumps(copy_edge_props1, sort_keys=True),
              object_pairs_hook=OrderedDict)
            != loads(dumps(copy_edge_props2, sort_keys=True),
                     object_pairs_hook=OrderedDict)):
        return False
    return True


def initialize_graph(graph, initial_state, final_state):
    is_edges = get_initial_state_edges(graph)
    if len(initial_state) != len(is_edges):
        raise ValueError("The graph initial state and the supplied initial"
                         "state are of different size! (" +
                         str(len(is_edges)) + " != " +
                         str(len(initial_state)) + ")")
    fs_edges = get_final_state_edges(graph)
    if len(final_state) != len(fs_edges):
        raise ValueError("The graph final state and the supplied final"
                         "state are of different size! (" +
                         str(len(fs_edges)) + " != " +
                         str(len(final_state)) + ")")

    # check if all initial and final state particles have spin projections set
    initial_state = [check_if_spin_projections_set(x) for x in initial_state]
    final_state = [check_if_spin_projections_set(x) for x in final_state]

    same_node_edges = get_edge_groups_full_attached_node(graph, is_edges)
    is_edge_particle_pairs = calculate_combinatorics(
        is_edges, initial_state, same_node_edges)
    same_node_edges = get_edge_groups_full_attached_node(graph, fs_edges)
    fs_edge_particle_pairs = calculate_combinatorics(
        fs_edges, final_state, same_node_edges)

    new_graphs = []
    for is_pair in is_edge_particle_pairs:
        for fs_pair in fs_edge_particle_pairs:
            new_graphs.extend(initialize_edges(
                graph, is_pair + fs_pair))

    return new_graphs


def check_if_spin_projections_set(state):
    spin_label = StateQuantumNumberNames.Spin
    mass_label = ParticlePropertyNames.Mass
    if isinstance(state, str):
        particle = get_particle_with_name(state)
        spin = get_particle_property(particle, spin_label,
                                     SpinQNConverter(False))
        if not isinstance(spin, Spin):
            raise ValueError(
                "Spin not defined for particle: \n" + str(particle))
        mag = spin.magnitude()
        spin_projs = arange(-mag, mag + 1, 1.0).tolist()
        mass = get_particle_property(particle, mass_label)
        if mass == 0.0:
            if 0.0 in spin_projs:
                del spin_projs[spin_projs.index(0.0)]
        state = (state, spin_projs)
    return state


def calculate_combinatorics(edges, state_particles, same_node_edges):
    combinatorics_list = [list(zip(edges, particles))
                          for particles in permutations(state_particles)]

    # remove equal combinations
    combinations_to_remove = set()
    for i in range(len(combinatorics_list)):
        for j in range(i + 1, len(combinatorics_list)):
            if combinations_equal(combinatorics_list[i],
                                  combinatorics_list[j],
                                  same_node_edges):
                combinations_to_remove.add(i)
                break

    for comb_index in sorted(combinations_to_remove, reverse=True):
        del combinatorics_list[comb_index]
    return combinatorics_list


def combinations_equal(comb1, comb2, same_node_edges):
    # if a node has all in or outgoing edges defined
    # (by initial and final state particles) and they are permuted

    # first check which edge ids are different
    different_edge_ids = []
    comp_dict = {}
    for eid, part in comb1:
        comp_dict[eid] = [part[0]]
    for eid, part in comb2:
        comp_dict[eid].append(part[0])
    for key, val in comp_dict.items():
        if val[0] != val[1]:
            different_edge_ids.append(key)

    if len(different_edge_ids) == 0:
        return True

    # check if the different edges are all belonging to the same node
    # and also check if all different edges belong to the same time level
    possible_time_level = -1
    for time_level, edge_lists in same_node_edges.items():
        edges_this_time_level = []
        for edge_list in edge_lists:
            if all(elem in edge_list for elem in different_edge_ids):
                return True
            for ele in different_edge_ids:
                if ele in edge_list:
                    edges_this_time_level.append(ele)
        if edges_this_time_level is different_edge_ids:
            possible_time_level = time_level

    if possible_time_level > 0:
        edge_lists = same_node_edges[possible_time_level]
        for perm in permutations(range(len(edge_lists))):
            good_permutation = True
            for i in range(len(edge_lists)):
                list1 = [x[1][0] for x in comb1 if x[0]
                         == eid for eid in edge_lists[perm[i]]]
                list2 = [comb2[eid][0] for eid in edge_lists[i]]
                if set(list1) == set(list2):
                    good_permutation = False
                    break
            if good_permutation:
                return True

    return False


def initialize_edges(graph, edge_particle_pairs):
    for edge, particle in edge_particle_pairs:
        # lookup the particle in the list
        name_label = get_xml_label(XMLLabelConstants.Name)
        found_particle = get_particle_with_name(particle[0])
        graph.edge_props[edge] = deepcopy(found_particle)

    # now add more quantum numbers given by user (spin_projection)
    new_graphs = [graph]
    for edge_part_pair in edge_particle_pairs:
        temp_graphs = new_graphs
        new_graphs = []
        for g in temp_graphs:
            new_graphs.extend(
                populate_edge_with_spin_projections(g,
                                                    edge_part_pair[0],
                                                    edge_part_pair[1][1]))

    return new_graphs


def populate_edge_with_spin_projections(graph, edge_id, spin_projections):
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)
    type_label = get_xml_label(XMLLabelConstants.Type)
    class_label = get_xml_label(XMLLabelConstants.Class)
    type_value = StateQuantumNumberNames.Spin
    class_value = QNNameClassMapping[type_value]

    new_graphs = []

    qn_list = graph.edge_props[edge_id][qns_label]
    index_list = [qn_list.index(x) for x in qn_list
                  if (type_label in x and class_label in x) and
                  (x[type_label] == type_value.name and
                   x[class_label] == class_value.name)]
    if index_list:
        for spin_proj in spin_projections:
            graph_copy = deepcopy(graph)
            graph_copy.edge_props[edge_id][qns_label][index_list[0]][
                get_xml_label(XMLLabelConstants.Projection)] = spin_proj
            new_graphs.append(graph_copy)

    return new_graphs


def initialize_graphs_with_particles(graphs, allowed_particle_list=[]):
    initialized_graphs = []
    mod_allowed_particle_list = []
    if len(allowed_particle_list) == 0:
        mod_allowed_particle_list = particle_list
    else:
        for x in allowed_particle_list:
            if isinstance(x, str):
                for p in particle_list:
                    if x in p[get_xml_label(XMLLabelConstants.Name)]:
                        mod_allowed_particle_list.append(p)
            else:
                mod_allowed_particle_list.append(x)

    for graph in graphs:
        logging.debug("initializing graph...")
        intermediate_edges = get_intermediate_state_edges(graph)
        current_new_graphs = [graph]
        for int_edge_id in intermediate_edges:
            particle_edges = get_particle_candidates_for_state(
                graph.edge_props[int_edge_id], mod_allowed_particle_list)
            if len(particle_edges) == 0:
                logging.debug("Did not find any particle candidates for")
                logging.debug("edge id: " + str(int_edge_id))
                logging.debug("edge properties:")
                logging.debug(graph.edge_props[int_edge_id])
            new_graphs_temp = []
            for curr_new_graph in current_new_graphs:
                for particle_edge in particle_edges:
                    temp_graph = deepcopy(curr_new_graph)
                    temp_graph.edge_props[int_edge_id] = particle_edge
                    new_graphs_temp.append(temp_graph)
            current_new_graphs = new_graphs_temp

        initialized_graphs.extend(current_new_graphs)
    return initialized_graphs


def get_particle_candidates_for_state(state, allowed_particle_list):
    particle_edges = []
    qns_label = get_xml_label(XMLLabelConstants.QuantumNumber)

    for allowed_state in allowed_particle_list:
        if check_qns_equal(state[qns_label],
                           allowed_state[qns_label]):
            temp_particle = deepcopy(allowed_state)
            temp_particle[qns_label] = merge_qn_props(state[qns_label],
                                                      allowed_state[qns_label])
            particle_edges.append(temp_particle)
    return particle_edges


def check_qns_equal(qns_state, qns_particle):
    equal = True
    class_label = get_xml_label(XMLLabelConstants.Class)
    type_label = get_xml_label(XMLLabelConstants.Type)
    for qn_entry in qns_state:
        qn_found = False
        qn_value_match = False
        for par_qn_entry in qns_particle:
            # first check if the type and class of these
            # qn entries are the same
            if (StateQuantumNumberNames[qn_entry[type_label]]
                is StateQuantumNumberNames[par_qn_entry[type_label]] and
                    QuantumNumberClasses[qn_entry[class_label]]
                    is QuantumNumberClasses[par_qn_entry[class_label]]):
                qn_found = True
                if compare_qns(qn_entry, par_qn_entry):
                    qn_value_match = True
                break
        if not qn_found:
            # check if there is a default value
            qn_name = StateQuantumNumberNames[qn_entry[type_label]]
            if qn_name in QNDefaultValues:
                if compare_qns(qn_entry,
                               QNDefaultValues[qn_name]):
                    qn_found = True
                    qn_value_match = True

        if not qn_found or not qn_value_match:
            equal = False
            break
    return equal


def compare_qns(qn_dict, qn_dict2):
    qn_class = QuantumNumberClasses[qn_dict[get_xml_label(
        XMLLabelConstants.Class)]]
    value_label = get_xml_label(XMLLabelConstants.Value)

    val1 = None
    val2 = qn_dict2
    if qn_class is QuantumNumberClasses.Int:
        val1 = int(qn_dict[value_label])
        if isinstance(qn_dict2, dict):
            val2 = int(qn_dict2[value_label])
    elif qn_class is QuantumNumberClasses.Float:
        val1 = float(qn_dict[value_label])
        if isinstance(qn_dict2, dict):
            val2 = float(qn_dict2[value_label])
    elif qn_class is QuantumNumberClasses.Spin:
        spin_proj_label = get_xml_label(XMLLabelConstants.Projection)
        if isinstance(qn_dict2, dict):
            if spin_proj_label in qn_dict and spin_proj_label in qn_dict2:
                val1 = Spin(qn_dict[value_label], qn_dict[spin_proj_label])
                val2 = Spin(qn_dict2[value_label], qn_dict2[spin_proj_label])
            else:
                val1 = float(qn_dict[value_label])
                val2 = float(qn_dict2[value_label])
        else:
            val1 = Spin(qn_dict[value_label], qn_dict[spin_proj_label])
    else:
        raise ValueError("Unknown quantum number class " + qn_class)

    return val1 == val2


def merge_qn_props(qns_state, qns_particle):
    class_label = get_xml_label(XMLLabelConstants.Class)
    type_label = get_xml_label(XMLLabelConstants.Type)
    qns = deepcopy(qns_particle)
    for qn_entry in qns_state:
        qn_found = False
        for par_qn_entry in qns:
            if (StateQuantumNumberNames[qn_entry[type_label]]
                is StateQuantumNumberNames[par_qn_entry[type_label]] and
                    QuantumNumberClasses[qn_entry[class_label]]
                    is QuantumNumberClasses[par_qn_entry[class_label]]):
                qn_found = True
                par_qn_entry.update(qn_entry)
                break
        if not qn_found:
            qns.append(qn_entry)
    return qns
