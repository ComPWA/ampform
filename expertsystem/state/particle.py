"""Collection of data structures and functions for particle information.

This module defines a particle as a collection of quantum numbers and things
related to this.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from enum import Enum, auto
from itertools import permutations
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from numpy import arange

from expertsystem import io
from expertsystem.data import (
    Parity,
    Particle,
    ParticleCollection,
    QuantumState,
    Spin,
)
from expertsystem.topology.graph import (
    StateTransitionGraph,
    get_final_state_edges,
    get_initial_state_edges,
    get_intermediate_state_edges,
    get_originating_final_state_edges,
    get_originating_initial_state_edges,
)


class Labels(Enum):
    """Labels that are useful in the particle module."""

    Class = auto()
    Component = auto()
    DecayInfo = auto()
    Name = auto()
    Parameter = auto()
    Pid = auto()
    PreFactor = auto()
    Projection = auto()
    QuantumNumber = auto()
    Type = auto()
    Value = auto()


def create_spin_domain(list_of_magnitudes, set_projection_zero=False):
    domain_list = []
    for mag in list_of_magnitudes:
        if set_projection_zero:
            domain_list.append(Spin(mag, 0.0))
        else:
            for proj in arange(-mag, mag + 1, 1.0):
                domain_list.append(Spin(mag, proj))
    return domain_list


class QuantumNumberClasses(Enum):
    """Types of quantum number classes in the form of an enumerate."""

    Int = auto()
    Float = auto()
    Spin = auto()


class StateQuantumNumberNames(Enum):
    """Definition of quantum number names for states."""

    BaryonNumber = auto()
    Bottomness = auto()
    Charge = auto()
    Charm = auto()
    CParity = auto()
    ElectronLN = auto()
    GParity = auto()
    IsoSpin = auto()
    MuonLN = auto()
    Parity = auto()
    Spin = auto()
    Strangeness = auto()
    TauLN = auto()
    Topness = auto()


class ParticlePropertyNames(Enum):
    """Definition of properties names of particles."""

    Pid = auto()
    Mass = auto()


class ParticleDecayPropertyNames(Enum):
    """Definition of decay properties names of particles."""

    Width = auto()


class InteractionQuantumNumberNames(Enum):
    """Definition of quantum number names for interaction nodes."""

    L = auto()
    S = auto()
    ParityPrefactor = auto()


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
    StateQuantumNumberNames.TauLN: 0,
}

QNNameClassMapping = {
    StateQuantumNumberNames.Charge: QuantumNumberClasses.Int,
    StateQuantumNumberNames.ElectronLN: QuantumNumberClasses.Int,
    StateQuantumNumberNames.MuonLN: QuantumNumberClasses.Int,
    StateQuantumNumberNames.TauLN: QuantumNumberClasses.Int,
    StateQuantumNumberNames.BaryonNumber: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Spin: QuantumNumberClasses.Spin,
    StateQuantumNumberNames.Parity: QuantumNumberClasses.Int,
    StateQuantumNumberNames.CParity: QuantumNumberClasses.Int,
    StateQuantumNumberNames.GParity: QuantumNumberClasses.Int,
    StateQuantumNumberNames.IsoSpin: QuantumNumberClasses.Spin,
    StateQuantumNumberNames.Strangeness: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Charm: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Bottomness: QuantumNumberClasses.Int,
    StateQuantumNumberNames.Topness: QuantumNumberClasses.Int,
    InteractionQuantumNumberNames.L: QuantumNumberClasses.Spin,
    InteractionQuantumNumberNames.S: QuantumNumberClasses.Spin,
    InteractionQuantumNumberNames.ParityPrefactor: QuantumNumberClasses.Int,
    ParticlePropertyNames.Pid: QuantumNumberClasses.Int,
    ParticlePropertyNames.Mass: QuantumNumberClasses.Float,
    ParticleDecayPropertyNames.Width: QuantumNumberClasses.Float,
}


class AbstractQNConverter(ABC):
    """Abstract interface for a quantum number converter."""

    @abstractmethod
    def parse_from_dict(self, data_dict):
        pass

    @abstractmethod
    def convert_to_dict(self, qn_type, qn_value):
        pass


class _IntQNConverter(AbstractQNConverter):
    """Interface for converting `int` quantum numbers."""

    value_label = Labels.Value.name
    type_label = Labels.Type.name
    class_label = Labels.Class.name

    def parse_from_dict(self, data_dict):
        return int(data_dict[self.value_label])

    def convert_to_dict(self, qn_type, qn_value):
        return {
            self.type_label: qn_type.name,
            self.class_label: QuantumNumberClasses.Int.name,
            self.value_label: qn_value,
        }


class _FloatQNConverter(AbstractQNConverter):
    """Interface for converting `float` quantum numbers."""

    value_label = Labels.Value.name
    type_label = Labels.Type.name
    class_label = Labels.Class.name

    def parse_from_dict(self, data_dict):
        return float(data_dict[self.value_label])

    def convert_to_dict(self, qn_type, qn_value):
        return {
            self.type_label: qn_type.name,
            self.class_label: QuantumNumberClasses.Float.name,
            self.value_label: qn_value,
        }


class _SpinQNConverter(AbstractQNConverter):
    """Interface for converting `.Spin` quantum numbers."""

    type_label = Labels.Type.name
    class_label = Labels.Class.name
    value_label = Labels.Value.name
    proj_label = Labels.Projection.name

    def __init__(self, parse_projection=True):
        self.parse_projection = parse_projection

    def parse_from_dict(self, data_dict):
        mag = data_dict[self.value_label]
        proj = 0.0
        if self.parse_projection:
            if self.proj_label not in data_dict:
                if float(mag) != 0.0:
                    raise ValueError(
                        "No projection set for spin-like quantum number!"
                    )
            else:
                proj = data_dict[self.proj_label]
        return Spin(mag, proj)

    def convert_to_dict(self, qn_type, qn_value):
        return {
            self.type_label: qn_type.name,
            self.class_label: QuantumNumberClasses.Spin.name,
            self.value_label: qn_value.magnitude,
            self.proj_label: qn_value.projection,
        }


QNClassConverterMapping = {
    QuantumNumberClasses.Int: _IntQNConverter(),
    QuantumNumberClasses.Float: _FloatQNConverter(),
    QuantumNumberClasses.Spin: _SpinQNConverter(),
}


def is_boson(qn_dict):
    spin_label = StateQuantumNumberNames.Spin
    return abs(qn_dict[spin_label].magnitude % 1) < 0.01


DATABASE = ParticleCollection()


def load_particles(filename: str) -> None:
    """Add entries to the particle database from a custom config file.

    By default, the expert system loads the particle database from the XML file
    :file:`particle_list.yml` that comes with the `expertsystem`. Use
    this function to append or overwrite definitions in the particle database.

    .. note::
        If a particle name in the loaded XML file already exists in the
        particle database, the one in the particle database will be
        overwritten.
    """
    particle_collection = io.load_particle_collection(filename)
    DATABASE.merge(particle_collection)


def get_particle_property(particle_properties, qn_name, converter=None):
    # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks
    qns_label = Labels.QuantumNumber.name
    type_label = Labels.Type.name
    value_label = Labels.Value.name

    found_prop = None
    if isinstance(qn_name, StateQuantumNumberNames):
        particle_qns = particle_properties[qns_label]
        for quantum_number in particle_qns:
            if quantum_number[type_label] == qn_name.name:
                found_prop = quantum_number
                break
    else:
        for key, val in particle_properties.items():
            if key == qn_name.name:
                found_prop = {value_label: val}
                break
            if key == "Parameter" and val[type_label] == qn_name.name:
                # parameters have a separate value tag
                tagname = Labels.Value.name
                found_prop = {value_label: val[tagname]}
                break
            if key == Labels.DecayInfo.name:
                for decinfo_key, decinfo_val in val.items():
                    if decinfo_key == qn_name.name:
                        found_prop = {value_label: decinfo_val}
                        break
                    if decinfo_key == "Parameter":
                        if not isinstance(decinfo_val, list):
                            decinfo_val = [decinfo_val]
                        for parval in decinfo_val:
                            if parval[type_label] == qn_name.name:
                                # parameters have a separate value tag
                                tagname = Labels.Value.name
                                found_prop = {value_label: parval[tagname]}
                                break
                        if found_prop:
                            break
            if found_prop:
                break
    # check for default value
    property_value = None
    if found_prop is not None:
        if converter is None:
            converter = QNClassConverterMapping[QNNameClassMapping[qn_name]]
        property_value = converter.parse_from_dict(found_prop)
    else:
        property_value = QNDefaultValues.get(qn_name, property_value)
    return property_value


def get_interaction_property(interaction_properties, qn_name, converter=None):
    qns_label = Labels.QuantumNumber.name
    type_label = Labels.Type.name

    found_prop = None
    if isinstance(qn_name, InteractionQuantumNumberNames):
        interaction_qns = interaction_properties[qns_label]
        for quantum_number in interaction_qns:
            if quantum_number[type_label] == qn_name.name:
                found_prop = quantum_number
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
        else:
            logging.warning(
                "Requested quantum number %s"
                " was not found in the interaction properties."
                "\nAlso no default setting for this quantum"
                " number is available. Perhaps you are using the"
                " wrong formalism?",
                str(qn_name),
            )
    return property_value


class CompareGraphElementPropertiesFunctor:
    """Functor for comparing graph elements."""

    def __init__(self, ignored_qn_list=None):
        if ignored_qn_list is None:
            ignored_qn_list = []
        self.ignored_qn_list = [
            x.name
            for x in ignored_qn_list
            if isinstance(
                x, (StateQuantumNumberNames, InteractionQuantumNumberNames)
            )
        ]

    def compare_qn_numbers(self, qns1, qns2):
        new_qns1 = {}
        new_qns2 = {}
        type_label = Labels.Type.name
        for quantum_number in qns1:
            if quantum_number[type_label] not in self.ignored_qn_list:
                temp_qn_dict = dict(quantum_number)
                type_name = temp_qn_dict[type_label]
                del temp_qn_dict[type_label]
                new_qns1[type_name] = temp_qn_dict
        for quantum_number in qns2:
            if quantum_number[type_label] not in self.ignored_qn_list:
                temp_qn_dict = dict(quantum_number)
                type_name = temp_qn_dict[type_label]
                del temp_qn_dict[type_label]
                new_qns2[type_name] = temp_qn_dict
        return json.loads(
            json.dumps(new_qns1, sort_keys=True), object_pairs_hook=OrderedDict
        ) == json.loads(
            json.dumps(new_qns2, sort_keys=True), object_pairs_hook=OrderedDict
        )

    def __call__(self, props1, props2):
        # for more speed first compare the names (if they exist)

        name_label = Labels.Name.name
        names1 = {
            k: v[name_label] for k, v in props1.items() if name_label in v
        }
        names2 = {
            k: v[name_label] for k, v in props2.items() if name_label in v
        }
        if set(names1.keys()) != set(names2.keys()):
            return False
        for k in names1.keys():
            if names1[k] != names2[k]:
                return False

        # then compare the qn lists (if they exist)
        qns_label = Labels.QuantumNumber.name
        for ele_id, props in props1.items():
            qns1 = []
            qns2 = []
            if qns_label in props:
                qns1 = props[qns_label]
            if ele_id in props2 and qns_label in props2[ele_id]:
                qns2 = props2[ele_id][qns_label]
            if not self.compare_qn_numbers(qns1, qns2):
                return False
        # if they are equal we have to make a deeper comparison
        copy_props1 = deepcopy(props1)
        copy_props2 = deepcopy(props2)
        # remove the qn dicts
        qns_label = Labels.QuantumNumber.name
        for ele_id in props1.keys():
            if qns_label in copy_props1[ele_id]:
                del copy_props1[ele_id][qns_label]
            if qns_label in copy_props2[ele_id]:
                del copy_props2[ele_id][qns_label]

        if json.loads(
            json.dumps(copy_props1, sort_keys=True),
            object_pairs_hook=OrderedDict,
        ) != json.loads(
            json.dumps(copy_props2, sort_keys=True),
            object_pairs_hook=OrderedDict,
        ):
            return False
        return True


def initialize_graph(graph, initial_state, final_state, final_state_groupings):
    is_edges = get_initial_state_edges(graph)
    if len(initial_state) != len(is_edges):
        raise ValueError(
            "The graph initial state and the supplied initial"
            "state are of different size! ("
            + str(len(is_edges))
            + " != "
            + str(len(initial_state))
            + ")"
        )
    fs_edges = get_final_state_edges(graph)
    if len(final_state) != len(fs_edges):
        raise ValueError(
            "The graph final state and the supplied final"
            "state are of different size! ("
            + str(len(fs_edges))
            + " != "
            + str(len(final_state))
            + ")"
        )

    # check if all initial and final state particles have spin projections set
    initial_state = [check_if_spin_projections_set(x) for x in initial_state]
    final_state = [check_if_spin_projections_set(x) for x in final_state]

    attached_is_edges = [
        get_originating_initial_state_edges(graph, i) for i in graph.nodes
    ]
    is_edge_particle_pairs = calculate_combinatorics(
        is_edges, initial_state, attached_is_edges
    )
    attached_fs_edges = [
        get_originating_final_state_edges(graph, i) for i in graph.nodes
    ]
    fs_edge_particle_pairs = calculate_combinatorics(
        fs_edges, final_state, attached_fs_edges, final_state_groupings
    )

    new_graphs = []
    for is_pair in is_edge_particle_pairs:
        for fs_pair in fs_edge_particle_pairs:
            merged_dicts = is_pair.copy()
            merged_dicts.update(fs_pair)
            new_graphs.extend(initialize_edges(graph, merged_dicts))

    return new_graphs


def check_if_spin_projections_set(
    state: Union[str, Tuple[str, List[float]]]
) -> Tuple[str, List[float]]:
    if isinstance(state, str):
        particle_name = state
        particle = DATABASE[state]
        spin_projections = arange(
            -particle.state.spin, particle.state.spin + 1, 1.0
        ).tolist()
        if particle.mass == 0.0:
            if 0.0 in spin_projections:
                del spin_projections[spin_projections.index(0.0)]
        state = (particle_name, spin_projections)
    return state


def calculate_combinatorics(
    edges,
    state_particles,
    attached_external_edges_per_node,
    allowed_particle_groupings=None,
):  # pylint: disable=too-many-branches,too-many-locals,too-many-nested-blocks
    combinatorics_list = [
        dict(zip(edges, particles))
        for particles in permutations(state_particles)
    ]

    # now initialize the attached external edge list with the particles
    comb_attached_ext_edges = [
        initialize_external_edge_lists(attached_external_edges_per_node, x)
        for x in combinatorics_list
    ]

    # remove combinations with wrong particle groupings
    if allowed_particle_groupings:
        sorted_allowed_particle_groupings = [
            sorted(sorted(group) for group in grouping)
            for grouping in allowed_particle_groupings
        ]
        combinations_to_remove = set()
        index_counter = 0
        for attached_ext_edge_comb in comb_attached_ext_edges:
            found_valid_grouping = False
            for particle_grouping in sorted_allowed_particle_groupings:
                # check if this grouping is available in this graph
                valid_grouping = True
                for grouping in particle_grouping:
                    found = False
                    for ext_edge_group in attached_ext_edge_comb:
                        if (
                            sorted([group[0] for group in ext_edge_group])
                            == grouping
                        ):
                            found = True
                            break
                    if not found:
                        valid_grouping = False
                        break
                if valid_grouping:
                    found_valid_grouping = True
            if not found_valid_grouping:
                combinations_to_remove.add(index_counter)
            index_counter += 1
        for i in sorted(combinations_to_remove, reverse=True):
            del comb_attached_ext_edges[i]
            del combinatorics_list[i]

    # remove equal combinations
    combinations_to_remove = set()
    for i, _ in enumerate(comb_attached_ext_edges):
        for j in range(i + 1, len(comb_attached_ext_edges)):
            if comb_attached_ext_edges[i] == comb_attached_ext_edges[j]:
                combinations_to_remove.add(i)
                break

    for comb_index in sorted(combinations_to_remove, reverse=True):
        del combinatorics_list[comb_index]
    return combinatorics_list


def initialize_external_edge_lists(
    attached_external_edges_per_node, edge_particle_mapping
):
    init_edge_lists = []
    for edge_list in attached_external_edges_per_node:
        init_edge_lists.append(
            sorted([edge_particle_mapping[i] for i in edge_list])
        )
    return sorted(init_edge_lists)


def initialize_edges(
    graph: StateTransitionGraph,
    edge_particle_dict: Dict[int, Tuple[str, List[float]]],
) -> List[StateTransitionGraph]:
    for edge_id, state_particle in edge_particle_dict.items():
        particle_name = state_particle[0]
        particle = DATABASE[particle_name]
        particle_properties = io.xml.object_to_dict(particle)
        graph.edge_props[edge_id] = particle_properties

    # now add more quantum numbers given by user (spin_projection)
    new_graphs: List[StateTransitionGraph] = [graph]
    for edge_id, state_particle in edge_particle_dict.items():
        spin_projections = state_particle[1]
        temp_graphs = new_graphs
        new_graphs = []
        for temp_graph in temp_graphs:
            new_graphs.extend(
                populate_edge_with_spin_projections(
                    temp_graph, edge_id, spin_projections
                )
            )

    return new_graphs


def populate_edge_with_spin_projections(
    graph: StateTransitionGraph, edge_id: int, spin_projections: List[float]
) -> List[StateTransitionGraph]:
    qns_label = Labels.QuantumNumber.name
    type_label = Labels.Type.name
    class_label = Labels.Class.name
    type_value = StateQuantumNumberNames.Spin
    class_value = QNNameClassMapping[type_value]

    new_graphs = []

    qn_list = graph.edge_props[edge_id][qns_label]
    index_list = [
        qn_list.index(x)
        for x in qn_list
        if (type_label in x and class_label in x)
        and (
            x[type_label] == type_value.name
            and x[class_label] == class_value.name
        )
    ]
    if index_list:
        for spin_proj in spin_projections:
            graph_copy = deepcopy(graph)
            graph_copy.edge_props[edge_id][qns_label][index_list[0]][
                Labels.Projection.name
            ] = spin_proj
            new_graphs.append(graph_copy)

    return new_graphs


def initialize_graphs_with_particles(graphs, allowed_particle_list=None):
    if allowed_particle_list is None:
        allowed_particle_list = []
    initialized_graphs = []
    mod_allowed_particle_list = initialize_allowed_particle_list(
        allowed_particle_list
    )

    for graph in graphs:
        logging.debug("initializing graph...")
        intermediate_edges = get_intermediate_state_edges(graph)
        current_new_graphs = [graph]
        for int_edge_id in intermediate_edges:
            particle_edges = get_particle_candidates_for_state(
                graph.edge_props[int_edge_id], mod_allowed_particle_list
            )
            if len(particle_edges) == 0:
                logging.debug("Did not find any particle candidates for")
                logging.debug("edge id: %d", int_edge_id)
                logging.debug("edge properties:")
                logging.debug(graph.edge_props[int_edge_id])
            new_graphs_temp = []
            for current_new_graph in current_new_graphs:
                for particle_edge in particle_edges:
                    temp_graph = deepcopy(current_new_graph)
                    temp_graph.edge_props[int_edge_id] = particle_edge
                    new_graphs_temp.append(temp_graph)
            current_new_graphs = new_graphs_temp

        initialized_graphs.extend(current_new_graphs)
    return initialized_graphs


def initialize_allowed_particle_list(allowed_particle_list):
    mod_allowed_particle_list = []
    if len(allowed_particle_list) == 0:
        mod_allowed_particle_list = list(
            io.xml.object_to_dict(DATABASE).values()
        )
    else:
        for allowed_particle in allowed_particle_list:
            if isinstance(allowed_particle, (int, str)):
                subset = DATABASE.find_subset(allowed_particle)
                search_results = io.xml.object_to_dict(subset)
                if "Pid" in search_results:  # one particle only
                    mod_allowed_particle_list.append(search_results)
                else:  # several particles found
                    mod_allowed_particle_list += list(search_results.values())
            else:
                mod_allowed_particle_list.append(allowed_particle)
    return mod_allowed_particle_list


def get_particle_candidates_for_state(state, allowed_particle_list):
    particle_edges = []
    qns_label = Labels.QuantumNumber.name

    for allowed_state in allowed_particle_list:
        if check_qns_equal(state[qns_label], allowed_state[qns_label]):
            temp_particle = deepcopy(allowed_state)
            temp_particle[qns_label] = merge_qn_props(
                state[qns_label], allowed_state[qns_label]
            )
            particle_edges.append(temp_particle)
    return particle_edges


def check_qns_equal(qns_state, qns_particle):
    equal = True
    class_label = Labels.Class.name
    type_label = Labels.Type.name
    for qn_entry in qns_state:
        qn_found = False
        qn_value_match = False
        for par_qn_entry in qns_particle:
            # first check if the type and class of these
            # qn entries are the same
            if (
                StateQuantumNumberNames[qn_entry[type_label]]
                is StateQuantumNumberNames[par_qn_entry[type_label]]
                and QuantumNumberClasses[qn_entry[class_label]]
                is QuantumNumberClasses[par_qn_entry[class_label]]
            ):
                qn_found = True
                if compare_qns(qn_entry, par_qn_entry):
                    qn_value_match = True
                break
        if not qn_found:
            # check if there is a default value
            qn_name = StateQuantumNumberNames[qn_entry[type_label]]
            if qn_name in QNDefaultValues:
                if compare_qns(qn_entry, QNDefaultValues[qn_name]):
                    qn_found = True
                    qn_value_match = True

        if not qn_found or not qn_value_match:
            equal = False
            break
    return equal


def compare_qns(qn_dict, qn_dict2):
    qn_class = QuantumNumberClasses[qn_dict[Labels.Class.name]]
    value_label = Labels.Value.name

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
        spin_proj_label = Labels.Projection.name
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
    class_label = Labels.Class.name
    type_label = Labels.Type.name
    qns = deepcopy(qns_particle)
    for qn_entry in qns_state:
        qn_found = False
        for par_qn_entry in qns:
            if (
                StateQuantumNumberNames[qn_entry[type_label]]
                is StateQuantumNumberNames[par_qn_entry[type_label]]
                and QuantumNumberClasses[qn_entry[class_label]]
                is QuantumNumberClasses[par_qn_entry[class_label]]
            ):
                qn_found = True
                par_qn_entry.update(qn_entry)
                break
        if not qn_found:
            qns.append(qn_entry)
    return qns


def create_particle(  # pylint: disable=too-many-arguments,too-many-locals
    template_particle: Particle,
    name: Optional[str] = None,
    pid: Optional[int] = None,
    mass: Optional[float] = None,
    width: Optional[float] = None,
    charge: Optional[int] = None,
    spin: Optional[float] = None,
    isospin: Optional[Spin] = None,
    strangeness: Optional[int] = None,
    charmness: Optional[int] = None,
    bottomness: Optional[int] = None,
    topness: Optional[int] = None,
    baryon_number: Optional[int] = None,
    electron_lepton_number: Optional[int] = None,
    muon_lepton_number: Optional[int] = None,
    tau_lepton_number: Optional[int] = None,
    parity: Optional[int] = None,
    c_parity: Optional[int] = None,
    g_parity: Optional[int] = None,
    state: Optional[QuantumState[float]] = None,
) -> Particle:
    if state is not None:
        new_state = state
    else:
        new_state = QuantumState[float](
            spin=spin if spin else template_particle.state.spin,
            charge=charge if charge else template_particle.state.charge,
            strangeness=strangeness
            if strangeness
            else template_particle.state.strangeness,
            charmness=charmness
            if charmness
            else template_particle.state.charmness,
            bottomness=bottomness
            if bottomness
            else template_particle.state.bottomness,
            topness=topness if topness else template_particle.state.topness,
            baryon_number=baryon_number
            if baryon_number
            else template_particle.state.baryon_number,
            electron_lepton_number=electron_lepton_number
            if electron_lepton_number
            else template_particle.state.electron_lepton_number,
            muon_lepton_number=muon_lepton_number
            if muon_lepton_number
            else template_particle.state.muon_lepton_number,
            tau_lepton_number=tau_lepton_number
            if tau_lepton_number
            else template_particle.state.tau_lepton_number,
            isospin=template_particle.state.isospin
            if isospin is None
            else template_particle.state.isospin,
            parity=template_particle.state.parity
            if parity is None
            else Parity(parity),
            c_parity=template_particle.state.c_parity
            if c_parity is None
            else Parity(c_parity),
            g_parity=template_particle.state.g_parity
            if g_parity is None
            else Parity(g_parity),
        )
    new_particle = Particle(
        name=name if name else template_particle.name,
        pid=pid if pid else template_particle.pid,
        mass=mass if mass else template_particle.mass,
        width=width if width else template_particle.width,
        state=new_state,
    )
    return new_particle


def create_antiparticle(
    template_particle: Particle, new_name: str = None
) -> Particle:
    isospin: Optional[Spin] = None
    if template_particle.state.isospin:
        isospin = -template_particle.state.isospin
    parity: Optional[Parity] = None
    if template_particle.state.parity is not None:
        if template_particle.state.spin.is_integer():
            parity = template_particle.state.parity
        else:
            parity = -template_particle.state.parity
    return Particle(
        name=new_name if new_name else "anti-" + template_particle.name,
        pid=-template_particle.pid,
        mass=template_particle.mass,
        width=template_particle.width,
        state=QuantumState[float](
            charge=-template_particle.state.charge,
            spin=template_particle.state.spin,
            isospin=isospin,
            strangeness=-template_particle.state.strangeness,
            charmness=-template_particle.state.charmness,
            bottomness=-template_particle.state.bottomness,
            topness=-template_particle.state.topness,
            baryon_number=-template_particle.state.baryon_number,
            electron_lepton_number=-template_particle.state.electron_lepton_number,
            muon_lepton_number=-template_particle.state.muon_lepton_number,
            tau_lepton_number=-template_particle.state.tau_lepton_number,
            parity=parity,
            c_parity=template_particle.state.c_parity,
            g_parity=template_particle.state.g_parity,
        ),
    )
