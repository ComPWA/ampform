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
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from numpy import arange

from expertsystem import io
from expertsystem.data import (
    ParticleCollection,
    Spin,
)
from expertsystem.topology import StateTransitionGraph


StateWithSpins = Tuple[str, Sequence[float]]
StateDefinition = Union[str, StateWithSpins]


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


def create_spin_domain(
    list_of_magnitudes: List[float], set_projection_zero: bool = False
) -> List[Spin]:
    domain_list = []
    for mag in list_of_magnitudes:
        if set_projection_zero:
            domain_list.append(
                Spin(mag, 0)
                if isinstance(mag, int) or mag.is_integer()
                else Spin(mag, mag)
            )
        else:
            for proj in arange(-mag, mag + 1, 1.0):  # type: ignore
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
    Charmness = auto()
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


QNDefaultValues: Dict[StateQuantumNumberNames, Any] = {
    StateQuantumNumberNames.Charge: 0,
    StateQuantumNumberNames.IsoSpin: Spin(0.0, 0.0),
    StateQuantumNumberNames.Strangeness: 0,
    StateQuantumNumberNames.Charmness: 0,
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
    StateQuantumNumberNames.Charmness: QuantumNumberClasses.Int,
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
    def parse_from_dict(self, data_dict: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def convert_to_dict(
        self,
        qn_type: Union[InteractionQuantumNumberNames, StateQuantumNumberNames],
        qn_value: Any,
    ) -> Dict[str, Any]:
        pass


class _IntQNConverter(AbstractQNConverter):
    """Interface for converting `int` quantum numbers."""

    value_label = Labels.Value.name
    type_label = Labels.Type.name
    class_label = Labels.Class.name

    def parse_from_dict(self, data_dict: Dict[str, Any]) -> int:
        return int(data_dict[self.value_label])

    def convert_to_dict(
        self,
        qn_type: Union[InteractionQuantumNumberNames, StateQuantumNumberNames],
        qn_value: int,
    ) -> Dict[str, Any]:
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

    def parse_from_dict(self, data_dict: Dict[str, Any]) -> float:
        return float(data_dict[self.value_label])

    def convert_to_dict(
        self,
        qn_type: Union[InteractionQuantumNumberNames, StateQuantumNumberNames],
        qn_value: float,
    ) -> Dict[str, Any]:
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

    def __init__(self, parse_projection: bool = True) -> None:
        self.parse_projection = bool(parse_projection)

    def parse_from_dict(self, data_dict: Dict[str, Any]) -> Spin:
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

    def convert_to_dict(
        self,
        qn_type: Union[InteractionQuantumNumberNames, StateQuantumNumberNames],
        qn_value: Spin,
    ) -> Dict[str, Any]:
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


def is_boson(qn_dict: Dict[StateQuantumNumberNames, Any]) -> bool:
    spin_label = StateQuantumNumberNames.Spin
    return abs(qn_dict[spin_label].magnitude % 1) < 0.01


def get_particle_property(
    particle_properties: Dict[str, Any],
    qn_name: Union[
        ParticleDecayPropertyNames,  # width
        ParticlePropertyNames,  # mass
        StateQuantumNumberNames,  # quantum numbers
    ],
    converter: Optional[AbstractQNConverter] = None,
) -> Optional[Dict[str, Any]]:
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
        if isinstance(qn_name, StateQuantumNumberNames):
            property_value = QNDefaultValues.get(qn_name, property_value)
    return property_value


def get_interaction_property(
    interaction_properties: Dict[str, Any],
    qn_name: Union[InteractionQuantumNumberNames, StateQuantumNumberNames],
    converter: Optional[AbstractQNConverter] = None,
) -> Optional[Dict[str, Any]]:
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
        if (
            isinstance(qn_name, StateQuantumNumberNames)
            and qn_name in QNDefaultValues
        ):
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

    def __init__(
        self,
        ignored_qn_list: Optional[
            List[Union[StateQuantumNumberNames, InteractionQuantumNumberNames]]
        ] = None,
    ) -> None:
        if ignored_qn_list is None:
            ignored_qn_list = []
        self.ignored_qn_list: List[str] = [
            x.name
            for x in ignored_qn_list
            if isinstance(
                x, (StateQuantumNumberNames, InteractionQuantumNumberNames)
            )
        ]

    def compare_qn_numbers(
        self, qns1: List[Dict[str, Any]], qns2: List[Dict[str, Any]]
    ) -> bool:
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

    def __call__(
        self,
        props1: Dict[int, Dict[str, List[Dict[str, Any]]]],  # QuantumNumber
        props2: Dict[int, Dict[str, List[Dict[str, Any]]]],  # QuantumNumber
    ) -> bool:
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
            qns1: List[Dict[str, Any]] = []
            qns2: List[Dict[str, Any]] = []
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


def initialize_graph(  # pylint: disable=too-many-locals
    empty_topology: StateTransitionGraph,
    particles: ParticleCollection,
    initial_state: Sequence[StateDefinition],
    final_state: Sequence[StateDefinition],
    final_state_groupings: Optional[List[List[List[str]]]] = None,
) -> List[StateTransitionGraph]:
    is_edges = empty_topology.get_initial_state_edges()
    if len(initial_state) != len(is_edges):
        raise ValueError(
            "The graph initial state and the supplied initial"
            "state are of different size! "
            f"({len(is_edges)} !=  {len(initial_state)})"
        )
    fs_edges = empty_topology.get_final_state_edges()
    if len(final_state) != len(fs_edges):
        raise ValueError(
            "The graph initial state and the supplied initial"
            "state are of different size! "
            f"({len(fs_edges)} !=  {len(final_state)})"
        )

    # check if all initial and final state particles have spin projections set
    initial_state_with_projections = [
        __safe_set_spin_projections(x, particles) for x in initial_state
    ]
    final_state_with_projections = [
        __safe_set_spin_projections(x, particles) for x in final_state
    ]

    attached_is_edges = [
        empty_topology.get_originating_initial_state_edges(i)
        for i in empty_topology.nodes
    ]
    is_edge_particle_pairs = __calculate_combinatorics(
        is_edges, initial_state_with_projections, attached_is_edges
    )

    attached_fs_edges = [
        empty_topology.get_originating_final_state_edges(i)
        for i in empty_topology.nodes
    ]
    fs_edge_particle_pairs = __calculate_combinatorics(
        fs_edges,
        final_state_with_projections,
        attached_fs_edges,
        final_state_groupings,
    )

    new_graphs: List[StateTransitionGraph] = list()
    for initial_state_pair in is_edge_particle_pairs:
        for fs_pair in fs_edge_particle_pairs:
            merged_dicts = initial_state_pair.copy()
            merged_dicts.update(fs_pair)
            new_graphs.extend(
                __initialize_edges(empty_topology, merged_dicts, particles)
            )

    return new_graphs


def __safe_set_spin_projections(
    state: StateDefinition, particle_db: ParticleCollection
) -> StateWithSpins:
    if isinstance(state, str):
        particle_name = state
        particle = particle_db[state]
        spin_projections = arange(  # type: ignore
            -particle.spin, particle.spin + 1, 1.0
        ).tolist()
        if particle.mass == 0.0:
            if 0.0 in spin_projections:
                del spin_projections[spin_projections.index(0.0)]
        state = (particle_name, spin_projections)
    return state


def __calculate_combinatorics(
    edges: List[int],
    state_particles: Sequence[StateWithSpins],
    attached_external_edges_per_node: List[List[int]],
    allowed_particle_groupings: Optional[List[List[List[str]]]] = None,
) -> List[
    Dict[int, StateWithSpins]
]:  # pylint: disable=too-many-branches,too-many-locals
    combinatorics_list = [
        dict(zip(edges, particles))
        for particles in permutations(state_particles)
    ]

    # now initialize the attached external edge list with the particles
    comb_attached_ext_edges = [
        __initialize_external_edge_lists(attached_external_edges_per_node, x)
        for x in combinatorics_list
    ]

    # remove combinations with wrong particle groupings
    if allowed_particle_groupings:  # pylint: disable=too-many-nested-blocks
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


def __initialize_external_edge_lists(
    attached_external_edges_per_node: List[List[int]],
    edge_particle_mapping: Dict[int, StateWithSpins],
) -> List[List[StateWithSpins]]:
    init_edge_lists = []
    for edge_list in attached_external_edges_per_node:
        init_edge_lists.append(
            sorted([edge_particle_mapping[i] for i in edge_list])
        )
    return sorted(init_edge_lists)


def __initialize_edges(
    graph: StateTransitionGraph,
    edge_particle_dict: Dict[int, StateWithSpins],
    particle_db: ParticleCollection,
) -> List[StateTransitionGraph]:
    for edge_id, state_particle in edge_particle_dict.items():
        particle_name = state_particle[0]
        particle = particle_db[particle_name]
        particle_properties = io.xml.object_to_dict(particle)
        graph.edge_props[edge_id] = particle_properties

    # now add more quantum numbers given by user (spin_projection)
    new_graphs: List[StateTransitionGraph] = [graph]
    for edge_id, state_particle in edge_particle_dict.items():
        if isinstance(state_particle, str):
            continue
        spin_projections = state_particle[1]
        temp_graphs = new_graphs
        new_graphs = []
        for temp_graph in temp_graphs:
            new_graphs.extend(
                __populate_edge_with_spin_projections(
                    temp_graph, edge_id, spin_projections
                )
            )

    return new_graphs


def __populate_edge_with_spin_projections(
    graph: StateTransitionGraph,
    edge_id: int,
    spin_projections: Sequence[float],
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


def initialize_graphs_with_particles(
    graphs: List[StateTransitionGraph],
    allowed_particle_list: List[Dict[str, Any]],
) -> List[StateTransitionGraph]:
    initialized_graphs = []

    for graph in graphs:
        logging.debug("initializing graph...")
        intermediate_edges = graph.get_intermediate_state_edges()
        current_new_graphs = [graph]
        for int_edge_id in intermediate_edges:
            particle_edges = get_particle_candidates_for_state(
                graph.edge_props[int_edge_id], allowed_particle_list
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


def filter_particles(
    particle_db: ParticleCollection, allowed_particle_names: List[str],
) -> List[Dict[str, Any]]:
    """Filters `.ParticleCollection` based on the allowed particle names.

    Note this function currently also converts back to dict structures, which
    are still used internally by the propagation code.
    """
    mod_allowed_particle_list = []
    if len(allowed_particle_names) == 0:
        mod_allowed_particle_list = list(
            io.xml.object_to_dict(particle_db).values()
        )
    else:
        for allowed_particle in allowed_particle_names:
            if isinstance(allowed_particle, (int, str)):
                subset = particle_db.find_subset(allowed_particle)
                search_results = io.xml.object_to_dict(subset)
                if "Pid" in search_results:  # one particle only
                    mod_allowed_particle_list.append(search_results)
                else:  # several particles found
                    mod_allowed_particle_list += list(search_results.values())
            else:
                mod_allowed_particle_list.append(allowed_particle)
    return mod_allowed_particle_list


def get_particle_candidates_for_state(
    state: Dict[str, List[Dict[str, Any]]],
    allowed_particle_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    particle_edges: List[Dict[str, Any]] = []
    qns_label = Labels.QuantumNumber.name

    for allowed_state in allowed_particle_list:
        if __check_qns_equal(state[qns_label], allowed_state[qns_label]):
            temp_particle = deepcopy(allowed_state)
            temp_particle[qns_label] = __merge_qn_props(
                state[qns_label], allowed_state[qns_label]
            )
            particle_edges.append(temp_particle)
    return particle_edges


def __check_qns_equal(
    qns_state: List[Dict[str, Any]], qns_particle: List[Dict[str, Any]]
) -> bool:
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
                if __compare_qns(qn_entry, par_qn_entry):
                    qn_value_match = True
                break
        if not qn_found:
            # check if there is a default value
            qn_name = StateQuantumNumberNames[qn_entry[type_label]]
            if qn_name in QNDefaultValues:
                if __compare_qns(qn_entry, QNDefaultValues[qn_name]):
                    qn_found = True
                    qn_value_match = True

        if not qn_found or not qn_value_match:
            equal = False
            break
    return equal


def __compare_qns(qn_dict: Dict[str, Any], qn_dict2: Dict[str, Any]) -> bool:
    qn_class = QuantumNumberClasses[qn_dict[Labels.Class.name]]
    value_label = Labels.Value.name

    val1: Any = None
    val2: Any = qn_dict2
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


def __merge_qn_props(
    qns_state: List[Dict[str, Any]], qns_particle: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
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


def match_external_edges(graphs: List[StateTransitionGraph]) -> None:
    if not isinstance(graphs, list):
        raise TypeError("graphs argument is not of type list!")
    if not graphs:
        return
    ref_graph_id = 0
    _match_external_edge_ids(graphs, ref_graph_id, "get_final_state_edges")
    _match_external_edge_ids(graphs, ref_graph_id, "get_initial_state_edges")


def _match_external_edge_ids(  # pylint: disable=too-many-locals
    graphs: List[StateTransitionGraph],
    ref_graph_id: int,
    external_edge_getter_function: str,
) -> None:
    ref_graph = graphs[ref_graph_id]
    # create external edge to particle mapping
    ref_edge_id_particle_mapping = _create_edge_id_particle_mapping(
        ref_graph, external_edge_getter_function
    )

    for graph in graphs[:ref_graph_id] + graphs[ref_graph_id + 1 :]:
        edge_id_particle_mapping = _create_edge_id_particle_mapping(
            graph, external_edge_getter_function
        )
        # remove matching entries
        ref_mapping_copy = deepcopy(ref_edge_id_particle_mapping)
        edge_ids_mapping = {}
        for key, value in edge_id_particle_mapping.items():
            if key in ref_mapping_copy and value == ref_mapping_copy[key]:
                del ref_mapping_copy[key]
            else:
                for key_2, value_2 in ref_mapping_copy.items():
                    if value == value_2:
                        edge_ids_mapping[key] = key_2
                        del ref_mapping_copy[key_2]
                        break
        if len(ref_mapping_copy) != 0:
            raise ValueError(
                "Unable to match graphs, due to inherent graph"
                " structure mismatch"
            )
        swappings = _calculate_swappings(edge_ids_mapping)
        for edge_id1, edge_id2 in swappings.items():
            graph.swap_edges(edge_id1, edge_id2)


def perform_external_edge_identical_particle_combinatorics(
    graph: StateTransitionGraph,
) -> List[StateTransitionGraph]:
    """Create combinatorics clones of the `.StateTransitionGraph`.

    In case of identical particles in the initial or final state. Only
    identical particles, which do not enter or exit the same node allow for
    combinatorics!
    """
    if not isinstance(graph, StateTransitionGraph):
        raise TypeError("graph argument is not of type StateTransitionGraph!")
    temp_new_graphs = _external_edge_identical_particle_combinatorics(
        graph, "get_final_state_edges"
    )
    new_graphs = []
    for new_graph in temp_new_graphs:
        new_graphs.extend(
            _external_edge_identical_particle_combinatorics(
                new_graph, "get_initial_state_edges"
            )
        )
    return new_graphs


def _external_edge_identical_particle_combinatorics(
    graph: StateTransitionGraph, external_edge_getter_function: str,
) -> List[StateTransitionGraph]:
    # pylint: disable=too-many-locals
    new_graphs = [graph]
    edge_particle_mapping = _create_edge_id_particle_mapping(
        graph, external_edge_getter_function
    )
    identical_particle_groups: Dict[str, Set[int]] = {}
    for key, value in edge_particle_mapping.items():
        if value not in identical_particle_groups:
            identical_particle_groups[value] = set()
        identical_particle_groups[value].add(key)
    identical_particle_groups = {
        key: value
        for key, value in identical_particle_groups.items()
        if len(value) > 1
    }
    # now for each identical particle group perform all permutations
    for edge_group in identical_particle_groups.values():
        combinations = permutations(edge_group)
        graph_combinations = set()
        ext_edge_combinations = []
        ref_node_origin = graph.get_originating_node_list(edge_group)
        for comb in combinations:
            temp_edge_node_mapping = tuple(sorted(zip(comb, ref_node_origin)))
            if temp_edge_node_mapping not in graph_combinations:
                graph_combinations.add(temp_edge_node_mapping)
                ext_edge_combinations.append(dict(zip(edge_group, comb)))
        temp_new_graphs = []
        for new_graph in new_graphs:
            for combination in ext_edge_combinations:
                graph_copy = deepcopy(new_graph)
                swappings = _calculate_swappings(combination)
                for edge_id1, edge_id2 in swappings.items():
                    graph_copy.swap_edges(edge_id1, edge_id2)
                temp_new_graphs.append(graph_copy)
        new_graphs = temp_new_graphs
    return new_graphs


def _calculate_swappings(id_mapping: Dict[int, int]) -> OrderedDict:
    """Calculate edge id swappings.

    Its important to use an ordered dict as the swappings do not commute!
    """
    swappings: OrderedDict = OrderedDict()
    for key, value in id_mapping.items():
        # go through existing swappings and use them
        newkey = key
        while newkey in swappings:
            newkey = swappings[newkey]
        if value != newkey:
            swappings[value] = newkey
    return swappings


def _create_edge_id_particle_mapping(
    graph: StateTransitionGraph, external_edge_getter_function: str,
) -> Dict[int, str]:
    name_label = Labels.Name.name
    return {
        i: graph.edge_props[i][name_label]
        for i in getattr(graph, external_edge_getter_function)()
    }
