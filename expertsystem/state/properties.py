"""Collection of data structures and functions for particle information.

This module defines a particle as a collection of quantum numbers and things
related to this.
"""

import json
import logging
from collections import OrderedDict
from copy import deepcopy
from itertools import permutations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Union,
)

from numpy import arange

from expertsystem import io
from expertsystem.data import (
    ParticleCollection,
    Spin,
)
from expertsystem.nested_dicts import (
    AbstractQNConverter,
    InteractionQuantumNumberNames,
    Labels,
    ParticleDecayPropertyNames,
    ParticlePropertyNames,
    QNClassConverterMapping,
    QNDefaultValues,
    QNNameClassMapping,
    QuantumNumberClasses,
    StateQuantumNumberNames,
)
from expertsystem.topology import StateTransitionGraph


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


def get_particle_property(
    particle_properties: Dict[str, Any],
    qn_name: Union[
        ParticleDecayPropertyNames,  # width
        ParticlePropertyNames,  # mass
        StateQuantumNumberNames,  # quantum numbers
    ],
    converter: Optional[AbstractQNConverter] = None,
) -> Optional[Union[Spin, float]]:
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
) -> Optional[Union[Spin, float]]:
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


def initialize_graphs_with_particles(
    graphs: List[StateTransitionGraph[dict]],
    allowed_particle_list: List[Dict[str, Any]],
) -> List[StateTransitionGraph[dict]]:
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
    particle_db: ParticleCollection,
    allowed_particle_names: List[str],
) -> List[Dict[str, Any]]:
    """Filters `.ParticleCollection` based on the allowed particle names.

    .. note::
        This function currently also converts back to `dict` structures, which
        are still used internally by the solver code.
    """
    mod_allowed_particle_list = []
    if len(allowed_particle_names) == 0:
        mod_allowed_particle_list = list(
            io.xml.object_to_dict(particle_db).values()
        )
    else:
        for allowed_particle in allowed_particle_names:
            if isinstance(allowed_particle, int):
                particle = particle_db.find(allowed_particle)
                particle_dict = io.xml.object_to_dict(particle)
                mod_allowed_particle_list.append(particle_dict)
            elif isinstance(allowed_particle, str):
                subset = particle_db.filter(
                    lambda p: allowed_particle  # pylint: disable=cell-var-from-loop
                    in p.name
                )
                particle_dicts = io.xml.object_to_dict(subset)
                mod_allowed_particle_list += list(particle_dicts.values())
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
    graphs: List[StateTransitionGraph[dict]],
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
    graph: StateTransitionGraph,
    external_edge_getter_function: str,
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
    graph: StateTransitionGraph[dict],
    external_edge_getter_function: str,
) -> Dict[int, str]:
    name_label = Labels.Name.name
    return {
        i: graph.edge_props[i][name_label]
        for i in getattr(graph, external_edge_getter_function)()
    }
