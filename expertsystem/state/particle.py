# pylint: disable=too-many-lines

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
    Callable,
    Dict,
    Generator,
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
    Particle,
    ParticleCollection,
    Spin,
)
from expertsystem.topology import StateTransitionGraph, Topology


StateWithSpins = Tuple[str, Sequence[float]]
StateDefinition = Union[str, StateWithSpins]
ParticleWithSpin = Tuple[Particle, float]


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


class KinematicRepresentation:
    def __init__(
        self,
        final_state: Optional[Union[Sequence[List[Any]], List[Any]]] = None,
        initial_state: Optional[Union[Sequence[List[Any]], List[Any]]] = None,
    ) -> None:
        self.__initial_state: Optional[List[List[Any]]] = None
        self.__final_state: Optional[List[List[Any]]] = None
        if initial_state is not None:
            self.__initial_state = self.__import(initial_state)
        if final_state is not None:
            self.__final_state = self.__import(final_state)

    @property
    def initial_state(self) -> Optional[List[List[Any]]]:
        return self.__initial_state

    @property
    def final_state(self) -> Optional[List[List[Any]]]:
        return self.__final_state

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KinematicRepresentation):
            return (
                self.initial_state == other.initial_state
                and self.final_state == other.final_state
            )
        raise ValueError(
            f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"initial_state={self.initial_state}, "
            f"final_state={self.final_state})"
        )

    def __contains__(self, other: object) -> bool:
        """Check if a `KinematicRepresentation` is contained within another.

        You can also compare with a `list` of `list` instances, such as:

        .. code-block::

            [["gamma", "pi0"], ["gamma", "pi0", "pi0"]]

        This list will be compared **only** with the
        `~KinematicRepresentation.final_state`!
        """

        def is_sublist(
            sub_representation: Optional[List[List[Any]]],
            main_representation: Optional[List[List[Any]]],
        ) -> bool:
            if main_representation is None:
                if sub_representation is None:
                    return True
                return False
            if sub_representation is None:
                return True
            for group in sub_representation:
                if group not in main_representation:
                    return False
            return True

        if isinstance(other, KinematicRepresentation):
            return is_sublist(
                other.initial_state, self.initial_state
            ) and is_sublist(other.final_state, self.final_state)
        if isinstance(other, list):
            for item in other:
                if not isinstance(item, list):
                    raise ValueError(
                        "Comparison representation needs to be a list of lists"
                    )
            return is_sublist(other, self.final_state)
        raise ValueError(
            f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}"
        )

    def __import(
        self, nested_list: Union[Sequence[Sequence[Any]], Sequence[Any]]
    ) -> List[List[Any]]:
        return self.__sort(self.__prepare(nested_list))

    def __prepare(
        self, nested_list: Union[Sequence[Sequence[Any]], Sequence[Any]]
    ) -> List[List[Any]]:
        if len(nested_list) == 0 or not isinstance(nested_list[0], list):
            nested_list = [nested_list]
        return [
            [self.__extract_particle_name(item) for item in sub_list]
            for sub_list in nested_list
        ]

    @staticmethod
    def __sort(nested_list: List[List[Any]]) -> List[List[Any]]:
        return sorted([sorted(sub_list) for sub_list in nested_list])

    @staticmethod
    def __extract_particle_name(item: object) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, (tuple, list)) and isinstance(item[0], str):
            return item[0]
        if isinstance(item, Particle):
            return item.name
        if isinstance(item, dict) and "Name" in item:
            return str(item["Name"])
        raise ValueError(
            f"Cannot extract particle name from {item.__class__.__name__}"
        )


def get_kinematic_representation(
    graph: StateTransitionGraph[StateWithSpins],
) -> KinematicRepresentation:
    r"""Group final or initial states by node, sorted by length of the group.

    The resulting sorted groups can be used to check whether two
    `.StateTransitionGraph` instances are kinematically identical. For
    instance, the following two graphs:

    .. code-block::

        J/psi -- 0 -- pi0
                  \
                   1 -- gamma
                    \
                     2 -- gamma
                      \
                       pi0

        J/psi -- 0 -- pi0
                  \
                   1 -- gamma
                    \
                     2 -- pi0
                      \
                       gamma

    both result in:

    .. code-block::

        kinematic_representation.final_state == \
            [["gamma", "gamma"], ["gamma", "gamma", "pi0"], \
             ["gamma", "gamma", "pi0", "pi0"]]
        kinematic_representation.initial_state == \
            [["J/psi"], ["J/psi"]]

    and are therefore kinematically identical. The nested lists are sorted (by
    `list` length and element content) for comparisons.

    Note: more precisely, the states represented here by a `str` only also have
    a list of allowed spin projections, for instance, :code:`("J/psi", [-1,
    +1])`. Note that a `tuple` is also sortable.
    """

    def get_state_groupings(
        edge_per_node_getter: Callable[[int], List[int]]
    ) -> List[List[int]]:
        return [edge_per_node_getter(i) for i in graph.nodes]

    def fill_groupings(grouping_with_ids: List[List[Any]]) -> List[List[Any]]:
        return [
            [graph.edge_props[edge_id] for edge_id in group]
            for group in grouping_with_ids
        ]

    initial_state_edge_groups = fill_groupings(
        get_state_groupings(graph.get_originating_initial_state_edges)
    )
    final_state_edge_groups = fill_groupings(
        get_state_groupings(graph.get_originating_final_state_edges)
    )
    return KinematicRepresentation(
        initial_state=initial_state_edge_groups,
        final_state=final_state_edge_groups,
    )


def particle_with_spin_projection_to_dict(instance: ParticleWithSpin) -> dict:
    particle, spin_projection = instance
    output = io.xml.object_to_dict(particle)
    for item in output["QuantumNumber"]:
        if item["Type"] == "Spin":
            item["Projection"] = spin_projection
    return output


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
    topology: Topology,
    particles: ParticleCollection,
    initial_state: Sequence[StateDefinition],
    final_state: Sequence[StateDefinition],
    final_state_groupings: Optional[
        Union[List[List[List[str]]], List[List[str]], List[str]]
    ] = None,
) -> List[StateTransitionGraph[ParticleWithSpin]]:
    def embed_in_list(some_list: List[Any]) -> List[List[Any]]:
        if not isinstance(some_list[0], list):
            return [some_list]
        return some_list

    allowed_kinematic_groupings = None
    if final_state_groupings is not None:
        final_state_groupings = embed_in_list(final_state_groupings)
        final_state_groupings = embed_in_list(final_state_groupings)
        allowed_kinematic_groupings = [
            KinematicRepresentation(final_state=grouping)
            for grouping in final_state_groupings
        ]

    kinematic_permutation_graphs = _generate_kinematic_permutations(
        topology=topology,
        particles=particles,
        initial_state=initial_state,
        final_state=final_state,
        allowed_kinematic_groupings=allowed_kinematic_groupings,
    )
    output_graphs = list()
    for kinematic_permutation in kinematic_permutation_graphs:
        spin_permutations = _generate_spin_permutations(
            kinematic_permutation, particles
        )
        output_graphs.extend(spin_permutations)
    return output_graphs


def _generate_kinematic_permutations(
    topology: Topology,
    particles: ParticleCollection,
    initial_state: Sequence[StateDefinition],
    final_state: Sequence[StateDefinition],
    allowed_kinematic_groupings: Optional[
        List[KinematicRepresentation]
    ] = None,
) -> List[StateTransitionGraph[StateWithSpins]]:
    def assert_number_of_states(
        state_definitions: Sequence, edge_ids: Sequence[int]
    ) -> None:
        if len(state_definitions) != len(edge_ids):
            raise ValueError(
                "Number of state definitions is not same as number of edge IDs:"
                f"(len({state_definitions}) != len({edge_ids})"
            )

    assert_number_of_states(initial_state, topology.get_initial_state_edges())
    assert_number_of_states(final_state, topology.get_final_state_edges())

    def is_allowed_grouping(
        kinematic_representation: KinematicRepresentation,
    ) -> bool:
        if allowed_kinematic_groupings is None:
            return True
        for allowed_kinematic_grouping in allowed_kinematic_groupings:
            if allowed_kinematic_grouping in kinematic_representation:
                return True
        return False

    initial_state_with_projections = _safe_set_spin_projections(
        initial_state, particles
    )
    final_state_with_projections = _safe_set_spin_projections(
        final_state, particles
    )

    graphs: List[StateTransitionGraph[StateWithSpins]] = list()
    kinematic_representations: List[KinematicRepresentation] = list()
    for permutation in _generate_outer_edge_permutations(
        topology,
        initial_state_with_projections,
        final_state_with_projections,
    ):
        graph: StateTransitionGraph[
            StateWithSpins
        ] = StateTransitionGraph.from_topology(topology)
        graph.edge_props.update(permutation)
        kinematic_representation = get_kinematic_representation(graph)
        if kinematic_representation in kinematic_representations:
            continue
        if not is_allowed_grouping(kinematic_representation):
            continue
        kinematic_representations.append(kinematic_representation)
        graphs.append(graph)

    return graphs


def _safe_set_spin_projections(
    list_of_states: Sequence[StateDefinition],
    particle_db: ParticleCollection,
) -> Sequence[StateWithSpins]:
    def safe_set_spin_projections(
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

    return [
        safe_set_spin_projections(state, particle_db)
        for state in list_of_states
    ]


def _generate_outer_edge_permutations(
    topology: Topology,
    initial_state: Sequence[StateWithSpins],
    final_state: Sequence[StateWithSpins],
) -> Generator[Dict[int, StateWithSpins], None, None]:
    initial_state_ids = topology.get_initial_state_edges()
    final_state_ids = topology.get_final_state_edges()
    for initial_state_permutation in permutations(initial_state):
        for final_state_permutation in permutations(final_state):
            yield dict(
                zip(
                    initial_state_ids + final_state_ids,
                    initial_state_permutation + final_state_permutation,
                )
            )


def _generate_spin_permutations(
    graph: StateTransitionGraph[StateWithSpins],
    particle_db: ParticleCollection,
) -> List[StateTransitionGraph[ParticleWithSpin]]:
    def populate_edge_with_spin_projections(
        uninitialized_graph: StateTransitionGraph[ParticleWithSpin],
        edge_id: int,
        state: StateWithSpins,
    ) -> List[StateTransitionGraph[ParticleWithSpin]]:
        particle_name, spin_projections = state
        particle = particle_db[particle_name]
        output_graph = []
        for projection in spin_projections:
            graph_copy = deepcopy(uninitialized_graph)
            graph_copy.edge_props[edge_id] = (particle, projection)
            output_graph.append(graph_copy)
        return output_graph

    edge_particle_dict = {
        edge_id: graph.edge_props[edge_id]
        for edge_id in graph.get_initial_state_edges()
        + graph.get_final_state_edges()
    }

    # now add more quantum numbers given by user (spin_projection)
    uninitialized_graph = StateTransitionGraph.from_topology(graph)
    output_graphs: List[StateTransitionGraph[ParticleWithSpin]] = [
        uninitialized_graph
    ]
    for edge_id, state in edge_particle_dict.items():
        temp_graphs = output_graphs
        output_graphs = []
        for temp_graph in temp_graphs:
            output_graphs.extend(
                populate_edge_with_spin_projections(temp_graph, edge_id, state)
            )

    return output_graphs


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
        are still used internally by the propagation code.
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
    graph: StateTransitionGraph,
    external_edge_getter_function: str,
) -> Dict[int, str]:
    name_label = Labels.Name.name
    return {
        i: graph.edge_props[i][name_label]
        for i in getattr(graph, external_edge_getter_function)()
    }
