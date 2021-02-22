"""Functions that steer operations of the `expertsystem`."""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set, Tuple, Type

import attr

from expertsystem.particle import Parity, Particle, ParticleCollection
from expertsystem.reaction.default_settings import InteractionTypes
from expertsystem.reaction.quantum_numbers import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumber,
    NodeQuantumNumbers,
    ParticleWithSpin,
)
from expertsystem.reaction.solving import (
    GraphEdgePropertyMap,
    GraphNodePropertyMap,
    GraphSettings,
)
from expertsystem.reaction.topology import StateTransitionGraph

Strength = float

GraphSettingsGroups = Dict[
    Strength, List[Tuple[StateTransitionGraph, GraphSettings]]
]


def create_edge_properties(
    particle: Particle,
    spin_projection: Optional[float] = None,
) -> GraphEdgePropertyMap:
    edge_qn_mapping: Dict[str, Type[EdgeQuantumNumber]] = {
        qn_name: qn_type
        for qn_name, qn_type in EdgeQuantumNumbers.__dict__.items()
        if not qn_name.startswith("__")
    }  # Note using attr.fields does not work here because init=False
    property_map: GraphEdgePropertyMap = {}
    isospin = None
    for qn_name, value in attr.asdict(particle).items():
        if isinstance(value, Parity):
            value = value.value
        if qn_name in edge_qn_mapping:
            property_map[edge_qn_mapping[qn_name]] = value
        else:
            if "isospin" in qn_name:
                isospin = value
            elif "spin" in qn_name:
                property_map[EdgeQuantumNumbers.spin_magnitude] = value

    if spin_projection is not None:
        property_map[EdgeQuantumNumbers.spin_projection] = spin_projection
    if isospin is not None:
        property_map[EdgeQuantumNumbers.isospin_magnitude] = isospin.magnitude
        property_map[
            EdgeQuantumNumbers.isospin_projection
        ] = isospin.projection
    return property_map


def create_node_properties(
    node_props: InteractionProperties,
) -> GraphNodePropertyMap:
    node_qn_mapping: Dict[str, Type[NodeQuantumNumber]] = {
        qn_name: qn_type
        for qn_name, qn_type in NodeQuantumNumbers.__dict__.items()
        if not qn_name.startswith("__")
    }  # Note using attr.fields does not work here because init=False
    property_map: GraphNodePropertyMap = {}
    for qn_name, value in attr.asdict(node_props).items():
        if value is None:
            continue
        if qn_name in node_qn_mapping:
            property_map[node_qn_mapping[qn_name]] = value
        else:
            raise TypeError(
                f"Missmatch between InteractionProperties and "
                f"NodeQuantumNumbers. NodeQuantumNumbers does not define "
                f"{qn_name}"
            )
    return property_map


def create_particle(
    edge_props: GraphEdgePropertyMap, particles: ParticleCollection
) -> ParticleWithSpin:
    """Create a Particle with spin projection from a qn dictionary.

    The implementation assumes the edge properties match the attributes of a
    particle inside the `.ParticleCollection`.

    Args:
        edge_props: The quantum number dictionary.
        particles: A `.ParticleCollection` which is used to retrieve a
          reference particle instance to lower the memory footprint.

    Raises:
        KeyError: If the edge properties do not contain the pid information or
          no particle with the same pid is found in the `.ParticleCollection`.

        ValueError: If the edge properties do not contain spin projection info.
    """
    particle = particles.find(int(edge_props[EdgeQuantumNumbers.pid]))
    if EdgeQuantumNumbers.spin_projection not in edge_props:
        raise ValueError(
            "GraphEdgePropertyMap does not contain a spin projection!"
        )
    spin_projection = edge_props[EdgeQuantumNumbers.spin_projection]

    return (particle, spin_projection)


def create_interaction_properties(
    qn_solution: GraphNodePropertyMap,
) -> InteractionProperties:
    converted_solution = {k.__name__: v for k, v in qn_solution.items()}
    kw_args = {
        x.name: converted_solution[x.name]
        for x in attr.fields(InteractionProperties)
        if x.name in converted_solution
    }

    return attr.evolve(InteractionProperties(), **kw_args)


def filter_interaction_types(
    valid_determined_interaction_types: List[InteractionTypes],
    allowed_interaction_types: List[InteractionTypes],
) -> List[InteractionTypes]:
    int_type_intersection = list(
        set(allowed_interaction_types)
        & set(valid_determined_interaction_types)
    )
    if int_type_intersection:
        return int_type_intersection
    logging.warning(
        "The specified list of interaction types %s"
        " does not intersect with the valid list of interaction types %s"
        ".\nUsing valid list instead.",
        allowed_interaction_types,
        valid_determined_interaction_types,
    )
    return valid_determined_interaction_types


class InteractionDeterminator(ABC):
    """Interface for interaction determination."""

    @abstractmethod
    def check(
        self,
        in_edge_props: List[ParticleWithSpin],
        out_edge_props: List[ParticleWithSpin],
        node_props: InteractionProperties,
    ) -> List[InteractionTypes]:
        pass


class GammaCheck(InteractionDeterminator):
    """Conservation check for photons."""

    def check(
        self,
        in_edge_props: List[ParticleWithSpin],
        out_edge_props: List[ParticleWithSpin],
        node_props: InteractionProperties,
    ) -> List[InteractionTypes]:
        int_types = list(InteractionTypes)
        for particle, _ in in_edge_props + out_edge_props:
            if "gamma" in particle.name:
                int_types = [InteractionTypes.EM]
                break
        return int_types


class LeptonCheck(InteractionDeterminator):
    """Conservation check lepton numbers."""

    def check(
        self,
        in_edge_props: List[ParticleWithSpin],
        out_edge_props: List[ParticleWithSpin],
        node_props: InteractionProperties,
    ) -> List[InteractionTypes]:
        node_interaction_types = list(InteractionTypes)
        for particle, _ in in_edge_props + out_edge_props:
            if particle.is_lepton():
                if particle.name.startswith("nu("):
                    node_interaction_types = [InteractionTypes.WEAK]
                    break
                node_interaction_types = [
                    InteractionTypes.EM,
                    InteractionTypes.WEAK,
                ]
        return node_interaction_types


def remove_duplicate_solutions(
    solutions: List[StateTransitionGraph[ParticleWithSpin]],
    remove_qns_list: Optional[Set[Type[NodeQuantumNumber]]] = None,
    ignore_qns_list: Optional[Set[Type[NodeQuantumNumber]]] = None,
) -> List[StateTransitionGraph[ParticleWithSpin]]:
    if remove_qns_list is None:
        remove_qns_list = set()
    if ignore_qns_list is None:
        ignore_qns_list = set()
    logging.info("removing duplicate solutions...")
    logging.info(f"removing these qns from graphs: {remove_qns_list}")
    logging.info(f"ignoring qns in graph comparison: {ignore_qns_list}")

    filtered_solutions: List[StateTransitionGraph[ParticleWithSpin]] = list()
    remove_counter = 0
    for sol_graph in solutions:
        sol_graph = _remove_qns_from_graph(sol_graph, remove_qns_list)
        found_graph = _check_equal_ignoring_qns(
            sol_graph, filtered_solutions, ignore_qns_list
        )
        if found_graph is None:
            filtered_solutions.append(sol_graph)
        else:
            # check if found solution also has the prefactors
            # if not overwrite them
            remove_counter += 1

    logging.info(f"removed {remove_counter} solutions")
    return filtered_solutions


def _remove_qns_from_graph(  # pylint: disable=too-many-branches
    graph: StateTransitionGraph[ParticleWithSpin],
    qn_list: Set[Type[NodeQuantumNumber]],
) -> StateTransitionGraph[ParticleWithSpin]:
    new_node_props = {}
    for node_id in graph.topology.nodes:
        node_props = graph.get_node_props(node_id)
        new_node_props[node_id] = attr.evolve(
            node_props, **{x.__name__: None for x in qn_list}
        )

    return graph.evolve(node_props=new_node_props)


def _check_equal_ignoring_qns(
    ref_graph: StateTransitionGraph,
    solutions: List[StateTransitionGraph],
    ignored_qn_list: Set[Type[NodeQuantumNumber]],
) -> Optional[StateTransitionGraph]:
    """Define equal operator for the graphs ignoring certain quantum numbers."""
    if not isinstance(ref_graph, StateTransitionGraph):
        raise TypeError(
            "Reference graph has to be of type StateTransitionGraph"
        )
    found_graph = None
    node_comparator = NodePropertyComparator(ignored_qn_list)
    for graph in solutions:
        if isinstance(graph, StateTransitionGraph):
            if graph.compare(
                ref_graph,
                edge_comparator=lambda e1, e2: e1 == e2,
                node_comparator=node_comparator,
            ):
                found_graph = graph
                break
    return found_graph


class NodePropertyComparator:
    """Functor for comparing node properties in two graphs."""

    def __init__(
        self,
        ignored_qn_list: Optional[Set[Type[NodeQuantumNumber]]] = None,
    ) -> None:
        self.__ignored_qn_list = ignored_qn_list if ignored_qn_list else set()

    def __call__(
        self,
        node_props1: InteractionProperties,
        node_props2: InteractionProperties,
    ) -> bool:
        return attr.evolve(
            node_props1,
            **{x.__name__: None for x in self.__ignored_qn_list},
        ) == attr.evolve(
            node_props2,
            **{x.__name__: None for x in self.__ignored_qn_list},
        )


def filter_graphs(
    graphs: List[StateTransitionGraph], filters: List[Callable]
) -> List[StateTransitionGraph]:
    r"""Implement filtering of a list of `.StateTransitionGraph` 's.

    This function can be used to select a subset of
    `.StateTransitionGraph` 's from a list. Only the graphs passing
    all supplied filters will be returned.

    Note:
        For the more advanced user, lambda functions can be used as filters.

    Args:
        graphs ([`.StateTransitionGraph`]): list of graphs to be
            filtered
        filters (list): list of functions, which take a single
            `.StateTransitionGraph` as an argument
    Returns:
        [`.StateTransitionGraph`]: filtered list of graphs

    Example:
        Selecting only the solutions, in which the :math:`\rho` decays via
        p-wave:

        >>> my_filter = require_interaction_property(
                'rho', InteractionQuantumNumberNames.L,
                create_spin_domain([1], True))
        >>> filtered_solutions = filter_graphs(solutions, [my_filter])
    """
    filtered_graphs = graphs
    for filter_ in filters:
        if not filtered_graphs:
            break
        filtered_graphs = list(filter(filter_, filtered_graphs))
    return filtered_graphs


def require_interaction_property(
    ingoing_particle_name: str,
    interaction_qn: Type[NodeQuantumNumber],
    allowed_values: List,
) -> Callable[[StateTransitionGraph[ParticleWithSpin]], bool]:
    """Filter function.

    Closure, which can be used as a filter function in :func:`.filter_graphs`.

    It selects graphs based on a requirement on the property of specific
    interaction nodes.

    Args:
        ingoing_particle_name (str): name of particle, used to find nodes which
            have a particle with this name as "ingoing"

        interaction_qn ([Type[NodeQuantumNumber]]): interaction quantum number

        allowed_values (list): list of allowed values, that the interaction
            quantum number may take

    Return:
        Callable[Any, bool]:
            - *True* if the graph has nodes with an ingoing particle of the
              given name, and the graph fullfills the quantum number
              requirement
            - *False* otherwise
    """

    def check(graph: StateTransitionGraph[ParticleWithSpin]) -> bool:
        node_ids = _find_node_ids_with_ingoing_particle_name(
            graph, ingoing_particle_name
        )
        if not node_ids:
            return False
        for i in node_ids:
            if (
                getattr(graph.get_node_props(i), interaction_qn.__name__)
                not in allowed_values
            ):
                return False
        return True

    return check


def _find_node_ids_with_ingoing_particle_name(
    graph: StateTransitionGraph[ParticleWithSpin], ingoing_particle_name: str
) -> List[int]:
    topology = graph.topology
    found_node_ids = []
    for node_id in topology.nodes:
        for edge_id in topology.get_edge_ids_ingoing_to_node(node_id):
            edge_props = graph.get_edge_props(edge_id)
            edge_particle_name = edge_props[0].name
            if str(ingoing_particle_name) in str(edge_particle_name):
                found_node_ids.append(node_id)
                break
    return found_node_ids
