# cspell:ignore einsum
"""Kinematics of an amplitude model in the helicity formalism."""

import textwrap
from typing import Dict, Mapping, Set, Tuple

import attr
import numpy as np
from attr.validators import instance_of
from qrules.io import asdot
from qrules.particle import Particle, ParticleWithSpin
from qrules.topology import (
    FrozenDict,
    StateTransitionGraph,
    Topology,
    create_isobar_topologies,
)

from ._graph_info import assert_isobar_topology, determine_attached_final_state
from .data import (
    DataSet,
    EventCollection,
    FourMomentumSequence,
    MatrixSequence,
    ScalarSequence,
)


@attr.s(frozen=True)
class ReactionInfo:
    initial_state: FrozenDict[int, Particle] = attr.ib(converter=FrozenDict)
    final_state: FrozenDict[int, Particle] = attr.ib(converter=FrozenDict)

    def __attrs_post_init__(self) -> None:
        initial_state_ids = set(self.initial_state)
        final_state_ids = set(self.final_state)
        if initial_state_ids & final_state_ids:
            raise ValueError(
                f"Initial state IDs {initial_state_ids} overlap"
                f" with final state IDs {final_state_ids}"
            )
        particles = {
            *self.initial_state.values(),
            *self.final_state.values(),
        }
        if not all(map(lambda p: isinstance(p, Particle), particles)):
            raise ValueError(
                f"Not all items in state ID mappings are {Particle.__name__}"
            )

    @staticmethod
    def from_graph(
        graph: StateTransitionGraph[ParticleWithSpin],
    ) -> "ReactionInfo":
        return ReactionInfo(
            initial_state={
                i: graph.get_edge_props(i)[0]
                for i in graph.topology.incoming_edge_ids
            },
            final_state={
                i: graph.get_edge_props(i)[0]
                for i in graph.topology.outgoing_edge_ids
            },
        )


@attr.s(on_setattr=attr.setters.frozen)
class HelicityAdapter:
    r"""Converter for four-momenta to kinematic variable data.

    The `.transform` method forms the bridge between four-momentum data for the
    decay you are studying and the kinematic variables that are in the
    `.HelicityModel`. These are invariant mass and the :math:`\theta` and
    :math:`\phi` helicity angles.
    """

    reaction_info: ReactionInfo = attr.ib(validator=instance_of(ReactionInfo))
    registered_topologies: Set[Topology] = attr.ib(
        factory=set, init=False, repr=False
    )

    def register_transition(
        self, transition: StateTransitionGraph[ParticleWithSpin]
    ) -> None:
        reaction_info = ReactionInfo.from_graph(transition)
        if reaction_info != self.reaction_info:
            raise ValueError(
                "Transition has different initial and final states",
                reaction_info,
                self.reaction_info,
            )
        self.register_topology(transition.topology)

    def register_topology(self, topology: Topology) -> None:
        assert_isobar_topology(topology)
        if len(self.registered_topologies) == 0:
            object.__setattr__(
                self,
                "final_state_ids",
                tuple(sorted(topology.outgoing_edge_ids)),
            )
        if len(topology.incoming_edge_ids) != 1:
            raise ValueError(
                f"Topology has {len(topology.incoming_edge_ids)} incoming"
                " edges, so is not isobar"
            )
        if len(self.registered_topologies) != 0:
            existing_topology = next(iter(self.registered_topologies))
            if (
                (
                    topology.incoming_edge_ids
                    != existing_topology.incoming_edge_ids
                )
                or (
                    topology.outgoing_edge_ids
                    != existing_topology.outgoing_edge_ids
                )
                or (
                    topology.outgoing_edge_ids
                    != existing_topology.outgoing_edge_ids
                )
                or (topology.nodes != existing_topology.nodes)
            ):
                raise ValueError("Edge or node IDs of topology do not match")
        self.registered_topologies.add(topology)

    def transform(self, events: EventCollection) -> DataSet:
        output: Dict[str, ScalarSequence] = {}
        for topology in self.registered_topologies:
            output.update(_compute_helicity_angles(events, topology))
            output.update(_compute_invariant_masses(events, topology))
        return DataSet(output)


def get_helicity_angle_label(
    topology: Topology, edge_id: int
) -> Tuple[str, str]:
    """Generate labels that can be used to identify helicity angles.

    >>> from ampform.kinematics import get_helicity_angle_label
    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> topology = topologies[0]
    >>> for i in topology.intermediate_edge_ids | topology.outgoing_edge_ids:
    ...     phi_label, theta_label = get_helicity_angle_label(topology, i)
    ...     print(f"{i}: '{phi_label}'")
    0: 'phi_0,0+3+4'
    1: 'phi_1,1+2'
    2: 'phi_2,1+2'
    3: 'phi_3,3+4,0+3+4'
    4: 'phi_4,3+4,0+3+4'
    5: 'phi_0+3+4'
    6: 'phi_1+2'
    7: 'phi_3+4,0+3+4'
    >>> topology = topologies[1]
    >>> for i in topology.intermediate_edge_ids | topology.outgoing_edge_ids:
    ...     phi_label, theta_label = get_helicity_angle_label(topology, i)
    ...     print(f"{i}: '{phi_label}'")
    0: 'phi_0,0+1'
    1: 'phi_1,0+1'
    2: 'phi_2,2+3+4'
    3: 'phi_3,3+4,2+3+4'
    4: 'phi_4,3+4,2+3+4'
    5: 'phi_0+1'
    6: 'phi_2+3+4'
    7: 'phi_3+4,2+3+4'
    """
    assert_isobar_topology(topology)

    def recursive_label(topology: Topology, edge_id: int) -> str:
        edge = topology.edges[edge_id]
        if edge.ending_node_id is None:
            label = f"{edge_id}"
        else:
            attached_final_state_ids = determine_attached_final_state(
                topology, edge_id
            )
            label = "+".join(map(str, attached_final_state_ids))
        if edge.originating_node_id is not None:
            in_edges = topology.get_edge_ids_ingoing_to_node(
                edge.originating_node_id
            )
            in_edge_id = next(iter(in_edges))
            if in_edge_id not in topology.incoming_edge_ids:
                label += f",{recursive_label(topology, in_edge_id)}"
        return label

    label = recursive_label(topology, edge_id)
    return f"phi_{label}", f"theta_{label}"


assert get_helicity_angle_label.__doc__ is not None
get_helicity_angle_label.__doc__ += f"""

.. panels::
  :body: text-center

  .. graphviz::

    {textwrap.indent(asdot(create_isobar_topologies(5)[0]), '    ')}

  :code:`topologies[0]`

  ---

  .. graphviz::

    {textwrap.indent(asdot(create_isobar_topologies(5)[1]), '    ')}

  :code:`topologies[1]`
"""


def get_invariant_mass_label(topology: Topology, edge_id: int) -> str:
    final_state_ids = determine_attached_final_state(topology, edge_id)
    return f"m_{''.join(map(str, sorted(final_state_ids)))}"


def _compute_helicity_angles(  # pylint: disable=too-many-locals
    events: EventCollection, topology: Topology
) -> DataSet:
    if topology.outgoing_edge_ids != set(events):
        raise ValueError(
            f"Momentum IDs {set(events)} do not match "
            f"final state edge IDs {set(topology.outgoing_edge_ids)}"
        )

    def __recursive_helicity_angles(  # pylint: disable=too-many-locals
        events: EventCollection, node_id: int
    ) -> DataSet:
        helicity_angles: Dict[str, ScalarSequence] = {}
        child_edge_ids = sorted(
            topology.get_edge_ids_outgoing_from_node(node_id)
        )
        if all(
            topology.edges[i].ending_node_id is None for i in child_edge_ids
        ):
            edge_id = child_edge_ids[0]
            four_momentum = events[edge_id]
            phi_label, theta_label = get_helicity_angle_label(
                topology, edge_id
            )
            helicity_angles[phi_label] = four_momentum.phi()
            helicity_angles[theta_label] = four_momentum.theta()
        for edge_id in child_edge_ids:
            edge = topology.edges[edge_id]
            if edge.ending_node_id is not None:
                # recursively determine all momenta ids in the list
                sub_momenta_ids = determine_attached_final_state(
                    topology, edge_id
                )
                if len(sub_momenta_ids) > 1:
                    # add all of these momenta together -> defines new subsystem
                    four_momentum = events.sum(sub_momenta_ids)

                    # boost all of those momenta into this new subsystem
                    phi = four_momentum.phi()
                    theta = four_momentum.theta()
                    p3_norm = four_momentum.p_norm()
                    beta = ScalarSequence(p3_norm / four_momentum.energy)
                    new_momentum_pool = EventCollection(
                        {
                            k: _get_boost_z_matrix(beta).dot(
                                _get_rotation_matrix_y(-theta).dot(
                                    _get_rotation_matrix_z(-phi).dot(v)
                                )
                            )
                            for k, v in events.items()
                            if k in sub_momenta_ids
                        }
                    )

                    # register current angle variables
                    phi_label, theta_label = get_helicity_angle_label(
                        topology, edge_id
                    )
                    helicity_angles[phi_label] = four_momentum.phi()
                    helicity_angles[theta_label] = four_momentum.theta()

                    # call next recursion
                    angles = __recursive_helicity_angles(
                        new_momentum_pool,
                        edge.ending_node_id,
                    )
                    helicity_angles.update(angles)

        return DataSet(helicity_angles)

    initial_state_id = next(iter(topology.incoming_edge_ids))
    initial_state_edge = topology.edges[initial_state_id]
    assert initial_state_edge.ending_node_id is not None
    return __recursive_helicity_angles(
        events, initial_state_edge.ending_node_id
    )


def _get_boost_z_matrix(beta: ScalarSequence) -> MatrixSequence:
    n_events = len(beta)
    gamma = 1 / np.sqrt(1 - beta ** 2)
    zeros = np.zeros(n_events)
    ones = np.ones(n_events)
    return MatrixSequence(
        np.array(
            [
                [gamma, zeros, zeros, -gamma * beta],
                [zeros, ones, zeros, zeros],
                [zeros, zeros, ones, zeros],
                [-gamma * beta, zeros, zeros, gamma],
            ]
        ).transpose(2, 0, 1)
    )


def _get_rotation_matrix_z(angle: ScalarSequence) -> MatrixSequence:
    n_events = len(angle)
    zeros = np.zeros(n_events)
    ones = np.ones(n_events)
    return MatrixSequence(
        np.array(
            [
                [ones, zeros, zeros, zeros],
                [zeros, np.cos(angle), -np.sin(angle), zeros],
                [zeros, np.sin(angle), np.cos(angle), zeros],
                [zeros, zeros, zeros, ones],
            ]
        ).transpose(2, 0, 1)
    )


def _get_rotation_matrix_y(angle: ScalarSequence) -> MatrixSequence:
    n_events = len(angle)
    zeros = np.zeros(n_events)
    ones = np.ones(n_events)
    return MatrixSequence(
        np.array(
            [
                [ones, zeros, zeros, zeros],
                [zeros, np.cos(angle), zeros, np.sin(angle)],
                [zeros, zeros, ones, zeros],
                [zeros, -np.sin(angle), zeros, np.cos(angle)],
            ]
        ).transpose(2, 0, 1)
    )


def _compute_invariant_masses(
    events: Mapping[int, FourMomentumSequence], topology: Topology
) -> DataSet:
    """Compute the invariant masses for all final state combinations."""
    if topology.outgoing_edge_ids != set(events):
        raise ValueError(
            f"Momentum IDs {set(events)} do not match "
            f"final state edge IDs {set(topology.outgoing_edge_ids)}"
        )
    invariant_masses = {}
    for edge_id in topology.edges:
        attached_edge_ids = determine_attached_final_state(topology, edge_id)
        total_momentum = FourMomentumSequence(
            sum(events[i] for i in attached_edge_ids)  # type: ignore
        )
        values = total_momentum.mass()
        name = get_invariant_mass_label(topology, edge_id)
        invariant_masses[name] = values
    return DataSet(invariant_masses)
