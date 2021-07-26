# cspell:ignore einsum
"""Kinematics of an amplitude model in the helicity formalism."""

from typing import Dict, List, Mapping, Set, Tuple

import attr
import numpy as np
from attr.validators import instance_of
from qrules.topology import Topology
from qrules.transition import ReactionInfo, StateTransition

from .data import (
    DataSet,
    EventCollection,
    FourMomentumSequence,
    MatrixSequence,
    ScalarSequence,
)


@attr.s(on_setattr=attr.setters.frozen)
class HelicityAdapter:
    r"""Converter for four-momenta to kinematic variable data.

    The `.transform` method forms the bridge between four-momentum data for the
    decay you are studying and the kinematic variables that are in the
    `.HelicityModel`. These are invariant mass (see
    :func:`.get_invariant_mass_label`) and the :math:`\theta` and :math:`\phi`
    helicity angles (see :func:`.get_helicity_angle_label`).
    """

    reaction_info: ReactionInfo = attr.ib(validator=instance_of(ReactionInfo))
    registered_topologies: Set[Topology] = attr.ib(
        factory=set, init=False, repr=False
    )

    def register_transition(self, transition: StateTransition) -> None:
        if set(self.reaction_info.initial_state) != set(
            transition.initial_states
        ):
            raise ValueError("Transition has mismatching initial state IDs")
        if set(self.reaction_info.final_state) != set(transition.final_states):
            raise ValueError("Transition has mismatching final state IDs")
        for state_id in self.reaction_info.final_state:
            particle = self.reaction_info.initial_state[state_id]
            state = transition.initial_states[state_id]
            if particle != state.particle:
                raise ValueError(
                    f"Transition has different initial particle at {state_id}.",
                    f" Expecting: {particle.name}"
                    f" In added transition: {state.particle.name}",
                )
        self.register_topology(transition.topology)

    def register_topology(self, topology: Topology) -> None:
        _assert_isobar_topology(topology)
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
    topology: Topology, state_id: int
) -> Tuple[str, str]:
    """Generate labels that can be used to identify helicity angles.

    The generated subscripts describe the decay sequence from the right to the
    left, separated by commas. Resonance edge IDs are expressed as a sum of the
    final state IDs that lie below them (see
    :func:`.determine_attached_final_state`). The generated label does not
    state the top-most edge (the initial state).

    Example
    -------
    The following two allowed isobar topologies for a **1-to-5-body** decay
    illustrates how the naming scheme results in a unique label for each of the
    **eight edges** in the decay topology. Note that label only uses final
    state IDs, but still reflects the internal decay topology.

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

    Some labels explained:

    - :code:`phi_1+2`: **edge 6** on the *left* topology, because for this
      topology, we have :math:`p_6=p_1+p_2`.
    - :code:`phi_2+3+4`: **edge 6** *right*, because for this topology,
      :math:`p_6=p_2+p_3+p_4`.
    - :code:`phi_1,1+2`: **edge 1** *left*, because 1 decays from
      :math:`p_6=p_1+p_2`.
    - :code:`phi_1,0+1`: **edge 1** *right*, because it decays from
      :math:`p_5=p_0+p_1`.
    - :code:`phi_4,3+4,2+3+4`: **edge 4** *right*, because it decays from edge
      7 (:math:`p_7=p_3+p_4`), which comes from edge 6
      (:math:`p_7=p_2+p_3+p_4`).

    As noted, the top-most parent (initial state) is not listed in the label.
    """
    _assert_isobar_topology(topology)

    def recursive_label(topology: Topology, state_id: int) -> str:
        edge = topology.edges[state_id]
        if edge.ending_node_id is None:
            label = f"{state_id}"
        else:
            attached_final_state_ids = determine_attached_final_state(
                topology, state_id
            )
            label = "+".join(map(str, attached_final_state_ids))
        if edge.originating_node_id is not None:
            incoming_state_ids = topology.get_edge_ids_ingoing_to_node(
                edge.originating_node_id
            )
            state_id = next(iter(incoming_state_ids))
            if state_id not in topology.incoming_edge_ids:
                label += f",{recursive_label(topology, state_id)}"
        return label

    label = recursive_label(topology, state_id)
    return f"phi_{label}", f"theta_{label}"


def get_invariant_mass_label(topology: Topology, state_id: int) -> str:
    """Generate an invariant mass label for a state (edge on a topology).

    Example
    -------
    In the case shown in Figure :ref:`one-to-five-topology-0`, the invariant
    mass of state :math:`5` is :math:`m_{034}`, because
    :math:`p_5=p_0+p_3+p_4`:

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> get_invariant_mass_label(topologies[0], state_id=5)
    'm_034'

    Naturally, the 'invariant' mass label for a final state is just the mass of the
    state itself:

    >>> get_invariant_mass_label(topologies[0], state_id=1)
    'm_1'
    """
    final_state_ids = determine_attached_final_state(topology, state_id)
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
        child_state_ids = sorted(
            topology.get_edge_ids_outgoing_from_node(node_id)
        )
        if all(
            topology.edges[i].ending_node_id is None for i in child_state_ids
        ):
            state_id = child_state_ids[0]
            four_momentum = events[state_id]
            phi_label, theta_label = get_helicity_angle_label(
                topology, state_id
            )
            helicity_angles[phi_label] = four_momentum.phi()
            helicity_angles[theta_label] = four_momentum.theta()
        for state_id in child_state_ids:
            edge = topology.edges[state_id]
            if edge.ending_node_id is not None:
                # recursively determine all momenta ids in the list
                sub_momenta_ids = determine_attached_final_state(
                    topology, state_id
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
                        topology, state_id
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
    for state_id in topology.edges:
        attached_state_ids = determine_attached_final_state(topology, state_id)
        total_momentum = FourMomentumSequence(
            sum(events[i] for i in attached_state_ids)  # type: ignore
        )
        values = total_momentum.mass()
        name = get_invariant_mass_label(topology, state_id)
        invariant_masses[name] = values
    return DataSet(invariant_masses)


def _assert_isobar_topology(topology: Topology) -> None:
    for node_id in topology.nodes:
        _assert_two_body_decay(topology, node_id)


def _assert_two_body_decay(topology: Topology, node_id: int) -> None:
    parent_state_ids = topology.get_edge_ids_ingoing_to_node(node_id)
    if len(parent_state_ids) != 1:
        raise ValueError(
            f"Node {node_id} has {len(parent_state_ids)} parent states,"
            " so this is not an isobar decay"
        )
    child_state_ids = topology.get_edge_ids_outgoing_from_node(node_id)
    if len(child_state_ids) != 2:
        raise ValueError(
            f"Node {node_id} decays to {len(child_state_ids)} states,"
            " so this is not an isobar decay"
        )


def determine_attached_final_state(
    topology: Topology, state_id: int
) -> List[int]:
    """Determine all final state particles of a transition.

    These are attached downward (forward in time) for a given edge (resembling
    the root).

    Example
    -------
    For **edge 5** in Figure :ref:`one-to-five-topology-0`, we get:

    >>> from qrules.topology import create_isobar_topologies
    >>> topologies = create_isobar_topologies(5)
    >>> determine_attached_final_state(topologies[0], state_id=5)
    [0, 3, 4]
    """
    edge = topology.edges[state_id]
    if edge.ending_node_id is None:
        return [state_id]
    return sorted(
        topology.get_originating_final_state_edge_ids(edge.ending_node_id)
    )
