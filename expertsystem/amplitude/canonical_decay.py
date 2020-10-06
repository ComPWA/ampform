"""Implementation of the canonical formalism for amplitude model generation."""

from typing import Any, Callable, List, Optional, Union

from expertsystem.data import Spin
from expertsystem.nested_dicts import (
    InteractionQuantumNumberNames,
    StateQuantumNumberNames,
)
from expertsystem.state.properties import (
    get_interaction_property,
    get_particle_property_from_dict,
)
from expertsystem.topology import StateTransitionGraph

from .helicity_decay import (
    HelicityAmplitudeGenerator,
    HelicityAmplitudeNameGenerator,
)
from .model import CanonicalDecay, ClebschGordan, DecayNode, HelicityDecay


def generate_clebsch_gordan_string(
    graph: StateTransitionGraph[dict], node_id: int
) -> str:
    node_props = graph.node_props[node_id]
    ang_orb_mom = __validate_spin_type(
        get_interaction_property(node_props, InteractionQuantumNumberNames.L)
    )
    spin = __validate_spin_type(
        get_interaction_property(node_props, InteractionQuantumNumberNames.S)
    )
    return f"_L_{ang_orb_mom.magnitude}_S_{spin.magnitude}"


class CanonicalAmplitudeNameGenerator(HelicityAmplitudeNameGenerator):
    """Generate names for canonical partial decays.

    That is, using the properties of the decay.
    """

    def generate_unique_amplitude_name(
        self, graph: StateTransitionGraph[dict], node_id: Optional[int] = None
    ) -> str:
        name = ""
        if isinstance(node_id, int):
            node_ids = {node_id}
        else:
            node_ids = graph.nodes
        for node in node_ids:
            name += (
                super().generate_unique_amplitude_name(graph, node)[:-1]
                + generate_clebsch_gordan_string(graph, node)
                + ";"
            )
        return name


def _clebsch_gordan_decorator(
    decay_generate_function: Callable[
        [Any, StateTransitionGraph[dict], int], DecayNode
    ]
) -> Callable[[Any, StateTransitionGraph[dict], int], DecayNode]:
    """Decorate a function with Clebsch-Gordan functionality.

    Decorator method which adds two clebsch gordan coefficients based on the
    translation of helicity amplitudes to canonical ones.
    """

    def wrapper(  # pylint: disable=too-many-locals
        self: Any, graph: StateTransitionGraph[dict], node_id: int
    ) -> DecayNode:
        spin_type = StateQuantumNumberNames.Spin
        amplitude = decay_generate_function(self, graph, node_id)
        if isinstance(amplitude, HelicityDecay):
            helicity_decay = amplitude
        else:
            raise TypeError(
                f"Can only decorate with return value {HelicityDecay.__name__}"
            )

        node_props = graph.node_props[node_id]
        ang_mom = __validate_spin_type(
            get_interaction_property(
                node_props, InteractionQuantumNumberNames.L
            )
        )
        if ang_mom.projection != 0.0:
            raise ValueError(f"Projection of L is non-zero!: {ang_mom}")

        spin = __validate_spin_type(
            get_interaction_property(
                node_props, InteractionQuantumNumberNames.S
            )
        )
        if not isinstance(spin, Spin):
            raise ValueError(
                f"{spin.__class__.__name__} is not of type {Spin.__name__}"
            )

        in_edge_ids = graph.get_edges_ingoing_to_node(node_id)

        parent_spin = __validate_spin_type(
            get_particle_property_from_dict(
                graph.edge_props[in_edge_ids[0]], spin_type
            )
        )

        daughter_spins: List[Spin] = []

        for out_edge_id in graph.get_edges_outgoing_from_node(node_id):
            daughter_spin = get_particle_property_from_dict(
                graph.edge_props[out_edge_id], spin_type
            )
            if daughter_spin is not None and isinstance(daughter_spin, Spin):
                daughter_spins.append(daughter_spin)

        decay_particle_lambda = (
            daughter_spins[0].projection - daughter_spins[1].projection
        )

        cg_ls = ClebschGordan(
            j_1=ang_mom.magnitude,
            m_1=ang_mom.projection,
            j_2=spin.magnitude,
            m_2=decay_particle_lambda,
            J=parent_spin.magnitude,
            M=decay_particle_lambda,
        )
        cg_ss = ClebschGordan(
            j_1=daughter_spins[0].magnitude,
            m_1=daughter_spins[0].projection,
            j_2=daughter_spins[1].magnitude,
            m_2=-daughter_spins[1].projection,
            J=spin.magnitude,
            M=decay_particle_lambda,
        )

        return CanonicalDecay(
            decaying_particle=helicity_decay.decaying_particle,
            decay_products=helicity_decay.decay_products,
            recoil_system=helicity_decay.recoil_system,
            l_s=cg_ls,
            s2s3=cg_ss,
        )

    return wrapper


class CanonicalAmplitudeGenerator(HelicityAmplitudeGenerator):
    r"""Amplitude model generator for the canonical helicity formalism.

    This class defines a full amplitude in the canonical formalism, using the
    helicity formalism as a foundation. The key here is that we take the full
    helicity intensity as a template, and just exchange the helicity amplitudes
    :math:`F` as a sum of canonical amplitudes a:

    .. math::
        F^J_{\lambda_1},\lambda_2 = sum_LS { norm * a^J_LS * CG * CG }.

    Here, :math:`CG` stands for Clebsch-Gordan factor.
    """

    def __init__(self, top_node_no_dynamics: bool = True) -> None:
        super().__init__(top_node_no_dynamics)
        self.name_generator = CanonicalAmplitudeNameGenerator()

    @_clebsch_gordan_decorator
    def _generate_partial_decay(  # type: ignore
        self, graph: StateTransitionGraph[dict], node_id: Optional[int] = None
    ) -> DecayNode:
        return super()._generate_partial_decay(graph, node_id)


def __validate_spin_type(
    interaction_property: Optional[Union[Spin, float]]
) -> Spin:
    if interaction_property is None or not isinstance(
        interaction_property, Spin
    ):
        raise TypeError(
            f"{interaction_property.__class__.__name__} is not of type {Spin.__name__}"
        )
    return interaction_property
