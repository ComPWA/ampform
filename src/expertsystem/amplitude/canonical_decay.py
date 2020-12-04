"""Implementation of the canonical formalism for amplitude model generation."""

from typing import Any, Callable, List, Optional

from expertsystem.particle import Spin
from expertsystem.reaction.quantum_numbers import (
    InteractionProperties,
    ParticleWithSpin,
)
from expertsystem.reaction.topology import StateTransitionGraph

from .helicity_decay import (
    HelicityAmplitudeGenerator,
    _HelicityAmplitudeNameGenerator,
)
from .model import CanonicalDecay, ClebschGordan, DecayNode, HelicityDecay


def _generate_clebsch_gordan_string(
    graph: StateTransitionGraph[ParticleWithSpin], node_id: int
) -> str:
    node_props = graph.get_node_props(node_id)
    ang_orb_mom = __get_angular_momentum(node_props)
    spin = __get_coupled_spin(node_props)
    return f"_L_{ang_orb_mom.magnitude}_S_{spin.magnitude}"


class _CanonicalAmplitudeNameGenerator(_HelicityAmplitudeNameGenerator):
    """Generate names for canonical partial decays.

    That is, using the properties of the decay.
    """

    def generate_unique_amplitude_name(
        self,
        graph: StateTransitionGraph[ParticleWithSpin],
        node_id: Optional[int] = None,
    ) -> str:
        name = ""
        if isinstance(node_id, int):
            node_ids = frozenset({node_id})
        else:
            node_ids = graph.nodes
        for node in node_ids:
            name += (
                super().generate_unique_amplitude_name(graph, node)[:-1]
                + _generate_clebsch_gordan_string(graph, node)
                + ";"
            )
        return name


def _clebsch_gordan_decorator(
    decay_generate_function: Callable[
        [Any, StateTransitionGraph[ParticleWithSpin], int], DecayNode
    ]
) -> Callable[[Any, StateTransitionGraph[ParticleWithSpin], int], DecayNode]:
    """Decorate a function with Clebsch-Gordan functionality.

    Decorator method which adds two clebsch gordan coefficients based on the
    translation of helicity amplitudes to canonical ones.
    """

    def wrapper(  # pylint: disable=too-many-locals
        self: Any, graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> DecayNode:
        amplitude = decay_generate_function(self, graph, node_id)
        if isinstance(amplitude, HelicityDecay):
            helicity_decay = amplitude
        else:
            raise TypeError(
                f"Can only decorate with return value {HelicityDecay.__name__}"
            )

        node_props = graph.get_node_props(node_id)
        ang_mom = __get_angular_momentum(node_props)
        if ang_mom.projection != 0.0:
            raise ValueError(f"Projection of L is non-zero!: {ang_mom}")

        spin = __get_coupled_spin(node_props)
        if not isinstance(spin, Spin):
            raise ValueError(
                f"{spin.__class__.__name__} is not of type {Spin.__name__}"
            )

        in_edge_ids = graph.get_edge_ids_ingoing_to_node(node_id)

        parent_spin = Spin(
            graph.get_edge_props(in_edge_ids[0])[0].spin,
            graph.get_edge_props(in_edge_ids[0])[1],
        )

        daughter_spins: List[Spin] = []

        for out_edge_id in graph.get_edge_ids_outgoing_from_node(node_id):
            daughter_spin = Spin(
                graph.get_edge_props(out_edge_id)[0].spin,
                graph.get_edge_props(out_edge_id)[1],
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
    :math:`F` as a sum of canonical amplitudes :math:`A`:

    .. math::

        F^J_{\lambda_1,\lambda_2} = \sum_{LS} \mathrm{norm}(A^J_{LS})C^2.

    Here, :math:`C` stands for `Clebsch-Gordan factor
    <https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients>`_.
    """

    def __init__(self, top_node_no_dynamics: bool = True) -> None:
        super().__init__(top_node_no_dynamics)
        self.name_generator = _CanonicalAmplitudeNameGenerator()

    @_clebsch_gordan_decorator
    def _generate_partial_decay(  # type: ignore
        self, graph: StateTransitionGraph[ParticleWithSpin], node_id: int
    ) -> DecayNode:
        return super()._generate_partial_decay(graph, node_id)


def __get_angular_momentum(node_props: InteractionProperties) -> Spin:
    l_mag = node_props.l_magnitude
    l_proj = node_props.l_projection
    if l_mag is None or l_proj is None:
        raise TypeError("Angular momentum L not defined!")
    return Spin(l_mag, l_proj)


def __get_coupled_spin(node_props: InteractionProperties) -> Spin:
    s_mag = node_props.s_magnitude
    s_proj = node_props.s_projection
    if s_mag is None or s_proj is None:
        raise TypeError("Coupled spin S not defined!")
    return Spin(s_mag, s_proj)
