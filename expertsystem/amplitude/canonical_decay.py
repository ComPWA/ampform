"""Implementation of the canonical formalism for amplitude model generation."""

from collections import OrderedDict

from expertsystem.state.particle import (
    InteractionQuantumNumberNames,
    StateQuantumNumberNames,
    get_interaction_property,
    get_particle_property,
)
from expertsystem.topology.graph import (
    get_edges_ingoing_to_node,
    get_edges_outgoing_to_node,
)

from .helicity_decay import (
    HelicityAmplitudeGenerator,
    HelicityAmplitudeNameGenerator,
    generate_particles_string,
)


def generate_clebsch_gordan_string(graph, node_id):
    node_props = graph.node_props[node_id]
    ang_orb_mom = get_interaction_property(
        node_props, InteractionQuantumNumberNames.L
    )
    spin = get_interaction_property(
        node_props, InteractionQuantumNumberNames.S
    )
    return f"_L_{ang_orb_mom.magnitude()}_S_{spin.magnitude()}"


class CanonicalAmplitudeNameGenerator(HelicityAmplitudeNameGenerator):
    """Generate names for canonical partial decays.

    That is, using the properties of the decay.
    """

    def _generate_amplitude_coefficient_name(self, graph, node_id):
        (in_hel_info, out_hel_info) = self._retrieve_helicity_info(
            graph, node_id
        )
        par_name_suffix = (
            generate_particles_string(in_hel_info, False)
            + "_to_"
            + generate_particles_string(out_hel_info, False)
        )
        cg_suffix = generate_clebsch_gordan_string(graph, node_id)
        return par_name_suffix + cg_suffix

    def generate_unique_amplitude_name(self, graph, node_id=None):
        name = ""
        if isinstance(node_id, int):
            nodelist = [node_id]
        else:
            nodelist = graph.nodes
        for node in nodelist:
            name += (
                super().generate_unique_amplitude_name(graph, node)[:-1]
                + generate_clebsch_gordan_string(graph, node)
                + ";"
            )
        return name


def _clebsch_gordan_decorator(decay_generate_function):
    """Decorate a function with Clebsch-Gordan functionality.

    Decorator method which adds two clebsch gordan coefficients based on
    the translation of helicity amplitudes to canonical ones.
    """

    def wrapper(self, graph, node_id):  # pylint: disable=too-many-locals
        spin_type = StateQuantumNumberNames.Spin
        partial_decay_dict = decay_generate_function(self, graph, node_id)
        node_props = graph.node_props[node_id]
        ang_mom = get_interaction_property(
            node_props, InteractionQuantumNumberNames.L
        )
        spin = get_interaction_property(
            node_props, InteractionQuantumNumberNames.S
        )

        in_edge_ids = get_edges_ingoing_to_node(graph, node_id)

        parent_spin = get_particle_property(
            graph.edge_props[in_edge_ids[0]], spin_type
        )

        daughter_spins = []

        for out_edge_id in get_edges_outgoing_to_node(graph, node_id):
            daughter_spins.append(
                get_particle_property(graph.edge_props[out_edge_id], spin_type)
            )

        decay_particle_lambda = (
            daughter_spins[0].projection() - daughter_spins[1].projection()
        )
        cg_ls = OrderedDict()
        cg_ls["Type"] = "LS"
        cg_ls["@j1"] = ang_mom.magnitude()
        if ang_mom.projection() != 0.0:
            raise ValueError(
                "Projection of L is non-zero!: " + str(ang_mom.projection())
            )
        cg_ls["@m1"] = ang_mom.projection()
        cg_ls["@j2"] = spin.magnitude()
        cg_ls["@m2"] = decay_particle_lambda
        cg_ls["J"] = parent_spin.magnitude()
        cg_ls["M"] = decay_particle_lambda
        cg_ss = OrderedDict()
        cg_ss["Type"] = "s2s3"
        cg_ss["@j1"] = daughter_spins[0].magnitude()
        cg_ss["@m1"] = daughter_spins[0].projection()
        cg_ss["@j2"] = daughter_spins[1].magnitude()
        cg_ss["@m2"] = -daughter_spins[1].projection()
        cg_ss["J"] = spin.magnitude()
        cg_ss["M"] = decay_particle_lambda
        cg_dict = {
            "CanonicalSum": {
                "L": int(ang_mom.magnitude()),
                "S": spin.magnitude(),
                "ClebschGordan": [cg_ls, cg_ss],
            }
        }
        partial_decay_dict.update(cg_dict)
        return partial_decay_dict

    return wrapper


class CanonicalAmplitudeGenerator(HelicityAmplitudeGenerator):
    r"""Amplitude model generator for the canonical helicity formalism.

    This class defines a full amplitude in the canonical formalism, using the
    helicity formalism as a foundation. The key here is that we take the full
    helicity intensity as a template, and just exchange the helicity amplitudes
    :math:`F` as a sum of canonical amplitudes a:
    :math:`F^J_{\lambda_1},\lambda_2 = sum_LS { norm * a^J_LS * CG * CG }`.
    Here, :math:`CG` stands for Clebsch-Gordan factor.
    """

    def __init__(
        self,
        top_node_no_dynamics=True,
        name_generator=CanonicalAmplitudeNameGenerator(),
    ):
        super().__init__(top_node_no_dynamics, name_generator=name_generator)

    @_clebsch_gordan_decorator
    def generate_partial_decay(self, graph, node_id):
        return super().generate_partial_decay(graph, node_id)
