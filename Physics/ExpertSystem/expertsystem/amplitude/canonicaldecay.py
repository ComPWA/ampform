from collections import OrderedDict

import xmltodict

from expertsystem.topology.graph import (get_initial_state_edges,
                                         get_edges_ingoing_to_node,
                                         get_edges_outgoing_to_node)

from expertsystem.state.particle import (
    StateQuantumNumberNames, InteractionQuantumNumberNames,
    XMLLabelConstants, get_xml_label,
    get_particle_property, get_interaction_property)

from expertsystem.amplitude.helicitydecay import (
    HelicityDecayAmplitudeGeneratorXML,
    HelicityPartialDecayNameGenerator
)


class CanonicalPartialDecayNameGenerator(HelicityPartialDecayNameGenerator):
    '''
    asdf
    '''
    def _canonical_ls_decorator(generate_function):
        '''
        Decorator method which adds two clebsch gordan coefficients based on
        the translation of helicity amplitudes to canonical ones.
        '''

        def wrapper(self, graph, node_id):
            name_pair = generate_function(self, graph, node_id)
            node_props = graph.node_props[node_id]
            L = get_interaction_property(node_props,
                                         InteractionQuantumNumberNames.L)
            S = get_interaction_property(node_props,
                                         InteractionQuantumNumberNames.S)
            addition_string = '_L_' + str(L.magnitude()) + \
                '_S_' + str(S.magnitude())
            return (name_pair[0] + addition_string,
                    name_pair[1] + addition_string)

        return wrapper

    @_canonical_ls_decorator
    def generate(self, graph, node_id):
        return super().generate(graph, node_id)


class CanonicalDecayAmplitudeGeneratorXML(HelicityDecayAmplitudeGeneratorXML):
    '''
    This class defines a full amplitude in the canonical formalism, using the
    heliclty formalism as a foundation.
    The key here is that we take the full helicity intensity as a template, and
    just exchange the helicity amplitudes F as a sum of canonical amplitudes a:
    F^J_lambda1,lambda2 = sum_LS { norm * a^J_LS * CG * CG }
    Here CG stands for Clebsch-Gordan factor.
    '''

    def __init__(self, top_node_no_dynamics=True,
                 use_parity_conservation=None):
        super().__init__(top_node_no_dynamics, use_parity_conservation)
        self.name_generator = CanonicalPartialDecayNameGenerator(
            self.use_parity_conservation)

    def _clebsch_gordan_decorator(decay_generate_function):
        '''
        Decorator method which adds two clebsch gordan coefficients based on
        the translation of helicity amplitudes to canonical ones.
        '''

        def wrapper(self, graph, node_id, parameter_props):
            spinqn = StateQuantumNumberNames.Spin
            partial_decay_dict = decay_generate_function(
                self, graph, node_id, parameter_props)
            node_props = graph.node_props[node_id]
            L = get_interaction_property(node_props,
                                         InteractionQuantumNumberNames.L)
            S = get_interaction_property(node_props,
                                         InteractionQuantumNumberNames.S)

            in_edge_ids = get_edges_ingoing_to_node(graph, node_id)

            parent_spin = get_particle_property(
                graph.edge_props[in_edge_ids[0]], spinqn)

            daughter_spins = []

            for out_edge_id in get_edges_outgoing_to_node(graph, node_id):
                daughter_spins.append(get_particle_property(
                    graph.edge_props[out_edge_id], spinqn)
                )

            decay_particle_lambda = (daughter_spins[0].projection() -
                                     daughter_spins[1].projection())
            cg_ls = OrderedDict()
            cg_ls['@Type'] = "LS"
            cg_ls['@j1'] = L.magnitude()
            if L.projection() != 0.0:
                raise ValueError("Projection of L is non-zero!: "
                                 + str(L.projection()))
            cg_ls['@m1'] = L.projection()
            cg_ls['@j2'] = S.magnitude()
            cg_ls['@m2'] = decay_particle_lambda
            cg_ls['@J'] = parent_spin.magnitude()
            cg_ls['@M'] = decay_particle_lambda
            cg_ss = OrderedDict()
            cg_ss['@Type'] = "s2s3"
            cg_ss['@j1'] = daughter_spins[0].magnitude()
            cg_ss['@m1'] = daughter_spins[0].projection()
            cg_ss['@j2'] = daughter_spins[1].magnitude()
            cg_ss['@m2'] = -daughter_spins[1].projection()
            cg_ss['@J'] = S.magnitude()
            cg_ss['@M'] = decay_particle_lambda
            cg_dict = {
                'CanonicalSum': {
                    '@L': L.magnitude(),
                    '@S': S.magnitude(),
                    'ClebschGordan': [cg_ls, cg_ss]
                }
            }
            partial_decay_dict.update(cg_dict)
            return partial_decay_dict

        return wrapper

    @_clebsch_gordan_decorator
    def generate_partial_decay(self, graph, node_id, parameter_props):
        return super().generate_partial_decay(graph, node_id, parameter_props)
