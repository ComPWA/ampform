"""Main interface of the `expertsystem`.

This module contains the functions that you need for the most common use cases
of the `expertsystem`.
"""

__all__ = [
    "StateTransitionManager",
    "load_default_particle_list",
    "InteractionTypes",
]

import logging
from copy import deepcopy
from multiprocessing import Pool
from os import path

from progress.bar import IncrementalBar

from expertsystem.amplitude.canonical_decay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.state import particle
from expertsystem.state.propagation import (
    FullPropagator,
    InteractionTypes,
)
from expertsystem.topology import graph
from expertsystem.topology.topology_builder import (
    SimpleStateTransitionTopologyBuilder,
)

from ._default_settings import (
    DEFAULT_PARTICLE_LIST_FILE,
    DEFAULT_PARTICLE_LIST_PATH,
    create_default_interaction_settings,
)
from ._system_control import (
    GammaCheck,
    LeptonCheck,
    analyse_solution_failure,
    create_interaction_setting_groups,
    filter_interaction_types,
    match_external_edges,
    perform_external_edge_identical_particle_combinatorics,
    remove_duplicate_solutions,
)


class StateTransitionManager:  # pylint: disable=too-many-instance-attributes
    """Main handler for decay topologies."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        initial_state,
        final_state,
        allowed_intermediate_particles=None,
        interaction_type_settings=None,
        formalism_type="helicity",
        topology_building="isobar",
        number_of_threads=4,
        propagation_mode="fast",
    ):
        if allowed_intermediate_particles is None:
            allowed_intermediate_particles = []
        if interaction_type_settings is None:
            interaction_type_settings = {}
        allowed_formalism_types = [
            "helicity",
            "canonical-helicity",
            "canonical",
        ]
        if formalism_type not in allowed_formalism_types:
            raise NotImplementedError(
                f"Formalism type {formalism_type} not implemented."
                f" Use {allowed_formalism_types} instead."
            )
        self.__formalism_type = formalism_type
        self.number_of_threads = number_of_threads
        self.propagation_mode = propagation_mode
        self.initial_state = initial_state
        self.final_state = final_state
        self.interaction_type_settings = interaction_type_settings
        if not self.interaction_type_settings:
            self.interaction_type_settings = create_default_interaction_settings(
                formalism_type
            )
        self.interaction_determinators = [LeptonCheck(), GammaCheck()]
        self.allowed_intermediate_particles = allowed_intermediate_particles
        self.final_state_groupings = []
        self.allowed_interaction_types = [
            InteractionTypes.Strong,
            InteractionTypes.EM,
            InteractionTypes.Weak,
        ]
        self.filter_remove_qns = []
        self.filter_ignore_qns = []
        if formalism_type == "helicity":
            self.filter_remove_qns = [
                particle.InteractionQuantumNumberNames.S,
                particle.InteractionQuantumNumberNames.L,
            ]
        if "helicity" in formalism_type:
            self.filter_ignore_qns = [
                particle.InteractionQuantumNumberNames.ParityPrefactor
            ]
        int_nodes = []
        if topology_building == "isobar":
            if len(initial_state) == 1:
                int_nodes.append(graph.InteractionNode("TwoBodyDecay", 1, 2))
        else:
            int_nodes.append(
                graph.InteractionNode(
                    "NBodyScattering", len(initial_state), len(final_state)
                )
            )
            # turn of mass conservation, in case more than one initial state
            # particle is present
            if len(initial_state) > 1:
                self.interaction_type_settings = create_default_interaction_settings(
                    formalism_type, False
                )
        self.topology_builder = SimpleStateTransitionTopologyBuilder(int_nodes)

        load_default_particle_list()

    @property
    def formalism_type(self) -> str:
        return self.__formalism_type

    def set_topology_builder(self, topology_builder):
        self.topology_builder = topology_builder

    def add_final_state_grouping(self, fs_group):
        if not isinstance(fs_group, list):
            raise ValueError(
                "The final state grouping has to be of type list."
            )
        if len(fs_group) > 0:
            if not isinstance(fs_group[0], list):
                fs_group = [fs_group]
            self.final_state_groupings.append(fs_group)

    def set_allowed_interaction_types(self, allowed_interaction_types):
        # verify order
        for allowed_types in allowed_interaction_types:
            if not isinstance(allowed_types, InteractionTypes):
                raise TypeError(
                    "allowed interaction types must be of type"
                    "[InteractionTypes]"
                )
            if allowed_types not in self.interaction_type_settings:
                logging.info(self.interaction_type_settings.keys())
                raise ValueError(
                    f"interaction {allowed_types} not found in settings"
                )
        self.allowed_interaction_types = allowed_interaction_types

    def prepare_graphs(self):
        topology_graphs = self._build_topologies()
        init_graphs = self._create_seed_graphs(topology_graphs)
        graph_node_setting_pairs = self._determine_node_settings(init_graphs)
        # create groups of settings ordered by "probability"
        graph_settings_groups = create_interaction_setting_groups(
            graph_node_setting_pairs
        )
        return graph_settings_groups

    def _build_topologies(self):
        all_graphs = self.topology_builder.build_graphs(
            len(self.initial_state), len(self.final_state)
        )
        logging.info(f"number of topology graphs: {len(all_graphs)}")
        return all_graphs

    def _create_seed_graphs(self, topology_graphs):
        # initialize the graph edges (initial and final state)
        init_graphs = []
        for topology_graph in topology_graphs:
            topology_graph.set_graph_element_properties_comparator(
                particle.CompareGraphElementPropertiesFunctor()
            )
            init_graphs.extend(
                particle.initialize_graph(
                    topology_graph,
                    self.initial_state,
                    self.final_state,
                    self.final_state_groupings,
                )
            )

        logging.info(f"initialized {len(init_graphs)} graphs!")
        return init_graphs

    def _determine_node_settings(self, graphs):
        # pylint: disable=too-many-locals
        graph_node_setting_pairs = []
        for instance in graphs:
            final_state_edges = graph.get_final_state_edges(instance)
            initial_state_edges = graph.get_initial_state_edges(instance)
            node_settings = {}
            for node_id in instance.nodes:
                node_int_types = []
                out_edge_ids = graph.get_edges_outgoing_to_node(
                    instance, node_id
                )
                in_edge_ids = graph.get_edges_outgoing_to_node(
                    instance, node_id
                )
                in_edge_props = [
                    instance.edge_props[edge_id]
                    for edge_id in [
                        x for x in in_edge_ids if x in initial_state_edges
                    ]
                ]
                out_edge_props = [
                    instance.edge_props[edge_id]
                    for edge_id in [
                        x for x in out_edge_ids if x in final_state_edges
                    ]
                ]
                node_props = {}
                if node_id in instance.node_props:
                    node_props = instance.node_props[node_id]
                for int_det in self.interaction_determinators:
                    determined_interactions = int_det.check(
                        in_edge_props, out_edge_props, node_props
                    )
                    if node_int_types:
                        node_int_types = list(
                            set(determined_interactions) & set(node_int_types)
                        )
                    else:
                        node_int_types = determined_interactions
                node_int_types = filter_interaction_types(
                    node_int_types, self.allowed_interaction_types
                )
                logging.debug(
                    "using %s interaction order for node: %s",
                    str(node_int_types),
                    str(node_id),
                )
                node_settings[node_id] = [
                    deepcopy(self.interaction_type_settings[x])
                    for x in node_int_types
                ]
            graph_node_setting_pairs.append((instance, node_settings))
        return graph_node_setting_pairs

    def find_solutions(
        self, graph_setting_groups
    ):  # pylint: disable=too-many-locals
        """Check for solutions for a specific set of interaction settings."""
        results = {}
        logging.info(
            "Number of interaction settings groups being processed: %d",
            len(graph_setting_groups),
        )
        for strength, graph_setting_group in sorted(
            graph_setting_groups.items(), reverse=True
        ):
            logging.info(
                "processing interaction settings group with "
                f"strength {strength}",
            )
            logging.info(f"{graph_setting_group} entries in this group")
            logging.info(f"running with {self.number_of_threads} threads...")

            temp_results = []
            progress_bar = IncrementalBar(
                "Propagating quantum numbers...", max=len(graph_setting_group)
            )
            progress_bar.update()
            if self.number_of_threads > 1:
                with Pool(self.number_of_threads) as pool:
                    for result in pool.imap_unordered(
                        self._propagate_quantum_numbers, graph_setting_group, 1
                    ):
                        temp_results.append(result)
                        progress_bar.next()
            else:
                for graph_setting_pair in graph_setting_group:
                    temp_results.append(
                        self._propagate_quantum_numbers(graph_setting_pair)
                    )
                    progress_bar.next()
            progress_bar.finish()
            logging.info("Finished!")
            if strength not in results:
                results[strength] = []
            results[strength].extend(temp_results)

        for key, value in results.items():
            logging.info(
                f"number of solutions for strength ({key}) "
                f"after qn propagation: {sum([len(x[0]) for x in value])}",
            )

        # remove duplicate solutions, which only differ in the interaction qn S
        results = remove_duplicate_solutions(
            results, self.filter_remove_qns, self.filter_ignore_qns
        )

        node_non_satisfied_rules = []
        solutions = []
        for result in results.values():
            for (tempsolutions, non_satisfied_laws) in result:
                solutions.extend(tempsolutions)
                node_non_satisfied_rules.append(non_satisfied_laws)
        logging.info(f"total number of found solutions: {len(solutions)}")
        violated_laws = []
        if len(solutions) == 0:
            violated_laws = analyse_solution_failure(node_non_satisfied_rules)
            logging.info(f"violated rules: {violated_laws}")

        # finally perform combinatorics of identical external edges
        # (initial or final state edges) and prepare graphs for
        # amplitude generation
        match_external_edges(solutions)
        final_solutions = []
        for sol in solutions:
            final_solutions.extend(
                perform_external_edge_identical_particle_combinatorics(sol)
            )

        return (final_solutions, violated_laws)

    def _propagate_quantum_numbers(self, state_graph_node_settings_pair):
        propagator = self._initialize_qn_propagator(
            state_graph_node_settings_pair[0],
            state_graph_node_settings_pair[1],
        )
        solutions = propagator.find_solutions()
        return (solutions, propagator.get_non_satisfied_conservation_laws())

    def _initialize_qn_propagator(self, state_graph, node_settings):
        propagator = FullPropagator(state_graph, self.propagation_mode)
        for node_id, interaction_settings in node_settings.items():
            propagator.assign_settings_to_node(node_id, interaction_settings)
        # specify set of particles which are allowed to be intermediate
        # particles. If list is empty, then all particles in the default
        # particle list are used
        propagator.set_allowed_intermediate_particles(
            self.allowed_intermediate_particles
        )

        return propagator

    def write_amplitude_model(self, solutions: list, output_file: str) -> None:
        """Generate an amplitude model from the solutions.

        The type of amplitude model (`.HelicityAmplitudeGenerator` or
        `.CanonicalAmplitudeGenerator`) is determined from the
        :code:`formalism_type` that was chosen when constructing the
        `.StateTransitionManager`.
        """
        if self.formalism_type == "helicity":
            amplitude_generator = HelicityAmplitudeGenerator()
        elif self.formalism_type in ["canonical-helicity", "canonical"]:
            amplitude_generator = CanonicalAmplitudeGenerator()
        amplitude_generator.generate(solutions)
        amplitude_generator.write_to_file(output_file)


def load_default_particle_list() -> None:
    """Load the default particle list that comes with the expertsystem.

    .. warning::
        This resets all particle definitions and the removes particles that
        don't exist in the particle list that ships with the `expertsystem`!
    """
    if not path.exists(DEFAULT_PARTICLE_LIST_PATH):
        raise FileNotFoundError(
            f"\n  Failed to load {DEFAULT_PARTICLE_LIST_FILE}!"
            "\n  Please contact the developers: https://github.com/ComPWA"
        )
    particle.DATABASE = dict()
    particle.load_particles(DEFAULT_PARTICLE_LIST_PATH)
    logging.info(
        f"Loaded {len(particle.DATABASE)} particles from {DEFAULT_PARTICLE_LIST_FILE}!"
    )
