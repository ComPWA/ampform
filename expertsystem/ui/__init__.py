"""Main interface of the `expertsystem`.

This module contains the functions that you need for the most common use cases
of the `expertsystem`.
"""

import logging
from copy import deepcopy
from multiprocessing import Pool
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from progress.bar import IncrementalBar

from expertsystem import io
from expertsystem.amplitude.canonical_decay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.data import ParticleCollection
from expertsystem.state.particle import (
    CompareGraphElementPropertiesFunctor,
    InteractionQuantumNumberNames,
    ParticleWithSpin,
    StateDefinition,
    filter_particles,
    initialize_graph,
    match_external_edges,
    particle_with_spin_projection_to_dict,
)
from expertsystem.state.propagation import (
    FullPropagator,
    InteractionNodeSettings,
    InteractionTypes,
)
from expertsystem.topology import (
    InteractionNode,
    StateTransitionGraph,
    Topology,
)
from expertsystem.topology import SimpleStateTransitionTopologyBuilder

from ._default_settings import (
    DEFAULT_PARTICLE_LIST_PATH,
    create_default_interaction_settings,
)
from ._system_control import (
    GammaCheck,
    GraphSettingsGroups,
    LeptonCheck,
    NodeSettings,
    SolutionMapping,
    ViolatedLaws,
    analyse_solution_failure,
    create_interaction_setting_groups,
    filter_interaction_types,
    remove_duplicate_solutions,
)


class StateTransitionManager:  # pylint: disable=too-many-instance-attributes
    """Main handler for decay topologies."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        initial_state: List[StateDefinition],
        final_state: List[StateDefinition],
        particles: ParticleCollection = ParticleCollection(),
        allowed_intermediate_particles: Optional[List[str]] = None,
        interaction_type_settings: Dict[
            InteractionTypes, InteractionNodeSettings
        ] = None,
        formalism_type: str = "helicity",
        topology_building: str = "isobar",
        number_of_threads: int = 4,
        propagation_mode: str = "fast",
        reload_pdg: bool = False,
    ) -> None:
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
        self.__formalism_type = str(formalism_type)
        self.__particles = particles
        self.number_of_threads = int(number_of_threads)
        self.propagation_mode = str(propagation_mode)
        self.initial_state = initial_state
        self.final_state = final_state
        self.interaction_type_settings = interaction_type_settings

        self.interaction_determinators = [LeptonCheck(), GammaCheck()]
        self.final_state_groupings: Optional[List[List[List[str]]]] = None
        self.allowed_interaction_types: List[InteractionTypes] = [
            InteractionTypes.Strong,
            InteractionTypes.EM,
            InteractionTypes.Weak,
        ]
        self.filter_remove_qns: List[InteractionQuantumNumberNames] = []
        self.filter_ignore_qns: List[InteractionQuantumNumberNames] = []
        if formalism_type == "helicity":
            self.filter_remove_qns = [
                InteractionQuantumNumberNames.S,
                InteractionQuantumNumberNames.L,
            ]
        if "helicity" in formalism_type:
            self.filter_ignore_qns = [
                InteractionQuantumNumberNames.ParityPrefactor
            ]
        int_nodes = []
        use_mass_conservation = True
        use_nbody_topology = False
        if topology_building == "isobar":
            if len(initial_state) == 1:
                int_nodes.append(InteractionNode("TwoBodyDecay", 1, 2))
        else:
            int_nodes.append(
                InteractionNode(
                    "NBodyScattering", len(initial_state), len(final_state)
                )
            )
            use_nbody_topology = True
            # turn of mass conservation, in case more than one initial state
            # particle is present
            if len(initial_state) > 1:
                use_mass_conservation = False

        if not self.interaction_type_settings:
            self.interaction_type_settings = (
                create_default_interaction_settings(
                    formalism_type,
                    nbody_topology=use_nbody_topology,
                    use_mass_conservation=use_mass_conservation,
                )
            )
        self.topology_builder = SimpleStateTransitionTopologyBuilder(int_nodes)

        if reload_pdg or len(self.__particles) == 0:
            self.__particles = load_default_particles()

        if allowed_intermediate_particles is None:
            allowed_intermediate_particles = []
        self.__allowed_intermediate_particles = filter_particles(
            self.__particles, allowed_intermediate_particles
        )

    @property
    def formalism_type(self) -> str:
        return self.__formalism_type

    def set_topology_builder(
        self, topology_builder: SimpleStateTransitionTopologyBuilder
    ) -> None:
        self.topology_builder = topology_builder

    def add_final_state_grouping(
        self, fs_group: List[Union[str, List[str]]]
    ) -> None:
        if not isinstance(fs_group, list):
            raise ValueError(
                "The final state grouping has to be of type list."
            )
        if len(fs_group) > 0:
            if self.final_state_groupings is None:
                self.final_state_groupings = list()
            if not isinstance(fs_group[0], list):
                fs_group = [fs_group]  # type: ignore
            self.final_state_groupings.append(fs_group)  # type: ignore

    def set_allowed_interaction_types(
        self, allowed_interaction_types: List[InteractionTypes]
    ) -> None:
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

    def prepare_graphs(self) -> GraphSettingsGroups:
        topology_graphs = self._build_topologies()
        init_graphs = self._create_seed_graphs(topology_graphs)
        init_graphs = self._convert_edges_to_dict(init_graphs)  # type: ignore
        graph_node_setting_pairs = self._determine_node_settings(init_graphs)
        # create groups of settings ordered by "probability"
        graph_settings_groups = create_interaction_setting_groups(
            graph_node_setting_pairs
        )
        return graph_settings_groups

    def _build_topologies(self) -> List[Topology]:
        all_graphs = self.topology_builder.build_graphs(
            len(self.initial_state), len(self.final_state)
        )
        logging.info(f"number of topology graphs: {len(all_graphs)}")
        return all_graphs

    def _create_seed_graphs(
        self, topology_graphs: List[Topology]
    ) -> List[StateTransitionGraph[ParticleWithSpin]]:
        # initialize the graph edges (initial and final state)
        init_graphs: List[StateTransitionGraph[ParticleWithSpin]] = []
        for topology_graph in topology_graphs:
            initialized_graphs = initialize_graph(
                topology=topology_graph,
                particles=self.__particles,
                initial_state=self.initial_state,
                final_state=self.final_state,
                final_state_groupings=self.final_state_groupings,
            )
            init_graphs.extend(initialized_graphs)
        init_graphs = self._convert_edges_to_dict(init_graphs)  # type: ignore
        for graph in init_graphs:
            graph.graph_element_properties_comparator = (
                CompareGraphElementPropertiesFunctor()
            )

        logging.info(f"initialized {len(init_graphs)} graphs!")
        return init_graphs

    @staticmethod
    def _convert_edges_to_dict(
        graphs: List[StateTransitionGraph[ParticleWithSpin]],
    ) -> List[StateTransitionGraph[dict]]:
        for graph in graphs:
            for edge_id, edge_property in graph.edge_props.items():
                if not isinstance(edge_property, dict):
                    graph.edge_props[
                        edge_id
                    ] = particle_with_spin_projection_to_dict(  # type: ignore
                        edge_property
                    )
        return graphs  # type: ignore

    def _determine_node_settings(
        self, graphs: List[StateTransitionGraph]
    ) -> List[Tuple[StateTransitionGraph, NodeSettings]]:
        # pylint: disable=too-many-locals
        graph_node_setting_pairs = []
        for instance in graphs:
            final_state_edges = instance.get_final_state_edges()
            initial_state_edges = instance.get_initial_state_edges()
            node_settings: NodeSettings = {}
            for node_id in instance.nodes:
                node_int_types: List[InteractionTypes] = []
                out_edge_ids = instance.get_edges_outgoing_from_node(node_id)
                in_edge_ids = instance.get_edges_outgoing_from_node(node_id)
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
        self,
        graph_setting_groups: GraphSettingsGroups,
    ) -> Tuple[
        List[StateTransitionGraph], List[str]
    ]:  # pylint: disable=too-many-locals
        """Check for solutions for a specific set of interaction settings."""
        results: SolutionMapping = {}
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

            temp_results: List[
                Tuple[List[StateTransitionGraph], ViolatedLaws]
            ] = []
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

        node_non_satisfied_rules: List[ViolatedLaws] = []
        solutions: List[StateTransitionGraph] = []
        for item in results.values():
            for (temp_solutions, non_satisfied_laws) in item:
                solutions.extend(temp_solutions)
                node_non_satisfied_rules.append(non_satisfied_laws)
        logging.info(f"total number of found solutions: {len(solutions)}")
        violated_laws = []
        if len(solutions) == 0:
            violated_laws = analyse_solution_failure(node_non_satisfied_rules)
            logging.info(f"violated rules: {violated_laws}")

        match_external_edges(solutions)
        return (solutions, violated_laws)

    def _propagate_quantum_numbers(
        self,
        state_graph_node_settings_pair: Tuple[
            StateTransitionGraph, Dict[int, InteractionNodeSettings]
        ],
    ) -> Tuple[List[StateTransitionGraph], ViolatedLaws,]:
        propagator = self._initialize_qn_propagator(
            state_graph_node_settings_pair[0],
            state_graph_node_settings_pair[1],
        )
        solutions = propagator.find_solutions()
        return (solutions, propagator.get_non_satisfied_conservation_laws())

    def _initialize_qn_propagator(
        self,
        state_graph: StateTransitionGraph,
        node_settings: Dict[int, InteractionNodeSettings],
    ) -> FullPropagator:
        propagator = FullPropagator(
            state_graph,
            self.__allowed_intermediate_particles,
            self.propagation_mode,
        )
        for node_id, interaction_settings in node_settings.items():
            propagator.assign_settings_to_node(node_id, interaction_settings)

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


def load_default_particles() -> ParticleCollection:
    """Load the default particle list that comes with the expertsystem.

    .. warning::
        This resets all particle definitions and the removes particles that
        don't exist in the particle list that ships with the `expertsystem`!
    """
    particles = io.load_pdg()
    particles.merge(io.load_particle_collection(DEFAULT_PARTICLE_LIST_PATH))
    logging.info(f"Loaded {len(particles)} particles!")
    return particles
