"""Main user interface of the `expertsystem`.

This module contains the functions that you need for the most common use cases
of the `expertsystem`. See the :doc:`/usage/quickstart`.
"""

import logging
from copy import deepcopy
from enum import Enum, auto
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type, Union

from progress.bar import IncrementalBar

from expertsystem import io
from expertsystem.amplitude.canonical_decay import CanonicalAmplitudeGenerator
from expertsystem.amplitude.helicity_decay import HelicityAmplitudeGenerator
from expertsystem.amplitude.model import AmplitudeModel
from expertsystem.particle import Particle, ParticleCollection
from expertsystem.reaction.combinatorics import (
    StateDefinition,
    initialize_graph,
    match_external_edges,
)
from expertsystem.reaction.quantum_numbers import (
    InteractionProperties,
    NodeQuantumNumber,
    NodeQuantumNumbers,
    ParticleWithSpin,
)
from expertsystem.reaction.solving import (
    CSPSolver,
    EdgeSettings,
    GraphSettings,
    InteractionTypes,
    NodeSettings,
    Result,
)
from expertsystem.reaction.topology import (
    InteractionNode,
    SimpleStateTransitionTopologyBuilder,
    StateTransitionGraph,
    Topology,
)

from ._default_settings import (
    DEFAULT_PARTICLE_LIST_PATH,
    create_default_interaction_settings,
)
from ._system_control import (
    CompareGraphNodePropertiesFunctor,
    GammaCheck,
    GraphSettingsGroups,
    InteractionDeterminator,
    LeptonCheck,
    filter_interaction_types,
    group_by_strength,
    remove_duplicate_solutions,
)


class SolvingMode(Enum):
    """Types of modes for solving."""

    Fast = auto()
    """find "likeliest" solutions only"""
    Full = auto()
    """find all possible solutions"""


class StateTransitionManager:  # pylint: disable=too-many-instance-attributes
    """Main handler for decay topologies."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        initial_state: Sequence[StateDefinition],
        final_state: Sequence[StateDefinition],
        particles: ParticleCollection = ParticleCollection(),
        allowed_intermediate_particles: Optional[List[str]] = None,
        interaction_type_settings: Dict[
            InteractionTypes, Tuple[EdgeSettings, NodeSettings]
        ] = None,
        formalism_type: str = "helicity",
        topology_building: str = "isobar",
        number_of_threads: int = 4,
        solving_mode: SolvingMode = SolvingMode.Fast,
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
        self.reaction_mode = str(solving_mode)
        self.initial_state = initial_state
        self.final_state = final_state
        self.interaction_type_settings = interaction_type_settings

        self.interaction_determinators: List[InteractionDeterminator] = [
            LeptonCheck(),
            GammaCheck(),
        ]
        self.final_state_groupings: Optional[List[List[List[str]]]] = None
        self.allowed_interaction_types: List[InteractionTypes] = [
            InteractionTypes.Strong,
            InteractionTypes.EM,
            InteractionTypes.Weak,
        ]
        self.filter_remove_qns: Set[Type[NodeQuantumNumber]] = set()
        self.filter_ignore_qns: Set[Type[NodeQuantumNumber]] = set()
        if formalism_type == "helicity":
            self.filter_remove_qns = {
                NodeQuantumNumbers.l_magnitude,
                NodeQuantumNumbers.l_projection,
                NodeQuantumNumbers.s_magnitude,
                NodeQuantumNumbers.s_projection,
            }
        if "helicity" in formalism_type:
            self.filter_ignore_qns = {NodeQuantumNumbers.parity_prefactor}
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

        self.__allowed_intermediate_particles = self.__particles
        if allowed_intermediate_particles is not None:
            self.set_allowed_intermediate_particles(
                allowed_intermediate_particles
            )

    def set_allowed_intermediate_particles(
        self, particle_names: List[str]
    ) -> None:
        self.__allowed_intermediate_particles = ParticleCollection()
        for particle_name in particle_names:
            self.__allowed_intermediate_particles += self.__particles.filter(
                lambda p: particle_name  # pylint: disable=cell-var-from-loop
                in p.name
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
        seed_graphs = self._create_seed_graphs(topology_graphs)
        graph_setting_pairs = []
        for seed_graph in seed_graphs:
            graph_setting_pairs.extend(
                [
                    (seed_graph, x)
                    for x in self._determine_graph_settings(seed_graph)
                ]
            )
        # create groups of settings ordered by "probability"
        graph_settings_groups = group_by_strength(graph_setting_pairs)
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
        for graph in init_graphs:
            graph.graph_node_properties_comparator = (
                CompareGraphNodePropertiesFunctor()
            )

        logging.info(f"initialized {len(init_graphs)} graphs!")
        return init_graphs

    def _determine_graph_settings(
        self, graph: StateTransitionGraph[ParticleWithSpin]
    ) -> List[GraphSettings]:
        # pylint: disable=too-many-locals
        final_state_edges = graph.get_final_state_edges()
        initial_state_edges = graph.get_initial_state_edges()
        graph_settings: List[GraphSettings] = [GraphSettings({}, {})]

        for node_id in graph.nodes:
            interaction_types: List[InteractionTypes] = []
            out_edge_ids = graph.get_edges_outgoing_from_node(node_id)
            in_edge_ids = graph.get_edges_outgoing_from_node(node_id)
            in_edge_props = [
                graph.edge_props[edge_id]
                for edge_id in [
                    x for x in in_edge_ids if x in initial_state_edges
                ]
            ]
            out_edge_props = [
                graph.edge_props[edge_id]
                for edge_id in [
                    x for x in out_edge_ids if x in final_state_edges
                ]
            ]
            node_props = InteractionProperties()
            if node_id in graph.node_props:
                node_props = graph.node_props[node_id]
            for int_det in self.interaction_determinators:
                determined_interactions = int_det.check(
                    in_edge_props, out_edge_props, node_props
                )
                if interaction_types:
                    interaction_types = list(
                        set(determined_interactions) & set(interaction_types)
                    )
                else:
                    interaction_types = determined_interactions
            interaction_types = filter_interaction_types(
                interaction_types, self.allowed_interaction_types
            )
            logging.debug(
                "using %s interaction order for node: %s",
                str(interaction_types),
                str(node_id),
            )

            temp_graph_settings: List[GraphSettings] = graph_settings
            graph_settings = []
            for temp_setting in temp_graph_settings:
                for int_type in interaction_types:
                    updated_setting = deepcopy(temp_setting)
                    updated_setting.edge_settings[node_id] = deepcopy(
                        self.interaction_type_settings[int_type][0]
                    )
                    updated_setting.node_settings[node_id] = deepcopy(
                        self.interaction_type_settings[int_type][1]
                    )
                    graph_settings.append(updated_setting)

        return graph_settings

    def find_solutions(
        self,
        graph_setting_groups: GraphSettingsGroups,
    ) -> Result:  # pylint: disable=too-many-locals
        """Check for solutions for a specific set of interaction settings."""
        results: Dict[float, Result] = {}
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

            temp_results: List[Result] = []
            progress_bar = IncrementalBar(
                "Propagating quantum numbers...", max=len(graph_setting_group)
            )
            progress_bar.update()
            if self.number_of_threads > 1:
                with Pool(self.number_of_threads) as pool:
                    for result in pool.imap_unordered(
                        self._solve, graph_setting_group, 1
                    ):
                        temp_results.append(result)
                        progress_bar.next()
            else:
                for graph_setting_pair in graph_setting_group:
                    temp_results.append(self._solve(graph_setting_pair))
                    progress_bar.next()
            progress_bar.finish()
            logging.info("Finished!")
            for temp_result in temp_results:
                if strength not in results:
                    results[strength] = temp_result
                else:
                    results[strength].extend(temp_result, True)
            if (
                results[strength].solutions
                and self.reaction_mode == SolvingMode.Fast
            ):
                break

        for key, result in results.items():
            logging.info(
                f"number of solutions for strength ({key}) "
                f"after qn solving: {len(result.solutions)}",
            )

        # merge strengths
        final_result = Result()
        for temp_result in results.values():
            final_result.extend(temp_result)

        # remove duplicate solutions, which only differ in the interaction qns
        final_solutions = remove_duplicate_solutions(
            final_result.solutions,
            self.filter_remove_qns,
            self.filter_ignore_qns,
        )

        if final_solutions:
            match_external_edges(final_solutions)
        return Result(
            final_solutions,
            final_result.not_executed_rules,
            final_result.violated_rules,
        )

    def _solve(
        self,
        state_graph_node_settings_pair: Tuple[
            StateTransitionGraph[ParticleWithSpin], GraphSettings
        ],
    ) -> Result:
        solver = CSPSolver(self.__allowed_intermediate_particles)

        return solver.find_solutions(*state_graph_node_settings_pair)

    def generate_amplitude_model(self, result: Result) -> AmplitudeModel:
        """Generate an amplitude model from a generated `.Result`.

        The type of amplitude model (`.HelicityAmplitudeGenerator` or
        `.CanonicalAmplitudeGenerator`) is determined from the
        :code:`formalism_type` that was chosen when constructing the
        `.StateTransitionManager`.
        """
        if self.formalism_type == "helicity":
            amplitude_generator = HelicityAmplitudeGenerator()
        elif self.formalism_type in ["canonical-helicity", "canonical"]:
            amplitude_generator = CanonicalAmplitudeGenerator()
        return amplitude_generator.generate(result.solutions)


def load_default_particles() -> ParticleCollection:
    """Load the default particle list that comes with the expertsystem.

    .. warning::
        This resets all particle definitions and the removes particles that
        don't exist in the particle list that ships with the `expertsystem`!
    """
    particles = io.load_pdg()
    particles.update(io.load_particle_collection(DEFAULT_PARTICLE_LIST_PATH))
    logging.info(f"Loaded {len(particles)} particles!")
    return particles


def get_intermediate_state_names(
    solutions: List[StateTransitionGraph],
) -> Set[str]:
    """Extract the names of the intermediate states in the solutions."""
    intermediate_states = set()
    for graph in solutions:
        for edge_id in graph.get_intermediate_state_edges():
            edge_property = graph.edge_props[edge_id]
            if isinstance(edge_property, dict):
                intermediate_states.add(edge_property["Name"])
            elif isinstance(edge_property, tuple) and isinstance(
                edge_property[0], Particle
            ):
                particle, _ = edge_property
                intermediate_states.add(particle.name)
            else:
                raise ValueError(
                    "Cannot extract name from edge property of type "
                    f"{edge_property.__class__.__name__}"
                )
    return intermediate_states
