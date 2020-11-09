"""Definition and solving of particle reaction problems.

This is the core component of the `expertsystem`: it defines the
`.StateTransitionGraph` data structure that represents a specific particle
reaction. The `solving` submodule is responsible for finding solutions for
particle reaction problems.
"""

import logging
import multiprocessing
from copy import copy, deepcopy
from enum import Enum, auto
from itertools import product
from multiprocessing import Pool
from typing import (
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from tqdm import tqdm

from expertsystem import io
from expertsystem.particle import ParticleCollection
from expertsystem.reaction.conservation_rules import (
    BaryonNumberConservation,
    BottomnessConservation,
    ChargeConservation,
    CharmConservation,
    ElectronLNConservation,
    GraphElementRule,
    MassConservation,
    MuonLNConservation,
    StrangenessConservation,
    TauLNConservation,
    c_parity_conservation,
    clebsch_gordan_helicity_to_canonical,
    g_parity_conservation,
    gellmann_nishijima,
    identical_particle_symmetrization,
    isospin_conservation,
    isospin_validity,
    parity_conservation,
    spin_magnitude_conservation,
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
from .combinatorics import (
    StateDefinition,
    initialize_graph,
    match_external_edges,
)
from .quantum_numbers import (
    InteractionProperties,
    NodeQuantumNumber,
    NodeQuantumNumbers,
    ParticleWithSpin,
)
from .solving import (
    CSPSolver,
    EdgeSettings,
    GraphSettings,
    InteractionTypes,
    NodeSettings,
    Result,
    Rule,
    validate_fully_initialized_graph,
)
from .topology import (
    InteractionNode,
    SimpleStateTransitionTopologyBuilder,
    StateTransitionGraph,
    Topology,
)


class SolvingMode(Enum):
    """Types of modes for solving."""

    Fast = auto()
    """find "likeliest" solutions only"""
    Full = auto()
    """find all possible solutions"""


class StateTransitionManager:  # pylint: disable=too-many-instance-attributes
    """Main handler for decay topologies.

    .. seealso:: :doc:`/usage/workflow` and `generate`
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-branches
        self,
        initial_state: Sequence[StateDefinition],
        final_state: Sequence[StateDefinition],
        particles: Optional[ParticleCollection] = None,
        allowed_intermediate_particles: Optional[List[str]] = None,
        interaction_type_settings: Dict[
            InteractionTypes, Tuple[EdgeSettings, NodeSettings]
        ] = None,
        formalism_type: str = "helicity",
        topology_building: str = "isobar",
        number_of_threads: Optional[int] = None,
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
        self.__particles = ParticleCollection()
        if particles is not None:
            self.__particles = particles
        if number_of_threads is None:
            self.number_of_threads = multiprocessing.cpu_count()
        else:
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
            matches = self.__particles.filter(
                lambda p: particle_name  # pylint: disable=cell-var-from-loop
                in p.name
            )
            if len(matches) == 0:
                raise LookupError(
                    "Could not find any matches for allowed intermediate "
                    f' particle "{particle_name}"'
                )
            self.__allowed_intermediate_particles += matches

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
        graph_settings: List[GraphSettings] = [
            GraphSettings(
                edge_settings={
                    edge_id: self.interaction_type_settings[
                        InteractionTypes.Weak
                    ][0]
                    for edge_id in graph.edges
                },
                node_settings={},
            )
        ]

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
        total = sum(map(len, graph_setting_groups.values()))
        progress_bar = tqdm(
            total=total,
            desc="Propagating quantum numbers",
            disable=logging.getLogger().level > logging.WARNING,
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
            if self.number_of_threads > 1:
                with Pool(self.number_of_threads) as pool:
                    for result in pool.imap_unordered(
                        self._solve, graph_setting_group, 1
                    ):
                        temp_results.append(result)
                        progress_bar.update()
            else:
                for graph_setting_pair in graph_setting_group:
                    temp_results.append(self._solve(graph_setting_pair))
                    progress_bar.update()
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
        progress_bar.close()

        for key, result in results.items():
            logging.info(
                f"number of solutions for strength ({key}) "
                f"after qn solving: {len(result.solutions)}",
            )

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
            final_result.not_executed_node_rules,
            final_result.violated_node_rules,
            final_result.not_executed_edge_rules,
            final_result.violated_edge_rules,
            formalism_type=self.formalism_type,
        )

    def _solve(
        self,
        state_graph_node_settings_pair: Tuple[
            StateTransitionGraph[ParticleWithSpin], GraphSettings
        ],
    ) -> Result:
        solver = CSPSolver(self.__allowed_intermediate_particles)

        return solver.find_solutions(*state_graph_node_settings_pair)


def check_reaction_violations(
    initial_state: Union[StateDefinition, Sequence[StateDefinition]],
    final_state: Sequence[StateDefinition],
) -> Set[FrozenSet[str]]:
    """Determine violated interaction rules for a given particle reaction.

    .. warning:: This function does only guarantees to find P, C and G parity
      violations, if it's a two body decay. If all initial and final states
      have the C/G parity defined, then these violations are also determined
      correctly.

    Args:
      initial_state: Shortform description of the initial state w/o spin
        projections.
      final_state: Shortform description of the final state w/o spin
        projections.

    Returns:
      Set of least violating rules. The set can have multiple entries, as
      several quantum numbers can be violated. Each entry in the frozenset
      represents a group of rules that together violate all possible quantum
      number configurations.
    """
    # pylint: disable=too-many-locals
    if not isinstance(initial_state, (list, tuple)):
        initial_state = [initial_state]  # type: ignore

    def _check_violations(
        graph: StateTransitionGraph, node_rules: Set[Rule]
    ) -> Set[str]:
        node_id = list(graph.nodes)[0]
        return validate_fully_initialized_graph(
            graph,
            rules_per_node={node_id: node_rules},
            rules_per_edge={},
        ).violated_node_rules[node_id]

    def check_pure_edge_rules(
        graph: StateTransitionGraph[ParticleWithSpin],
    ) -> None:
        pure_edge_rules: Set[GraphElementRule] = {
            gellmann_nishijima,
            isospin_validity,
        }

        edge_check_result = validate_fully_initialized_graph(
            graph,
            rules_per_node={},
            rules_per_edge={
                edge_id: pure_edge_rules
                for edge_id in graph.get_initial_state_edges()
                + graph.get_final_state_edges()
            },
        )

        if edge_check_result.violated_edge_rules:
            raise ValueError(
                f"Some edges violate"
                f" {edge_check_result.violated_edge_rules.values()}"
            )

    def create_n_body_topology() -> Topology:
        topology_builder = SimpleStateTransitionTopologyBuilder(
            [
                InteractionNode(
                    "NBodyScattering", len(initial_state), len(final_state)
                )
            ]
        )
        return topology_builder.build_graphs(
            len(initial_state), len(final_state)
        )[0]

    def check_edge_qn_conservation(
        graph: StateTransitionGraph[ParticleWithSpin],
    ) -> Set[FrozenSet[str]]:
        """Check if edge quantum numbers are conserved.

        Those rules give the same results, independent on the node and spin
        props. Note they are also independent of the topology and hence their
        results are always correct.
        """
        edge_qn_conservation_rules: Set[Rule] = {
            BaryonNumberConservation(),
            BottomnessConservation(),
            ChargeConservation(),
            CharmConservation(),
            ElectronLNConservation(),
            MuonLNConservation(),
            StrangenessConservation(),
            TauLNConservation(),
            isospin_conservation,
        }
        if len(initial_state) == 1:
            edge_qn_conservation_rules.add(MassConservation(5))

        return {
            frozenset((x,))
            for x in _check_violations(graph, edge_qn_conservation_rules)
        }

    # Using a n-body topology is enough, to determine the violations reliably
    # since only certain spin rules require the isobar model. These spin rules
    # are not required here though.
    topology = create_n_body_topology()

    initialized_graphs = initialize_graph(
        topology=topology,
        particles=load_default_particles(),
        initial_state=initial_state,
        final_state=final_state,
    )

    check_pure_edge_rules(initialized_graphs[0])
    violations = check_edge_qn_conservation(initialized_graphs[0])

    # Create combinations of graphs for magnitudes of S and L, but only
    # if it is a two body reaction
    ls_combinations = [
        InteractionProperties(l_magnitude=l_mag, s_magnitude=s_mag)
        for l_mag, s_mag in product([0, 1], [0, 0.5, 1, 1.5, 2])
    ]
    node_id = next(iter(topology.nodes))
    graphs = []
    for ls_combi in ls_combinations:
        for graph in initialized_graphs:
            new_graph = copy(graph)
            new_graph.node_props = {node_id: ls_combi}
            graphs.append(new_graph)

    # Verify each graph with the interaction rules.
    # Spin projection rules are skipped as they can only be checked reliably
    # for a isobar topology (too difficult to solve)
    conservation_rules: Set[Rule] = {
        c_parity_conservation,
        clebsch_gordan_helicity_to_canonical,
        g_parity_conservation,
        parity_conservation,
        spin_magnitude_conservation,
        identical_particle_symmetrization,
    }

    conservation_rule_violations: List[Set[str]] = []
    for graph in graphs:
        rule_violations = _check_violations(graph, conservation_rules)
        conservation_rule_violations.append(rule_violations)

    # first add rules which consistently fail
    common_ruleset = set(conservation_rule_violations[0])
    for rule_set in conservation_rule_violations[1:]:
        common_ruleset &= rule_set

    violations.update({frozenset((x,)) for x in common_ruleset})

    conservation_rule_violations = [
        x - common_ruleset for x in conservation_rule_violations
    ]

    # if there is not non-violated graph with the remaining violations then
    # the collection of violations also violate everything as a group.
    if all(map(len, conservation_rule_violations)):
        rule_group: Set[str] = set()
        for graph_violations in conservation_rule_violations:
            rule_group.update(graph_violations)
        violations.add(frozenset(rule_group))

    return violations


def load_default_particles() -> ParticleCollection:
    """Load the default particle list that comes with the `expertsystem`.

    Runs `.load_pdg` and supplements its output definitions from the file
    :download:`additional_particle_definitions.yml
    </../src/expertsystem/additional_particle_definitions.yml>`.
    """
    particles = io.load_pdg()
    particles.update(io.load_particle_collection(DEFAULT_PARTICLE_LIST_PATH))
    logging.info(f"Loaded {len(particles)} particles!")
    return particles


def generate(  # pylint: disable=too-many-arguments
    initial_state: Union[StateDefinition, Sequence[StateDefinition]],
    final_state: Sequence[StateDefinition],
    allowed_intermediate_particles: Optional[List[str]] = None,
    allowed_interaction_types: Optional[Union[str, List[str]]] = None,
    formalism_type: str = "helicity",
    particles: Optional[ParticleCollection] = None,
    topology_building: str = "isobar",
    number_of_threads: Optional[int] = None,
) -> Result:
    """Generate allowed transitions between an initial and final state.

    Serves as a facade to the `.StateTransitionManager` (see
    :doc:`/usage/workflow`).

    Arguments:
        initial_state (list): A list of particle names in the initial
            state. You can specify spin projections for these particles with a
            `tuple`, e.g. :code:`("J/psi(1S)", [-1, 0, +1])`. If spin
            projections are not specified, all projections are taken, so the
            example here would be equivalent to :code:`"J/psi(1S)"`.

        final_state (list): Same as :code:`initial_state`, but for final state
            particles.

        allowed_intermediate_particles (`list`, optional): A list of particle
            states that you want to allow as intermediate states. This helps
            (1) filter out resonances in the eventual `.AmplitudeModel` and (2)
            speed up computation time.

        allowed_interaction_types (`str`, optional): Interaction types you want
            to consider. For instance, both :code:`"strong and EM"` and
            :code:`["s", "em"]` results in `~.InteractionTypes.EM` and
            `~.InteractionTypes.Strong`.

        formalism_type (`str`, optional): Formalism that you intend to use in the
            eventual `.AmplitudeModel`.

        particles (`.ParticleCollection`, optional): The particles that you
            want to be involved in the reaction. Uses `.load_default_particles`
            by default. It's better to use a subset for larger reactions,
            because of the computation times. This argument is especially
            useful when you want to use your own particle definitions (see
            :doc:`/usage/particles`).

        topology_building (str): Technique with which to build the `.Topology`
            instances. Allowed values are:

            - :code:`"isobar"`: Isobar model (each state decays into two states)
            - :code:`"nbody"`: Use one central node and connect initial and final
              states to it

        number_of_threads (int): Number of cores with which to compute the
            allowed transitions. Defaults to all cores on the system.

    An example (where, for illustrative purposes only, we specify all
    arguments) would be:

    >>> import expertsystem as es
    >>> result = es.reaction.generate(
    ...     initial_state="D0",
    ...     final_state=["K~0", "K+", "K-"],
    ...     allowed_intermediate_particles=["a(0)(980)", "a(2)(1320)-"],
    ...     allowed_interaction_types="ew",
    ...     formalism_type="helicity",
    ...     particles=es.io.load_pdg(),
    ...     topology_building="isobar",
    ... )
    >>> len(result.solutions)
    4
    """
    if isinstance(initial_state, str) or (
        isinstance(initial_state, tuple)
        and len(initial_state) == 2
        and isinstance(initial_state[0], str)
    ):
        initial_state = [initial_state]  # type: ignore
    stm = StateTransitionManager(
        initial_state=initial_state,  # type: ignore
        final_state=final_state,
        particles=particles,
        allowed_intermediate_particles=allowed_intermediate_particles,
        formalism_type=formalism_type,
        topology_building=topology_building,
        number_of_threads=number_of_threads,
    )
    if allowed_interaction_types is not None:
        interaction_types = _determine_interaction_types(
            allowed_interaction_types
        )
        stm.set_allowed_interaction_types(list(interaction_types))
    graph_interaction_settings_groups = stm.prepare_graphs()
    return stm.find_solutions(graph_interaction_settings_groups)


def _determine_interaction_types(
    description: Union[str, List[str]]
) -> Set[InteractionTypes]:
    interaction_types: Set[InteractionTypes] = set()
    if isinstance(description, list):
        for i in description:
            interaction_types.update(
                _determine_interaction_types(description=i)
            )
        return interaction_types
    if not isinstance(description, str):
        raise ValueError(
            "Cannot handle interaction description of type "
            f"{description.__class__.__name__}"
        )
    if len(description) == 0:
        raise ValueError('Provided an empty interaction name ("")')
    interaction_name_lower = description.lower()
    if "all" in interaction_name_lower:
        for interaction in InteractionTypes:
            interaction_types.add(interaction)
    if (
        "em" in interaction_name_lower
        or "ele" in interaction_name_lower
        or interaction_name_lower.startswith("e")
    ):
        interaction_types.add(InteractionTypes.EM)
    if "w" in interaction_name_lower:
        interaction_types.add(InteractionTypes.Weak)
    if "strong" in interaction_name_lower or interaction_name_lower == "s":
        interaction_types.add(InteractionTypes.Strong)
    if len(interaction_types) == 0:
        raise ValueError(
            f'Could not determine interaction type from "{description}"'
        )
    return interaction_types
