"""Definition and solving of particle reaction problems.

This is the core component of the `expertsystem`: it defines the
`.StateTransitionGraph` data structure that represents a specific particle
reaction. The `solving` submodule is responsible for finding solutions for
particle reaction problems.
"""

# pylint: disable=duplicate-code,too-many-lines

import logging
import multiprocessing
from collections import defaultdict
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

import attr
from tqdm import tqdm

from expertsystem import io
from expertsystem.particle import Particle, ParticleCollection
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

from ._system_control import (
    CompareGraphNodePropertiesFunctor,
    GammaCheck,
    InteractionDeterminator,
    LeptonCheck,
    create_edge_properties,
    create_interaction_properties,
    create_node_properties,
    create_particle,
    filter_interaction_types,
    remove_duplicate_solutions,
)
from .combinatorics import (
    InitialFacts,
    StateDefinition,
    create_initial_facts,
    match_external_edges,
)
from .default_settings import (
    DEFAULT_PARTICLE_LIST_PATH,
    InteractionTypes,
    create_default_interaction_settings,
)
from .quantum_numbers import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumber,
    NodeQuantumNumbers,
    ParticleWithSpin,
)
from .solving import (
    CSPSolver,
    EdgeSettings,
    GraphEdgePropertyMap,
    GraphElementProperties,
    GraphSettings,
    NodeSettings,
    QNProblemSet,
    QNResult,
    Rule,
    validate_full_solution,
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


class Result:
    """Defines a result of a `.ProblemSet`.

    Returned by the `.StateTransitionManager`
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        solutions: Optional[
            List[StateTransitionGraph[ParticleWithSpin]]
        ] = None,
        not_executed_node_rules: Optional[Dict[int, Set[str]]] = None,
        violated_node_rules: Optional[Dict[int, Set[str]]] = None,
        not_executed_edge_rules: Optional[Dict[int, Set[str]]] = None,
        violated_edge_rules: Optional[Dict[int, Set[str]]] = None,
        formalism_type: Optional[str] = None,
    ) -> None:
        # pylint: disable=too-many-locals
        if solutions and (violated_node_rules or violated_edge_rules):
            raise ValueError(
                "Invalid Result! Found solutions, but also violated rules."
            )

        self.__formalism_type = formalism_type
        self.__solutions: List[StateTransitionGraph[ParticleWithSpin]] = list()
        if solutions is not None:
            self.__solutions = solutions

        self.__not_executed_node_rules: Dict[int, Set[str]] = defaultdict(set)
        if not_executed_node_rules is not None:
            self.__not_executed_node_rules = not_executed_node_rules

        self.__violated_node_rules: Dict[int, Set[str]] = defaultdict(set)
        if violated_node_rules is not None:
            self.__violated_node_rules = violated_node_rules

        self.__not_executed_edge_rules: Dict[int, Set[str]] = defaultdict(set)
        if not_executed_edge_rules is not None:
            self.__not_executed_edge_rules = not_executed_edge_rules

        self.__violated_edge_rules: Dict[int, Set[str]] = defaultdict(set)
        if violated_edge_rules is not None:
            self.__violated_edge_rules = violated_edge_rules

    @property
    def formalism_type(self) -> Optional[str]:
        return self.__formalism_type

    @property
    def solutions(self) -> List[StateTransitionGraph[ParticleWithSpin]]:
        return self.__solutions

    @property
    def not_executed_node_rules(self) -> Dict[int, Set[str]]:
        return self.__not_executed_node_rules

    @property
    def violated_node_rules(self) -> Dict[int, Set[str]]:
        return self.__violated_node_rules

    @property
    def not_executed_edge_rules(self) -> Dict[int, Set[str]]:
        return self.__not_executed_edge_rules

    @property
    def violated_edge_rules(self) -> Dict[int, Set[str]]:
        return self.__violated_edge_rules

    def extend(
        self, other_result: "Result", intersect_violations: bool = False
    ) -> None:
        if self.solutions or other_result.solutions:
            self.__solutions.extend(other_result.solutions)
            self.__not_executed_node_rules.clear()
            self.__violated_node_rules.clear()
            self.__not_executed_edge_rules.clear()
            self.__violated_edge_rules.clear()
        else:
            for key, rules in other_result.not_executed_node_rules.items():
                self.__not_executed_node_rules[key].update(rules)

            for key, rules in other_result.not_executed_edge_rules.items():
                self.__not_executed_edge_rules[key].update(rules)

            for key, rules2 in other_result.violated_node_rules.items():
                if intersect_violations:
                    self.__violated_node_rules[key] &= rules2
                else:
                    self.__violated_node_rules[key].update(rules2)

            for key, rules2 in other_result.violated_edge_rules.items():
                if intersect_violations:
                    self.__violated_edge_rules[key] &= rules2
                else:
                    self.__violated_edge_rules[key].update(rules2)

    def get_initial_state(self) -> List[Particle]:
        graph = self.__get_first_graph()
        return [
            x[0]
            for x in map(
                graph.get_edge_props, graph.get_initial_state_edge_ids()
            )
            if x
        ]

    def get_final_state(self) -> List[Particle]:
        graph = self.__get_first_graph()
        return [
            x[0]
            for x in map(
                graph.get_edge_props, graph.get_final_state_edge_ids()
            )
            if x
        ]

    def __get_first_graph(self) -> StateTransitionGraph[ParticleWithSpin]:
        if len(self.solutions) == 0:
            raise ValueError(
                f"No solutions in {self.__class__.__name__} object"
            )
        return self.solutions[0]

    def get_intermediate_particles(self) -> ParticleCollection:
        """Extract the names of the intermediate state particles."""
        intermediate_states = ParticleCollection()
        for graph in self.solutions:
            for edge_props in map(
                graph.get_edge_props, graph.get_intermediate_state_edge_ids()
            ):
                if edge_props:
                    particle, _ = edge_props
                    if particle not in intermediate_states:
                        intermediate_states.add(particle)
        return intermediate_states

    def get_particle_graphs(self) -> List[StateTransitionGraph[Particle]]:
        """Strip `list` of `.StateTransitionGraph` s of the spin projections.

        Extract a `list` of `.StateTransitionGraph` instances with only
        particles on the edges.

        .. seealso:: :doc:`/usage/visualization`
        """
        inventory: List[StateTransitionGraph[Particle]] = list()
        for graph in self.solutions:
            if any(
                [
                    graph.compare(
                        other, edge_comparator=lambda e1, e2: e1[0] == e2
                    )
                    for other in inventory
                ]
            ):
                continue
            new_edge_props = dict()
            for edge_id in graph.edges:
                edge_props = graph.get_edge_props(edge_id)
                if edge_props:
                    new_edge_props[edge_id] = edge_props[0]
            inventory.append(
                StateTransitionGraph[Particle](
                    topology=Topology(
                        nodes=set(graph.nodes), edges=graph.edges
                    ),
                    node_props={
                        i: node_props
                        for i, node_props in zip(
                            graph.nodes, map(graph.get_node_props, graph.nodes)
                        )
                        if node_props
                    },
                    edge_props=new_edge_props,
                )
            )
        inventory = sorted(
            inventory,
            key=lambda g: [
                g.get_edge_props(i).mass
                for i in g.get_intermediate_state_edge_ids()
            ],
        )
        return inventory

    def collapse_graphs(
        self,
    ) -> List[StateTransitionGraph[ParticleCollection]]:
        def merge_into(
            graph: StateTransitionGraph[Particle],
            merged_graph: StateTransitionGraph[ParticleCollection],
        ) -> None:
            if (
                graph.get_intermediate_state_edge_ids()
                != merged_graph.get_intermediate_state_edge_ids()
            ):
                raise ValueError(
                    "Cannot merge graphs that don't have the same edge IDs"
                )
            for i in graph.edges:
                particle = graph.get_edge_props(i)
                other_particles = merged_graph.get_edge_props(i)
                if particle not in other_particles:
                    other_particles += particle

        def is_same_shape(
            graph: StateTransitionGraph[Particle],
            merged_graph: StateTransitionGraph[ParticleCollection],
        ) -> bool:
            if graph.edges != merged_graph.edges:
                return False
            for edge_id in (
                graph.get_initial_state_edge_ids()
                + graph.get_final_state_edge_ids()
            ):
                edge_prop = merged_graph.get_edge_props(edge_id)
                if len(edge_prop) != 1:
                    return False
                other_particle = next(iter(edge_prop))
                if other_particle != graph.get_edge_props(edge_id):
                    return False
            return True

        graphs = self.get_particle_graphs()
        inventory: List[StateTransitionGraph[ParticleCollection]] = list()
        for graph in graphs:
            append_to_inventory = True
            for merged_graph in inventory:
                if is_same_shape(graph, merged_graph):
                    merge_into(graph, merged_graph)
                    append_to_inventory = False
                    break
            if append_to_inventory:
                new_edge_props = {
                    edge_id: ParticleCollection(
                        {graph.get_edge_props(edge_id)}
                    )
                    for edge_id in graph.edges
                }
                inventory.append(
                    StateTransitionGraph[ParticleCollection](
                        topology=Topology(
                            nodes=set(graph.nodes), edges=graph.edges
                        ),
                        node_props={
                            i: graph.get_node_props(i) for i in graph.nodes
                        },
                        edge_props=new_edge_props,
                    )
                )
        return inventory


@attr.s
class ProblemSet:
    """Particle reaction problem set, defined as a graph like data structure.

    Args:
        topology: `~.Topology` that contains the structure of the reaction.
        initial_facts: `~.InitialFacts` that contain the info of initial and
          final state in connection with the topology.
        solving_settings: Solving related settings such as the conservation
          rules and the quantum number domains.
    """

    topology: Topology = attr.ib()
    initial_facts: InitialFacts = attr.ib()
    solving_settings: GraphSettings = attr.ib()


def _group_by_strength(
    problem_sets: List[ProblemSet],
) -> Dict[float, List[ProblemSet]]:
    def calculate_strength(
        node_interaction_settings: Dict[int, NodeSettings]
    ) -> float:
        strength = 1.0
        for int_setting in node_interaction_settings.values():
            strength *= int_setting.interaction_strength
        return strength

    strength_sorted_problem_sets: Dict[float, List[ProblemSet]] = defaultdict(
        list
    )
    for problem_set in problem_sets:
        strength = calculate_strength(
            problem_set.solving_settings.node_settings
        )
        strength_sorted_problem_sets[strength].append(problem_set)
    return strength_sorted_problem_sets


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

        self.__user_allowed_intermediate_particles = (
            allowed_intermediate_particles
        )
        self.__allowed_intermediate_particles: List[
            GraphEdgePropertyMap
        ] = list()
        if allowed_intermediate_particles is not None:
            self.set_allowed_intermediate_particles(
                allowed_intermediate_particles
            )
        else:
            self.__allowed_intermediate_particles = [
                create_edge_properties(x) for x in self.__particles
            ]

    def set_allowed_intermediate_particles(
        self, particle_names: List[str]
    ) -> None:
        self.__allowed_intermediate_particles = list()
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
            self.__allowed_intermediate_particles += [
                create_edge_properties(x) for x in matches
            ]

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

    def create_problem_sets(self) -> Dict[float, List[ProblemSet]]:
        topology_graphs = self.__build_topologies()
        problem_sets = []
        for topology in topology_graphs:
            for initial_facts in self.__create_initial_facts(topology):
                problem_sets.extend(
                    [
                        ProblemSet(
                            topology=topology,
                            initial_facts=initial_facts,
                            solving_settings=x,
                        )
                        for x in self.__determine_graph_settings(
                            topology, initial_facts
                        )
                    ]
                )
        # create groups of settings ordered by "probability"
        return _group_by_strength(problem_sets)

    def __build_topologies(self) -> List[Topology]:
        all_graphs = self.topology_builder.build_graphs(
            len(self.initial_state), len(self.final_state)
        )
        logging.info(f"number of topology graphs: {len(all_graphs)}")
        return all_graphs

    def __create_initial_facts(self, topology: Topology) -> List[InitialFacts]:
        initial_facts = create_initial_facts(
            topology=topology,
            particles=self.__particles,
            initial_state=self.initial_state,
            final_state=self.final_state,
            final_state_groupings=self.final_state_groupings,
        )

        logging.info(f"initialized {len(initial_facts)} graphs!")
        return initial_facts

    def __determine_graph_settings(
        self, topology: Topology, initial_facts: InitialFacts
    ) -> List[GraphSettings]:
        # pylint: disable=too-many-locals
        def create_intermediate_edge_qn_domains() -> Dict:
            # if a list of intermediate states is given by user,
            # built a domain based on these states
            if self.__user_allowed_intermediate_particles:
                intermediate_edge_domains: Dict[
                    Type[EdgeQuantumNumber], Set
                ] = defaultdict(set)
                intermediate_edge_domains[
                    EdgeQuantumNumbers.spin_projection
                ].update(
                    self.interaction_type_settings[InteractionTypes.Weak][
                        0
                    ].qn_domains[EdgeQuantumNumbers.spin_projection]
                )
                for particle_props in self.__allowed_intermediate_particles:
                    for edge_qn, qn_value in particle_props.items():
                        intermediate_edge_domains[edge_qn].add(qn_value)

                return dict(
                    {
                        k: list(v)
                        for k, v in intermediate_edge_domains.items()
                        if k is not EdgeQuantumNumbers.pid
                        and k is not EdgeQuantumNumbers.mass
                        and k is not EdgeQuantumNumbers.width
                    }
                )

            return self.interaction_type_settings[InteractionTypes.Weak][
                0
            ].qn_domains

        intermediate_state_edges = topology.get_intermediate_state_edge_ids()
        int_edge_domains = create_intermediate_edge_qn_domains()

        def create_edge_settings(edge_id: int) -> EdgeSettings:
            settings = copy(
                self.interaction_type_settings[InteractionTypes.Weak][0]
            )
            if edge_id in intermediate_state_edges:
                settings.qn_domains = int_edge_domains
            else:
                settings.qn_domains = {}
            return settings

        final_state_edges = topology.get_final_state_edge_ids()
        initial_state_edges = topology.get_initial_state_edge_ids()

        graph_settings: List[GraphSettings] = [
            GraphSettings(
                edge_settings={
                    edge_id: create_edge_settings(edge_id)
                    for edge_id in topology.edges
                },
                node_settings={},
            )
        ]

        for node_id in topology.nodes:
            interaction_types: List[InteractionTypes] = []
            out_edge_ids = topology.get_edge_ids_outgoing_from_node(node_id)
            in_edge_ids = topology.get_edge_ids_outgoing_from_node(node_id)
            in_edge_props = [
                initial_facts.edge_props[edge_id]
                for edge_id in [
                    x for x in in_edge_ids if x in initial_state_edges
                ]
            ]
            out_edge_props = [
                initial_facts.edge_props[edge_id]
                for edge_id in [
                    x for x in out_edge_ids if x in final_state_edges
                ]
            ]
            node_props = InteractionProperties()
            if node_id in initial_facts.node_props:
                node_props = initial_facts.node_props[node_id]
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
        problem_sets: Dict[float, List[ProblemSet]],
    ) -> Result:
        # pylint: disable=too-many-locals
        """Check for solutions for a specific set of interaction settings."""
        results: Dict[float, Result] = {}
        logging.info(
            "Number of interaction settings groups being processed: %d",
            len(problem_sets),
        )
        total = sum(map(len, problem_sets.values()))
        progress_bar = tqdm(
            total=total,
            desc="Propagating quantum numbers",
            disable=logging.getLogger().level > logging.WARNING,
        )
        for strength, problems in sorted(problem_sets.items(), reverse=True):
            logging.info(
                "processing interaction settings group with "
                f"strength {strength}",
            )
            logging.info(f"{len(problems)} entries in this group")
            logging.info(f"running with {self.number_of_threads} threads...")

            qn_problems = [_convert_to_qn_problem_set(x) for x in problems]

            # Because of pickling problems of Generic classes (in this case
            # StateTransitionGraph), multithreaded code has to work with
            # QNProblemSet's and QNResult's. So the appropriate conversions
            # have to be done before and after
            temp_qn_results: List[Tuple[QNProblemSet, QNResult]] = []
            if self.number_of_threads > 1:
                with Pool(self.number_of_threads) as pool:
                    for qn_result in pool.imap_unordered(
                        self._solve, qn_problems, 1
                    ):
                        temp_qn_results.append(qn_result)
                        progress_bar.update()
            else:
                for problem in qn_problems:
                    temp_qn_results.append(self._solve(problem))
                    progress_bar.update()
            for temp_qn_result in temp_qn_results:
                temp_result = self.__convert_result(
                    temp_qn_result[0].topology,
                    temp_qn_result[1],
                )
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
        self, qn_problem_set: QNProblemSet
    ) -> Tuple[QNProblemSet, QNResult]:
        solver = CSPSolver(self.__allowed_intermediate_particles)

        return (qn_problem_set, solver.find_solutions(qn_problem_set))

    def __convert_result(
        self, topology: Topology, qn_result: QNResult
    ) -> Result:
        """Converts a `.QNResult` with a `.Topology` into a `.Result`.

        The ParticleCollection is used to retrieve a particle instance
        reference to lower the memory footprint.
        """
        solutions = []
        for solution in qn_result.solutions:
            graph = StateTransitionGraph[ParticleWithSpin](
                topology=topology,
                node_props={
                    i: create_interaction_properties(x)
                    for i, x in solution.node_quantum_numbers.items()
                },
                edge_props={
                    i: create_particle(x, self.__particles)
                    for i, x in solution.edge_quantum_numbers.items()
                },
            )
            graph.graph_node_properties_comparator = (
                CompareGraphNodePropertiesFunctor()
            )
            solutions.append(graph)

        return Result(
            solutions=solutions,
            violated_edge_rules=qn_result.violated_edge_rules,
            violated_node_rules=qn_result.violated_node_rules,
            not_executed_node_rules=qn_result.not_executed_node_rules,
            not_executed_edge_rules=qn_result.not_executed_edge_rules,
            formalism_type=self.__formalism_type,
        )


def _convert_to_qn_problem_set(
    problem_set: ProblemSet,
) -> QNProblemSet:
    node_props = {
        k: create_node_properties(v)
        for k, v in problem_set.initial_facts.node_props.items()
    }
    edge_props = {
        k: create_edge_properties(v[0], v[1])
        for k, v in problem_set.initial_facts.edge_props.items()
    }

    return QNProblemSet(
        topology=problem_set.topology,
        initial_facts=GraphElementProperties(
            node_props=node_props, edge_props=edge_props
        ),
        solving_settings=problem_set.solving_settings,
    )


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
        facts: InitialFacts,
        node_rules: Dict[int, Set[Rule]],
        edge_rules: Dict[int, Set[GraphElementRule]],
    ) -> QNResult:
        return validate_full_solution(
            _convert_to_qn_problem_set(
                ProblemSet(
                    topology=topology,
                    initial_facts=facts,
                    solving_settings=GraphSettings(
                        node_settings={
                            i: NodeSettings(conservation_rules=rules)
                            for i, rules in node_rules.items()
                        },
                        edge_settings={
                            i: EdgeSettings(conservation_rules=rules)
                            for i, rules in edge_rules.items()
                        },
                    ),
                )
            )
        )

    def check_pure_edge_rules() -> None:
        pure_edge_rules: Set[GraphElementRule] = {
            gellmann_nishijima,
            isospin_validity,
        }

        edge_check_result = _check_violations(
            initial_facts[0],
            node_rules={},
            edge_rules={
                edge_id: pure_edge_rules
                for edge_id in topology.get_initial_state_edge_ids()
                + topology.get_final_state_edge_ids()
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

    def check_edge_qn_conservation() -> Set[FrozenSet[str]]:
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
            for x in _check_violations(
                initial_facts[0],
                node_rules={
                    i: edge_qn_conservation_rules for i in topology.nodes
                },
                edge_rules={},
            ).violated_node_rules[node_id]
        }

    # Using a n-body topology is enough, to determine the violations reliably
    # since only certain spin rules require the isobar model. These spin rules
    # are not required here though.
    topology = create_n_body_topology()
    node_id = next(iter(topology.nodes))

    initial_facts = create_initial_facts(
        topology=topology,
        particles=load_default_particles(),
        initial_state=initial_state,
        final_state=final_state,
    )

    check_pure_edge_rules()
    violations = check_edge_qn_conservation()

    # Create combinations of graphs for magnitudes of S and L, but only
    # if it is a two body reaction
    ls_combinations = [
        InteractionProperties(l_magnitude=l_mag, s_magnitude=s_mag)
        for l_mag, s_mag in product([0, 1], [0, 0.5, 1, 1.5, 2])
    ]

    initial_facts_list = []
    for ls_combi in ls_combinations:
        for facts_combination in initial_facts:
            new_facts = attr.evolve(
                facts_combination,
                node_props={node_id: ls_combi},
            )
            initial_facts_list.append(new_facts)

    # Verify each graph with the interaction rules.
    # Spin projection rules are skipped as they can only be checked reliably
    # for a isobar topology (too difficult to solve)
    conservation_rules: Dict[int, Set[Rule]] = {
        node_id: {
            c_parity_conservation,
            clebsch_gordan_helicity_to_canonical,
            g_parity_conservation,
            parity_conservation,
            spin_magnitude_conservation,
            identical_particle_symmetrization,
        }
    }

    conservation_rule_violations: List[Set[str]] = []
    for facts in initial_facts_list:
        rule_violations = _check_violations(
            facts=facts, node_rules=conservation_rules, edge_rules={}
        ).violated_node_rules[node_id]
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
    problem_sets = stm.create_problem_sets()
    return stm.find_solutions(problem_sets)


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
