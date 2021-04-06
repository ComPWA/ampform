"""Find allowed transitions between an initial and final state."""

import logging
import multiprocessing
from collections import defaultdict
from copy import copy, deepcopy
from enum import Enum, auto
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type, Union

import attr
from tqdm.auto import tqdm

from ._system_control import (
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
    InteractionTypes,
    create_default_interaction_settings,
)
from .particle import Particle, ParticleCollection, ParticleWithSpin, load_pdg
from .quantum_numbers import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumber,
    NodeQuantumNumbers,
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
)
from .topology import (
    StateTransitionGraph,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
)


class SolvingMode(Enum):
    """Types of modes for solving."""

    FAST = auto()
    """Find "likeliest" solutions only."""
    FULL = auto()
    """Find all possible solutions."""


@attr.s(on_setattr=attr.setters.frozen)
class ExecutionInfo:
    not_executed_node_rules: Dict[int, Set[str]] = attr.ib(
        factory=lambda: defaultdict(set)
    )
    violated_node_rules: Dict[int, Set[str]] = attr.ib(
        factory=lambda: defaultdict(set)
    )
    not_executed_edge_rules: Dict[int, Set[str]] = attr.ib(
        factory=lambda: defaultdict(set)
    )
    violated_edge_rules: Dict[int, Set[str]] = attr.ib(
        factory=lambda: defaultdict(set)
    )

    def extend(
        self, other_result: "ExecutionInfo", intersect_violations: bool = False
    ) -> None:
        for key, rules in other_result.not_executed_node_rules.items():
            self.not_executed_node_rules[key].update(rules)

        for key, rules in other_result.not_executed_edge_rules.items():
            self.not_executed_edge_rules[key].update(rules)

        for key, rules2 in other_result.violated_node_rules.items():
            if intersect_violations:
                self.violated_node_rules[key] &= rules2
            else:
                self.violated_node_rules[key].update(rules2)

        for key, rules2 in other_result.violated_edge_rules.items():
            if intersect_violations:
                self.violated_edge_rules[key] &= rules2
            else:
                self.violated_edge_rules[key].update(rules2)

    def clear(self) -> None:
        self.not_executed_node_rules.clear()
        self.violated_node_rules.clear()
        self.not_executed_edge_rules.clear()
        self.violated_edge_rules.clear()


@attr.s(frozen=True)
class _SolutionContainer:
    """Defines a result of a `.ProblemSet`."""

    solutions: List[StateTransitionGraph[ParticleWithSpin]] = attr.ib(
        factory=list
    )
    execution_info: ExecutionInfo = attr.ib(ExecutionInfo())

    def __attrs_post_init__(self) -> None:
        if self.solutions and (
            self.execution_info.violated_node_rules
            or self.execution_info.violated_edge_rules
        ):
            raise ValueError(
                f"Invalid {self.__class__.__name__}!"
                f" Found {len(self.solutions)} solutions, but also violated rules.",
                self.execution_info.violated_node_rules,
                self.execution_info.violated_edge_rules,
            )

    def extend(
        self, other: "_SolutionContainer", intersect_violations: bool = False
    ) -> None:
        if self.solutions or other.solutions:
            self.solutions.extend(other.solutions)
            self.execution_info.clear()
        else:
            self.execution_info.extend(
                other.execution_info, intersect_violations
            )


@attr.s(on_setattr=attr.setters.frozen)
class Result:
    transitions: List[StateTransitionGraph[ParticleWithSpin]] = attr.ib(
        factory=list
    )
    formalism_type: Optional[str] = attr.ib(default=None)

    def get_initial_state(self) -> List[Particle]:
        graph = self.__get_first_graph()
        return [
            x[0]
            for x in map(
                graph.get_edge_props, graph.topology.incoming_edge_ids
            )
            if x
        ]

    def get_final_state(self) -> List[Particle]:
        graph = self.__get_first_graph()
        return [
            x[0]
            for x in map(
                graph.get_edge_props, graph.topology.outgoing_edge_ids
            )
            if x
        ]

    def __get_first_graph(self) -> StateTransitionGraph[ParticleWithSpin]:
        if len(self.transitions) == 0:
            raise ValueError(
                f"No solutions in {self.__class__.__name__} object"
            )
        return next(iter(self.transitions))

    def get_intermediate_particles(self) -> ParticleCollection:
        """Extract the names of the intermediate state particles."""
        intermediate_states = ParticleCollection()
        for transition in self.transitions:
            for edge_props in map(
                transition.get_edge_props,
                transition.topology.intermediate_edge_ids,
            ):
                if edge_props:
                    particle, _ = edge_props
                    if particle not in intermediate_states:
                        intermediate_states.add(particle)
        return intermediate_states


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

    def to_qn_problem_set(self) -> QNProblemSet:
        node_props = {
            k: create_node_properties(v)
            for k, v in self.initial_facts.node_props.items()
        }
        edge_props = {
            k: create_edge_properties(v[0], v[1])
            for k, v in self.initial_facts.edge_props.items()
        }
        return QNProblemSet(
            topology=self.topology,
            initial_facts=GraphElementProperties(
                node_props=node_props, edge_props=edge_props
            ),
            solving_settings=self.solving_settings,
        )


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

    .. seealso:: :doc:`/usage/reaction` and `.reaction.generate`
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
        solving_mode: SolvingMode = SolvingMode.FAST,
        reload_pdg: bool = False,
        mass_conservation_factor: Optional[float] = 3.0,
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
            InteractionTypes.STRONG,
            InteractionTypes.EM,
            InteractionTypes.WEAK,
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
        use_nbody_topology = False
        topology_building = topology_building.lower()
        if topology_building == "isobar":
            self.__topologies = create_isobar_topologies(len(final_state))
        elif "n-body" in topology_building or "nbody" in topology_building:
            self.__topologies = (
                create_n_body_topology(len(initial_state), len(final_state)),
            )
            use_nbody_topology = True
            # turn of mass conservation, in case more than one initial state
            # particle is present
            if len(initial_state) > 1:
                mass_conservation_factor = None

        if not self.interaction_type_settings:
            self.interaction_type_settings = (
                create_default_interaction_settings(
                    formalism_type,
                    nbody_topology=use_nbody_topology,
                    mass_conservation_factor=mass_conservation_factor,
                )
            )

        if reload_pdg or len(self.__particles) == 0:
            self.__particles = load_pdg()

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
        problem_sets = []
        for topology in self.__topologies:
            for initial_facts in create_initial_facts(
                topology=topology,
                particles=self.__particles,
                initial_state=self.initial_state,
                final_state=self.final_state,
                final_state_groupings=self.final_state_groupings,
            ):
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
                    self.interaction_type_settings[InteractionTypes.WEAK][
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

            return self.interaction_type_settings[InteractionTypes.WEAK][
                0
            ].qn_domains

        intermediate_state_edges = topology.intermediate_edge_ids
        int_edge_domains = create_intermediate_edge_qn_domains()

        def create_edge_settings(edge_id: int) -> EdgeSettings:
            settings = copy(
                self.interaction_type_settings[InteractionTypes.WEAK][0]
            )
            if edge_id in intermediate_state_edges:
                settings.qn_domains = int_edge_domains
            else:
                settings.qn_domains = {}
            return settings

        final_state_edges = topology.outgoing_edge_ids
        initial_state_edges = topology.incoming_edge_ids

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

    def find_solutions(  # pylint: disable=too-many-branches
        self,
        problem_sets: Dict[float, List[ProblemSet]],
    ) -> Result:
        # pylint: disable=too-many-locals
        """Check for solutions for a specific set of interaction settings."""
        results: Dict[float, _SolutionContainer] = {}
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

            qn_problems = [x.to_qn_problem_set() for x in problems]

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
                and self.reaction_mode == SolvingMode.FAST
            ):
                break
        progress_bar.close()

        for key, result in results.items():
            logging.info(
                f"number of solutions for strength ({key}) "
                f"after qn solving: {len(result.solutions)}",
            )

        final_result = _SolutionContainer()
        for temp_result in results.values():
            final_result.extend(temp_result)

        # remove duplicate solutions, which only differ in the interaction qns
        final_solutions = remove_duplicate_solutions(
            final_result.solutions,
            self.filter_remove_qns,
            self.filter_ignore_qns,
        )

        if (
            final_result.execution_info.violated_edge_rules
            or final_result.execution_info.violated_node_rules
        ):
            execution_info = final_result.execution_info
            violated_rules: Set[str] = set()
            for rules in execution_info.violated_edge_rules.values():
                violated_rules |= rules
            for rules in execution_info.violated_node_rules.values():
                violated_rules |= rules
            if violated_rules:
                raise RuntimeError(
                    "There were violated conservation rules: "
                    + ", ".join(violated_rules)
                )
        if (
            final_result.execution_info.not_executed_edge_rules
            or final_result.execution_info.not_executed_node_rules
        ):
            not_executed_rules: Set[str] = set()
            for rules in execution_info.not_executed_edge_rules.values():
                not_executed_rules |= rules
            for rules in execution_info.not_executed_node_rules.values():
                not_executed_rules |= rules
            raise RuntimeWarning(
                "There are conservation rules that were not executed: "
                + ", ".join(not_executed_rules)
            )
        if not final_solutions:
            raise ValueError("No solutions were found")

        match_external_edges(final_solutions)
        return Result(
            final_solutions,
            formalism_type=self.formalism_type,
        )

    def _solve(
        self, qn_problem_set: QNProblemSet
    ) -> Tuple[QNProblemSet, QNResult]:
        solver = CSPSolver(self.__allowed_intermediate_particles)

        return (qn_problem_set, solver.find_solutions(qn_problem_set))

    def __convert_result(
        self, topology: Topology, qn_result: QNResult
    ) -> _SolutionContainer:
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
            solutions.append(graph)

        return _SolutionContainer(
            solutions,
            ExecutionInfo(
                violated_edge_rules=qn_result.violated_edge_rules,
                violated_node_rules=qn_result.violated_node_rules,
                not_executed_node_rules=qn_result.not_executed_node_rules,
                not_executed_edge_rules=qn_result.not_executed_edge_rules,
            ),
        )
