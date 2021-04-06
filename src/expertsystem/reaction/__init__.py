"""Definition and solving of particle reaction problems.

This is the core component of the `expertsystem`: it defines the
`.StateTransitionGraph` data structure that represents a specific particle
reaction. The `solving` submodule is responsible for finding solutions for
particle reaction problems.
"""

from itertools import product
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Union

import attr

from .combinatorics import InitialFacts, StateDefinition, create_initial_facts
from .conservation_rules import (
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
from .default_settings import InteractionTypes
from .particle import ParticleCollection, load_pdg
from .quantum_numbers import InteractionProperties
from .solving import (
    GraphSettings,
    NodeSettings,
    QNResult,
    Rule,
    validate_full_solution,
)
from .topology import create_n_body_topology
from .transition import (
    EdgeSettings,
    ProblemSet,
    Result,
    StateTransitionManager,
)


def check_reaction_violations(
    initial_state: Union[StateDefinition, Sequence[StateDefinition]],
    final_state: Sequence[StateDefinition],
    mass_conservation_factor: Optional[float] = 3.0,
) -> Set[FrozenSet[str]]:
    """Determine violated interaction rules for a given particle reaction.

    .. warning:: This function only guarantees to find P, C and G parity
      violations, if it's a two body decay. If all initial and final states
      have the C/G parity defined, then these violations are also determined
      correctly.

    Args:
      initial_state: Shortform description of the initial state w/o spin
        projections.
      final_state: Shortform description of the final state w/o spin
        projections.
      mass_conservation_factor: Factor with which the width is multiplied when
        checking for `.MassConservation`. Set to `None` in order to deactivate
        mass conservation.

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
        problem_set = ProblemSet(
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
        return validate_full_solution(problem_set.to_qn_problem_set())

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
                for edge_id in topology.incoming_edge_ids
                | topology.outgoing_edge_ids
            },
        )

        if edge_check_result.violated_edge_rules:
            raise ValueError(
                f"Some edges violate"
                f" {edge_check_result.violated_edge_rules.values()}"
            )

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
        if len(initial_state) == 1 and mass_conservation_factor is not None:
            edge_qn_conservation_rules.add(
                MassConservation(mass_conservation_factor)
            )

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
    topology = create_n_body_topology(len(initial_state), len(final_state))
    node_id = next(iter(topology.nodes))

    initial_facts = create_initial_facts(
        topology=topology,
        particles=load_pdg(),
        initial_state=initial_state,
        final_state=final_state,
    )

    check_pure_edge_rules()
    violations = check_edge_qn_conservation()

    # Create combinations of graphs for magnitudes of S and L, but only
    # if it is a two body reaction
    ls_combinations = [
        InteractionProperties(l_magnitude=l_magnitude, s_magnitude=s_magnitude)
        for l_magnitude, s_magnitude in product([0, 1], [0, 0.5, 1, 1.5, 2])
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


def generate(  # pylint: disable=too-many-arguments
    initial_state: Union[StateDefinition, Sequence[StateDefinition]],
    final_state: Sequence[StateDefinition],
    allowed_intermediate_particles: Optional[List[str]] = None,
    allowed_interaction_types: Optional[Union[str, List[str]]] = None,
    formalism_type: str = "helicity",
    particles: Optional[ParticleCollection] = None,
    mass_conservation_factor: Optional[float] = 3.0,
    topology_building: str = "isobar",
    number_of_threads: Optional[int] = None,
) -> Result:
    """Generate allowed transitions between an initial and final state.

    Serves as a facade to the `.StateTransitionManager` (see
    :doc:`/usage/reaction`).

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
            (1) filter out resonances in the eventual `.HelicityModel` and (2)
            speed up computation time.

        allowed_interaction_types (`str`, optional): Interaction types you want
            to consider. For instance, both :code:`"strong and EM"` and
            :code:`["s", "em"]` results in `~.InteractionTypes.EM` and
            `~.InteractionTypes.STRONG`.

        formalism_type (`str`, optional): Formalism that you intend to use in the
            eventual `.HelicityModel`.

        particles (`.ParticleCollection`, optional): The particles that you
            want to be involved in the reaction. Uses `.load_pdg` by default.
            It's better to use a subset for larger reactions, because of
            the computation times. This argument is especially useful when you
            want to use your own particle definitions (see
            :doc:`/usage/particle`).

        mass_conservation_factor: Width factor that is taken into account for
            for the `.MassConservation` rule.

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
    ...     particles=es.reaction.load_pdg(),
    ...     topology_building="isobar",
    ... )
    >>> len(result.transitions)
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
        mass_conservation_factor=mass_conservation_factor,
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
        interaction_types.add(InteractionTypes.WEAK)
    if "strong" in interaction_name_lower or interaction_name_lower == "s":
        interaction_types.add(InteractionTypes.STRONG)
    if len(interaction_types) == 0:
        raise ValueError(
            f'Could not determine interaction type from "{description}"'
        )
    return interaction_types
