# cspell:ignore pksigma

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, SupportsFloat

import attrs
import pytest
import sympy as sp
from qrules import InteractionType, StateTransitionManager
from qrules.topology import FrozenTransition, create_isobar_topologies

from ampform.adapter.qrules import (
    _convert_transition,
    _get_equal_final_state_ids,
    _prepare_transitions,
    _to_decay_chain,
    convert_transitions,
    filter_min_ls,
    normalize_state_ids,
    permute_equal_final_states,
    to_decay,
    to_three_body_decay,
)
from ampform.amplitude.helicity.naming import get_boost_chain_suffix
from ampform.decay import (
    DecayChain,
    IsobarNode,
    LSCoupling,
    Particle,
    generate_helicity_assignments,
)
from ampform.kinematics.angles import compute_helicity_angles
from ampform.kinematics.lorentz import (
    compute_invariant_masses,
    create_four_momentum_symbols,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from qrules.transition import ReactionInfo, StateTransition


def test_convert_transitions(xib2pkk_reaction: ReactionInfo):
    reaction = normalize_state_ids(xib2pkk_reaction)
    assert reaction.get_intermediate_particles().names == ["Lambda(1520)"]
    assert len(reaction.transitions) == 16
    transitions = convert_transitions(reaction.transitions)
    assert len(transitions) == 2
    decay = to_three_body_decay(transitions, min_ls=True)
    assert len(decay.chains) == 2


def test_filter_min_ls(jpsi2pksigma_reaction: ReactionInfo):
    reaction = jpsi2pksigma_reaction
    transitions = tuple(
        t for t in reaction.transitions if t.states[3].spin_projection == +0.5
    )

    ls_couplings = _group_couplings(transitions)
    if reaction.formalism == "canonical-helicity":
        assert ls_couplings == {
            "N(1700)+": [
                ({"L": 1, "S": 2}, {"L": 2, "S": 0.5}),
            ],
            "Sigma(1660)~-": [
                ({"L": 0, "S": 1}, {"L": 1, "S": 0.5}),
                ({"L": 2, "S": 1}, {"L": 1, "S": 0.5}),
            ],
        }
    else:
        assert len(ls_couplings) == 2
        assert ls_couplings == {
            "N(1700)+": [({"L": None, "S": None}, {"L": None, "S": None})],
            "Sigma(1660)~-": [({"L": None, "S": None}, {"L": None, "S": None})],
        }

    if reaction.formalism != "canonical-helicity":
        return

    min_ls_transitions = filter_min_ls(transitions)
    ls_couplings = _group_couplings(min_ls_transitions)
    assert ls_couplings == {
        "N(1700)+": [
            ({"L": 1, "S": 2}, {"L": 2, "S": 0.5}),
        ],
        "Sigma(1660)~-": [
            ({"L": 0, "S": 1}, {"L": 1, "S": 0.5}),
        ],
    }

    min_ls_transitions = filter_min_ls(transitions, node_ids={0})
    ls_couplings = _group_couplings(min_ls_transitions)
    assert ls_couplings == {
        "N(1700)+": [
            ({"L": 1, "S": 2}, {"L": 2, "S": 0.5}),
        ],
        "Sigma(1660)~-": [
            ({"L": 0, "S": 1}, {"L": 1, "S": 0.5}),
        ],
    }

    min_ls_transitions = filter_min_ls(transitions, node_ids={1})
    ls_couplings = _group_couplings(min_ls_transitions)
    assert ls_couplings == {
        "N(1700)+": [
            ({"L": 1, "S": 2}, {"L": 2, "S": 0.5}),
        ],
        "Sigma(1660)~-": [
            ({"L": 0, "S": 1}, {"L": 1, "S": 0.5}),
            ({"L": 2, "S": 1}, {"L": 1, "S": 0.5}),
        ],
    }


@pytest.mark.parametrize("converter", [lambda x: x, _convert_transition])
def test_get_equal_final_state_ids(
    a2pipipi_reaction: ReactionInfo,
    jpsi2pksigma_reaction: ReactionInfo,
    xib2pkk_reaction: ReactionInfo,
    converter: Callable[[FrozenTransition], FrozenTransition],
):
    test_cases = [
        (a2pipipi_reaction, (1, 2, 3)),
        (jpsi2pksigma_reaction, tuple()),
        (xib2pkk_reaction, (2, 3)),
    ]
    for reaction012, expected in test_cases:
        reaction = normalize_state_ids(reaction012)
        transition = converter(reaction.transitions[0])
        equal_ids = _get_equal_final_state_ids(transition)
        assert equal_ids == expected


def test_normalize_state_ids_reaction(jpsi2pksigma_reaction: ReactionInfo):
    reaction012 = jpsi2pksigma_reaction
    reaction123 = normalize_state_ids(reaction012)
    assert set(reaction123.initial_state) == {0}
    assert set(reaction123.final_state) == {1, 2, 3}

    transitions123 = normalize_state_ids(reaction012.transitions)
    for transition012, transition123 in zip(
        reaction012.transitions, transitions123, strict=True
    ):
        assert set(transition123.initial_states) == {0}
        assert set(transition123.final_states) == {1, 2, 3}
        assert set(transition123.intermediate_states) == {4}

        topology123 = normalize_state_ids(transition123.topology)
        assert topology123.incoming_edge_ids == {0}
        assert topology123.outgoing_edge_ids == {1, 2, 3}
        assert topology123.intermediate_edge_ids == {4}

        for i in transition012.states:
            assert transition012.states[i] == transition123.states[i + 1]


def test_normalize_state_ids_problem_set():
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)", [-1, +1])],
        final_state=["K0", "Sigma+", "p~"],
        allowed_intermediate_particles=["N(1700)", "Sigma(1750)"],
        formalism="helicity",
        mass_conservation_factor=0,
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG, InteractionType.EM])
    problem_sets = stm.create_problem_sets()
    some_problem_set = normalize_state_ids(problem_sets[3600.0][0])
    assert set(some_problem_set.initial_facts.initial_states) == {0}
    assert set(some_problem_set.initial_facts.final_states) == {1, 2, 3}


def test_permute_equal_final_states(
    a2pipipi_reaction: ReactionInfo,
    jpsi2pksigma_reaction: ReactionInfo,
    xib2pkk_reaction: ReactionInfo,
):
    test_cases = [
        (1, jpsi2pksigma_reaction),
        (2, xib2pkk_reaction),
        (3, a2pipipi_reaction),
    ]
    for n_permutations, reaction012 in test_cases:
        reaction = normalize_state_ids(reaction012)
        transition = reaction.transitions[0]
        permutations = permute_equal_final_states(transition)
        assert len(permutations) == n_permutations

        permuted_reaction = permute_equal_final_states(reaction)
        n_transitions = len(permuted_reaction.transitions)
        assert n_transitions == n_permutations * len(reaction.transitions)


def test_generate_helicity_assignments_matches_qrules(
    jpsi2pksigma_reaction: ReactionInfo, xib2pkk_reaction: ReactionInfo
):
    """Helicity enumeration from spin magnitudes reproduces QRules transitions.

    The projection restrictions mirror the ones imposed on the reaction fixtures, which
    are user input to QRules and not derivable from the decay structure.
    """
    half = sp.Rational(1, 2)
    test_cases = [
        (jpsi2pksigma_reaction, {(1, 2, 3): {1}, (2,): {half}, (3,): {half}}),
        (xib2pkk_reaction, {}),
    ]
    for reaction012, restrictions in test_cases:
        reaction = normalize_state_ids(reaction012)
        expected: defaultdict[DecayChain, set] = defaultdict(set)
        for transition in reaction.transitions:
            chain = _to_decay_chain(_convert_transition(transition), DecayChain)
            assignment = tuple(
                sorted(
                    (
                        _get_edge_key(transition.topology, i),
                        sp.Rational(s.spin_projection),
                    )
                    for i, s in transition.states.items()
                )
            )
            expected[chain].add(assignment)
        assert len(expected) > 0
        for chain, expected_assignments in expected.items():
            computed = {
                tuple(sorted(assignment.items()))
                for assignment in generate_helicity_assignments(chain)
                if all(
                    assignment[key] in allowed for key, allowed in restrictions.items()
                )
            }
            assert computed == expected_assignments


def test_kinematics_functions_match_topology_implementation(
    jpsi2pksigma_reaction: ReactionInfo,
):
    """Kinematics of a `.DecayChain` match those of a `~qrules.topology.Topology`."""
    transitions = [
        *_prepare_transitions(jpsi2pksigma_reaction.transitions),
        *_prepare_transitions(_create_four_body_transitions()),
    ]
    for transition in transitions:
        topology = transition.topology
        chain = to_decay([transition]).chains[0]
        momenta = create_four_momentum_symbols(topology)
        assert create_four_momentum_symbols(chain) == momenta
        assert compute_helicity_angles(momenta, chain) == compute_helicity_angles(
            momenta, topology
        )
        assert compute_invariant_masses(momenta, chain) == compute_invariant_masses(
            momenta, topology
        )
        for state_id in topology.edges:
            edge = _get_edge_key(topology, state_id)
            assert get_boost_chain_suffix(chain, edge) == get_boost_chain_suffix(
                topology, state_id
            )


def _get_edge_key(topology, state_id: int) -> tuple[int, ...]:
    edge = topology.edges[state_id]
    if edge.ending_node_id is None:
        return (state_id,)
    return tuple(
        sorted(topology.get_originating_final_state_edge_ids(edge.ending_node_id))
    )


def _create_four_body_transitions() -> list[FrozenTransition]:
    """Create dummy J/ψ → 4π transitions for each four-body isobar topology."""
    dummy_args = dict(mass=0.0, width=0.0)
    jpsi = Particle("J/psi(1S)", latex=R"J/\psi(1S)", spin=1, parity=-1, **dummy_args)
    π = Particle("pi0", latex=R"\pi^0", spin=0, parity=-1, **dummy_args)
    ω = Particle("omega(782)", latex=R"\omega(782)", spin=1, parity=-1, **dummy_args)
    f0 = Particle("f(0)(980)", latex=R"f_0(980)", spin=0, parity=+1, **dummy_args)
    transitions = []
    for topology in create_isobar_topologies(4):
        resonances = dict(
            zip(sorted(topology.intermediate_edge_ids), [ω, f0], strict=True)
        )
        transitions.append(
            FrozenTransition(
                topology,
                states={
                    **dict.fromkeys(topology.incoming_edge_ids, jpsi),
                    **dict.fromkeys(topology.outgoing_edge_ids, π),
                    **resonances,
                },
                interactions=dict.fromkeys(topology.nodes),
            )
        )
    return transitions


def test_to_decay_four_body():
    decay = to_decay(_create_four_body_transitions())
    assert decay.initial_state.name == "J/psi(1S)"
    assert list(decay.final_state) == [1, 2, 3, 4]
    assert len(decay.chains) == 2
    tree_shapes = set()
    for chain in decay.chains:
        assert len(chain.nodes) == 3
        assert {p.name for p in chain.resonances} == {"omega(782)", "f(0)(980)"}
        tree_shapes.add(tuple(isinstance(c, IsobarNode) for c in chain.decay.children))
    assert tree_shapes == {(True, False), (True, True)}


def test_to_decay_min_ls_four_body():
    transition = _create_four_body_transitions()[0]
    transitions = [
        attrs.evolve(
            transition,
            interactions={
                node_id: LSCoupling(L=L if node_id == 0 else 1, S=1)
                for node_id in transition.topology.nodes
            },
        )
        for L in [1, 3]
    ]
    decay = to_decay(transitions, min_ls=True)
    assert len(decay.chains) == 1
    chain = decay.chains[0]
    ls_values = sorted(
        (node.interaction.L, node.interaction.S)
        for node in chain.nodes
        if node.interaction is not None
    )
    assert ls_values == [(1, 1), (1, 1), (1, 1)]


def test_to_decay_matches_to_three_body_decay(jpsi2pksigma_reaction: ReactionInfo):
    reaction = normalize_state_ids(jpsi2pksigma_reaction)
    decay = to_decay(reaction.transitions)
    three_body_decay = to_three_body_decay(reaction.transitions)
    assert decay.states == three_body_decay.states
    assert [c.decay for c in decay.chains] == [c.decay for c in three_body_decay.chains]


@pytest.mark.parametrize("min_ls", [False, True])
def test_to_three_body_decay(jpsi2pksigma_reaction: ReactionInfo, min_ls: bool):
    reaction = normalize_state_ids(jpsi2pksigma_reaction)
    decay = to_three_body_decay(reaction.transitions, min_ls)
    assert decay.initial_state.name == "J/psi(1S)"
    assert {i: p.name for i, p in decay.final_state.items()} == {
        1: "K0",
        2: "Sigma+",
        3: "p~",
    }
    n_chains = len(decay.chains)
    if reaction.formalism == "canonical-helicity":
        production_ls = [c.incoming_ls for c in decay.chains]
        decay_ls = [c.outgoing_ls for c in decay.chains]
        if min_ls:
            assert n_chains == 2
            assert production_ls == [
                LSCoupling(L=1, S=1),
                LSCoupling(L=0, S=1),
            ]
            assert decay_ls == [
                LSCoupling(L=2, S=0.5),
                LSCoupling(L=1, S=0.5),
            ]
        else:
            assert n_chains == 4
            assert production_ls == [
                LSCoupling(L=1, S=1),
                LSCoupling(L=1, S=2),
                LSCoupling(L=0, S=1),
                LSCoupling(L=2, S=1),
            ]
            assert decay_ls == [
                LSCoupling(L=2, S=0.5),
                LSCoupling(L=2, S=0.5),
                LSCoupling(L=1, S=0.5),
                LSCoupling(L=1, S=0.5),
            ]
    elif reaction.formalism == "helicity":
        assert n_chains == 2
        for chain in decay.chains:
            assert chain.incoming_ls is None
            assert chain.outgoing_ls is None
    resonance_names = set()
    for chain in decay.chains:
        assert isinstance(chain.resonance, Particle)
        resonance_names.add(chain.resonance.name)
    assert resonance_names == {
        "N(1700)+",
        "Sigma(1660)~-",
    }


def _group_couplings(
    transitions: Iterable[StateTransition],
) -> dict[str, tuple[dict, dict]]:
    groupings = defaultdict(list)
    for transition in transitions:
        resonance, *_ = transition.intermediate_states.values()
        ls_values = _get_couplings(transition)
        groupings[resonance.particle.name].append(ls_values)
    return dict(groupings)


def _get_couplings(transition: StateTransition) -> tuple[dict, dict]:
    if len(transition.interactions) != 2:
        msg = "Expected exactly two interaction nodes"
        raise ValueError(msg)
    return tuple(
        {"L": _to_float(node.l_magnitude), "S": _to_float(node.s_magnitude)}
        for node in transition.interactions.values()
    )  # ty:ignore[invalid-return-type]


def _to_float(value: SupportsFloat | None) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    if value.is_integer():
        return int(value)
    return value
