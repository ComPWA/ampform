"""Convert `qrules` objects to decay structures from the :mod:`ampform.decay` module."""

from __future__ import annotations

import itertools
import logging
from collections import abc, defaultdict
from functools import singledispatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar, overload

import attrs
import qrules

from ampform._qrules import get_qrules_version

if get_qrules_version() < (0, 10):
    msg = f"The {__name__} module requires qrules v0.10 or higher"
    raise ImportError(msg)

from qrules.quantum_numbers import InteractionProperties
from qrules.topology import (
    EdgeType,
    FrozenTransition,
    MutableTransition,
    NodeType,
    Transition,
)
from qrules.transition import ProblemSet, ReactionInfo, Topology

from ampform.decay import (
    ChainType,
    Decay,
    DecayChain,
    FinalStateID,
    IsobarNode,
    LSCoupling,
    Particle,
    State,
    StateIDTemplate,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

_LOGGER = logging.getLogger(__name__)


def to_decay(
    transitions: Iterable[FrozenTransition],
    min_ls: bool = False,
) -> Decay[DecayChain]:
    """Convert isobar transitions of any final-state size to a `~ampform.decay.Decay`."""
    transitions = _prepare_transitions(transitions)
    if min_ls:
        transitions = filter_min_ls(transitions)
    some_transition = transitions[0]
    return Decay(
        states=_get_outer_states(some_transition),
        chains=_sort_chains(_to_decay_chain(t, DecayChain) for t in transitions),
    )


def to_three_body_decay(
    transitions: Iterable[FrozenTransition],
    min_ls: bool | tuple[bool, bool] = False,
) -> ThreeBodyDecay:
    transitions = _prepare_transitions(transitions)
    if min_ls:
        if isinstance(min_ls, bool):
            node_ids = None
        else:
            production_min_ls, decay_min_ls = min_ls
            node_ids = set()
            if production_min_ls:
                node_ids.add(0)
            if decay_min_ls:
                node_ids.add(1)
        transitions = filter_min_ls(transitions, node_ids)
    some_transition = transitions[0]
    return ThreeBodyDecay(
        states=_get_outer_states(some_transition),
        chains=_sort_chains(
            _to_decay_chain(t, ThreeBodyDecayChain) for t in transitions
        ),
    )


def _prepare_transitions(
    transitions: Iterable[FrozenTransition],
) -> tuple[FrozenTransition[Particle | State, LSCoupling | None], ...]:
    transitions = tuple(transitions)
    if not transitions:
        msg = "Need at least one transition object"
        raise ValueError(msg)
    some_transition = transitions[0]
    expected_final_state_ids = set(range(1, len(some_transition.final_states) + 1))
    if (
        set(some_transition.initial_states) != {0}
        or set(some_transition.final_states) != expected_final_state_ids
    ):
        transitions = normalize_state_ids(transitions)
        final_state_ids = ", ".join(str(i) for i in sorted(expected_final_state_ids))
        _LOGGER.warning(
            f"Relabeled initial state to 0 and final states to {final_state_ids}"
        )
    return convert_transitions(transitions)


def _get_outer_states(
    transition: FrozenTransition[Particle | State, LSCoupling | None],
) -> dict[int, State]:
    (initial_state_id, initial_state), *_ = transition.initial_states.items()
    outer_states = (
        _to_state(initial_state, index=initial_state_id),
        *[
            _to_state(particle, index=idx)
            for idx, particle in transition.final_states.items()
        ],
    )
    return {state.index: state for state in outer_states}


def _sort_chains(chains: Iterable[ChainType]) -> tuple[ChainType, ...]:
    """Sort decay chains, first grouping them by tree shape.

    Decay chains with the same tree shape are comparable through their `attrs.field`
    order, whereas comparing trees of different shapes would compare an `.IsobarNode`
    with a `.State`.
    """
    groups: defaultdict[str, list[ChainType]] = defaultdict(list)
    for chain in chains:
        groups[_get_tree_signature(chain.decay)].append(chain)
    return tuple(
        chain for signature in sorted(groups) for chain in sorted(groups[signature])
    )


def _get_tree_signature(node: IsobarNode | State) -> str:
    if isinstance(node, IsobarNode):
        child_signatures = ",".join(_get_tree_signature(c) for c in node.children)
        return f"({child_signatures})"
    return str(node.index)


def _to_decay_chain(
    transition: FrozenTransition[Particle | State, LSCoupling | None],
    chain_type: type[ChainType],
) -> ChainType:
    if len(transition.initial_states) != 1:
        msg = f"Can only handle one initial state, but got {len(transition.initial_states)}"
        raise ValueError(msg)
    initial_state_id, *_ = transition.initial_states
    root_node_id = transition.topology.edges[initial_state_id].ending_node_id
    if root_node_id is None:
        msg = f"Initial state {initial_state_id} does not decay"
        raise ValueError(msg)
    return chain_type(decay=_to_isobar_node(transition, root_node_id))


def _to_isobar_node(
    transition: FrozenTransition[Particle | State, LSCoupling | None],
    node_id: int,
) -> IsobarNode:
    topology = transition.topology
    parent_id, *_ = topology.get_edge_ids_ingoing_to_node(node_id)
    child_ids = sorted(
        topology.get_edge_ids_outgoing_from_node(node_id),
        key=lambda i: (topology.edges[i].ending_node_id is None, i),
    )
    if len(child_ids) != 2:  # ruff: ignore[magic-value-comparison]
        msg = (
            f"Node {node_id} decays to {len(child_ids)} states, so this is not an"
            " isobar decay"
        )
        raise ValueError(msg)
    return IsobarNode(
        parent=transition.states[parent_id],
        child1=_to_isobar_child(transition, child_ids[0]),
        child2=_to_isobar_child(transition, child_ids[1]),
        interaction=transition.interactions[node_id],
    )


def _to_isobar_child(
    transition: FrozenTransition[Particle | State, LSCoupling | None],
    state_id: int,
) -> IsobarNode | Particle | State:
    node_id = transition.topology.edges[state_id].ending_node_id
    if node_id is None:
        return transition.states[state_id]
    return _to_isobar_node(transition, node_id)


def convert_transitions(
    transitions: Iterable[FrozenTransition],
) -> tuple[FrozenTransition[Particle | State, LSCoupling | None], ...]:
    unique_transitions = {_convert_transition(t) for t in transitions}
    return tuple(sorted(unique_transitions))


def _convert_transition(
    transition: FrozenTransition,
) -> FrozenTransition[Particle | State, LSCoupling | None]:
    return FrozenTransition(
        transition.topology,
        states={
            index: _to_particle(state)
            if index in transition.intermediate_states
            else _to_state(state, index=index)
            for index, state in transition.states.items()
        },
        interactions={
            i: _to_ls_coupling(interaction)
            for i, interaction in transition.interactions.items()
        },
    )


def _to_particle(
    particle: qrules.particle.Particle | qrules.transition.State,
) -> Particle:
    if isinstance(particle, qrules.transition.State):
        particle = particle.particle
    return Particle(
        name=particle.name,
        latex=particle.name if particle.latex is None else particle.latex,
        spin=particle.spin,
        parity=int(particle.parity),
        mass=particle.mass,
        width=particle.width,
    )


def _to_state(obj: Any, index: StateIDTemplate | None = None):
    if isinstance(obj, qrules.transition.State):
        obj = obj.particle
    if isinstance(obj, State):
        index = obj.index
    if index is None:
        msg = f"Cannot create a {State} from a {type(obj)} without an index"
        raise ValueError(msg)
    if not isinstance(obj, Particle) and not isinstance(obj, qrules.particle.Particle):
        msg = f"Cannot convert object of type {type(obj)} to a {State}"
        raise NotImplementedError(msg)
    return State(
        name=obj.name,
        latex=obj.name if obj.latex is None else obj.latex,
        spin=obj.spin,
        parity=int(obj.parity),
        mass=obj.mass,
        width=obj.width,
        index=index,
    )


def _to_ls_coupling(node: Any) -> LSCoupling | None:
    if node is None:
        return None
    if isinstance(node, LSCoupling):
        return node
    if not isinstance(node, InteractionProperties):
        msg = f"Cannot convert node of type {type(node)}"
        raise NotImplementedError(msg)
    if node.l_magnitude is None or node.s_magnitude is None:
        return None
    return LSCoupling(
        L=node.l_magnitude,
        S=node.s_magnitude,
    )


def filter_min_ls(
    transitions: Iterable[FrozenTransition[EdgeType, NodeType]],
    node_ids: set[int] | None = None,
) -> tuple[FrozenTransition[EdgeType, NodeType], ...]:
    """Select the transitions with the lowest LS-couplings at the given nodes.

    Transitions are grouped by topology and intermediate states, ignoring the
    interactions at the selected :code:`node_ids` (default: all nodes), and each group
    is collapsed into a single transition with the lowest LS-coupling per node.
    """
    transitions = tuple(transitions)
    if node_ids is None:
        node_ids = {i for t in transitions for i in t.interactions}
    grouped_transitions: defaultdict[
        tuple[Topology, tuple[tuple[int, EdgeType], ...], tuple[NodeType, ...]],
        list[FrozenTransition[EdgeType, NodeType]],
    ] = defaultdict(list)
    for transition in transitions:
        key = (
            transition.topology,
            tuple(transition.intermediate_states.items()),
            tuple(
                node
                for i, node in transition.interactions.items()
                if node and i not in node_ids
            ),
        )
        grouped_transitions[key].append(transition)
    min_transitions: list[FrozenTransition[EdgeType, NodeType]] = []
    for group in grouped_transitions.values():
        transition0, *_ = group
        min_transition: FrozenTransition[EdgeType, NodeType] = FrozenTransition(
            topology=transition0.topology,
            states=transition0.states,
            interactions={
                i: None
                if any(t.interactions[i] is None for t in group)
                else (
                    min(t.interactions[i] for t in group)
                    if i in node_ids
                    else transition0.interactions[i]
                )
                for i in transition0.interactions
            },
        )
        min_transitions.append(min_transition)
    return tuple(min_transitions)


def load_particles() -> qrules.particle.ParticleCollection:
    src_dir = Path(__file__).parent.parent
    particle_database = qrules.load_default_particles()
    additional_definitions = qrules.io.load(src_dir / "particle-definitions.yml")
    particle_database.update(additional_definitions)
    return particle_database


@overload
def normalize_state_ids(obj: T) -> T: ...
@overload
def normalize_state_ids(obj: Iterable[T]) -> list[T]: ...
def normalize_state_ids(obj):
    """Relabel the state IDs so that they lie in the range :math:`[0, N)`."""
    return _impl_normalize_state_ids(obj)


@singledispatch
def _impl_normalize_state_ids(obj) -> NoReturn:
    """Relabel the state IDs so that they lie in the range :math:`[0, N)`."""
    msg = f"Cannot relabel edge IDs of a {type(obj).__name__}"
    raise NotImplementedError(msg)


@_impl_normalize_state_ids.register(ReactionInfo)
def _(obj: ReactionInfo) -> ReactionInfo:
    return ReactionInfo(
        # no attrs.evolve() in order to call __attrs_post_init__()
        transitions=[_impl_normalize_state_ids(g) for g in obj.transitions],
        formalism=obj.formalism,
    )


_Transition = TypeVar("_Transition", FrozenTransition, MutableTransition)


@_impl_normalize_state_ids.register(FrozenTransition)
@_impl_normalize_state_ids.register(MutableTransition)
def _(obj: _Transition) -> _Transition:
    return attrs.evolve(
        obj,
        topology=_impl_normalize_state_ids(obj.topology),
        states={new: obj.states[old] for new, old in enumerate(sorted(obj.states))},
    )


@_impl_normalize_state_ids.register(ProblemSet)
def _(obj: ProblemSet) -> ProblemSet:
    return ProblemSet(
        initial_facts=_impl_normalize_state_ids(obj.initial_facts),
        solving_settings=_impl_normalize_state_ids(obj.solving_settings),
        topology=_impl_normalize_state_ids(obj.topology),
    )


@_impl_normalize_state_ids.register(Topology)
def _(obj: Topology) -> Topology:
    mapping = {old: new for new, old in enumerate(sorted(obj.edges))}
    return obj.relabel_edges(mapping)


@_impl_normalize_state_ids.register(abc.Iterable)
def _(obj: abc.Iterable[T]) -> list[T]:
    return [_impl_normalize_state_ids(x) for x in obj]


T = TypeVar(
    "T",
    FrozenTransition,
    MutableTransition,
    ProblemSet,
    ReactionInfo,
    Topology,
    Transition,
)
"""Type variable for the input and output of :func:`normalize_state_ids`."""


@overload
def permute_equal_final_states(obj: ReactionInfo) -> ReactionInfo: ...
@overload
def permute_equal_final_states(
    obj: Iterable[FrozenTransition[EdgeType, NodeType]],
) -> list[FrozenTransition[EdgeType, NodeType]]: ...
@overload
def permute_equal_final_states(
    obj: FrozenTransition[EdgeType, NodeType],
) -> list[FrozenTransition[EdgeType, NodeType]]: ...
def permute_equal_final_states(obj: T) -> T:
    return _impl_permute_equal_final_states(obj)


@singledispatch
def _impl_permute_equal_final_states(obj) -> NoReturn:
    msg = f"Cannot permute equal final states of a {type(obj)}"
    raise NotImplementedError(msg)


@_impl_permute_equal_final_states.register(ReactionInfo)
def _(obj: ReactionInfo) -> ReactionInfo:
    return ReactionInfo(
        transitions=permute_equal_final_states(obj.transitions),
        formalism=obj.formalism,
    )


@_impl_permute_equal_final_states.register(abc.Iterable)
def _(
    obj: Iterable[FrozenTransition[EdgeType, NodeType]],
) -> list[FrozenTransition[EdgeType, NodeType]]:
    permuted_transitions = []
    for transition in obj:
        permuted_transitions.extend(permute_equal_final_states(transition))
    return permuted_transitions


@_impl_permute_equal_final_states.register(FrozenTransition)
def _(
    obj: FrozenTransition[EdgeType, NodeType],
) -> list[FrozenTransition[EdgeType, NodeType]]:
    transition = obj
    equal_state_ids = _get_equal_final_state_ids(transition)
    if not equal_state_ids:
        return [transition]
    unique_permutations = {transition} | {
        attrs.evolve(transition, topology=transition.topology.swap_edges(i, j))
        for i, j in itertools.combinations(equal_state_ids, 2)
    }
    return sorted(unique_permutations)


def _get_equal_final_state_ids(
    transition: FrozenTransition,
) -> (
    tuple[()]
    | tuple[FinalStateID, FinalStateID]
    | tuple[FinalStateID, FinalStateID, FinalStateID]
):
    particle_to_id = defaultdict(list)
    for idx, state in transition.final_states.items():
        key = _uniqueness_repr(state)
        particle_to_id[key].append(idx)
    all_equal_state_ids = [set(ids) for ids in particle_to_id.values() if len(ids) > 1]
    if not all_equal_state_ids:
        return ()
    return tuple(sorted(all_equal_state_ids[0]))


def _uniqueness_repr(obj: Any) -> str:
    if isinstance(obj, qrules.transition.State):
        return _uniqueness_repr(obj.particle)
    if isinstance(obj, (Particle, State, qrules.particle.Particle)):
        return obj.name
    msg = f"Cannot create a uniqueness key for {type(obj)}"
    raise NotImplementedError(msg)
