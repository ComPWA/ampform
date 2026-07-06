r"""Data structures that describe an isobar decay.

The classes in this module are the interface between decay descriptions from external
packages (see e.g. the `.adapter.qrules` module) and the amplitude builders provided by
AmpForm. The structures support :math:`1 \to n` isobar decays of arbitrary decay depth.
Three-body decays are described by the specializations `ThreeBodyDecay` and
`ThreeBodyDecayChain`.
"""

from __future__ import annotations

from functools import cache
from itertools import product
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    Protocol,
    SupportsFloat,
    SupportsInt,
    TypeVar,
    overload,
)
from warnings import warn

from attrs import field, frozen
from attrs.validators import instance_of

from ampform._attrs import assert_spin_value, to_chains, to_ls, to_rational

if TYPE_CHECKING:
    import sympy as sp

InitialStateID = Literal[0]
"""ID for the initial state particle in an isobar decay."""
FinalStateID = Literal[1, 2, 3]
"""ID for a particle in the final state of a three-body decay."""
StateID = Literal[0, 1, 2, 3]
"""ID for any of the initial or final state particles in a three-body decay."""
StateIDTemplate = TypeVar("StateIDTemplate", bound=int)
"""Generic template for the ID of a particle in an isobar decay."""


class ParticleLike(Protocol):
    """Structural type for the particle info that amplitude builders require.

    Both `.Particle` and `qrules.particle.Particle` satisfy this interface.
    """

    @property
    def name(self) -> str: ...
    @property
    def latex(self) -> str | None: ...
    @property
    def spin(self) -> SupportsFloat: ...
    @property
    def parity(self) -> SupportsInt | None: ...
    @property
    def mass(self) -> float: ...
    @property
    def width(self) -> float: ...


@frozen(order=True)
class Particle:
    name: str
    latex: str
    spin: sp.Rational = field(converter=to_rational, validator=assert_spin_value)
    parity: Literal[-1, 1] | None
    mass: float
    width: float


@frozen(order=True)
class State(Particle, Generic[StateIDTemplate]):
    """Initial or final state `.Particle` in a `Decay`, carrying an index."""

    index: StateIDTemplate


InitialState = State[InitialStateID]
"""The initial state particle."""
FinalState = State[FinalStateID]
"""One of the final state particles in a three-body decay."""
ParentType = TypeVar("ParentType", Particle, InitialState)
"""Type of the parent of an `IsobarNode`."""


@frozen(order=True)
class IsobarNode(Generic[ParentType]):
    parent: ParentType
    child1: IsobarNode[Particle] | State
    child2: IsobarNode[Particle] | State
    interaction: LSCoupling | None = field(default=None, converter=to_ls)

    @property
    def children(self) -> tuple[DecayNode | State, DecayNode | State]:
        return self.child1, self.child2


ProductionNode = IsobarNode[InitialState]
"""The first `IsobarNode` in a `DecayChain`."""
DecayNode = IsobarNode[Particle]
"""Any `IsobarNode` in a `DecayChain` that is not the `ProductionNode`."""


@frozen(order=True)
class DecayChain:
    r"""Isobar decay chain for a :math:`1 \to n` decay of arbitrary decay depth.

    Each decay chain is a rooted binary tree of `IsobarNode` instances, where the
    leaves are the final `State` particles and the root is the `ProductionNode` that
    has the `.initial_state` as its parent.
    """

    decay: ProductionNode = field(validator=instance_of(IsobarNode))

    def __attrs_post_init__(self) -> None:
        outer_states = [self.parent, *_collect_leaves(self.decay)]
        non_states = [p for p in outer_states if not isinstance(p, State)]
        if non_states:
            names = ", ".join(p.name for p in non_states)
            msg = (
                "All particles in the initial and final state have to be of type"
                f" {State.__name__}, but the following are not: {names}"
            )
            raise TypeError(msg)
        if len({state.index for state in outer_states}) != len(outer_states):
            msg = "The initial and/or final state contains particles with the same ID:"
            for state in outer_states:
                msg += f"\n  {state.index}: {state.name}"
            raise ValueError(msg)

    @property
    def initial_state(self) -> InitialState:
        return self.parent

    @property
    def final_state(self) -> dict[int, State]:
        """Final state particles of the decay chain, keyed by their ID."""
        leaves = _collect_leaves(self.decay)
        return {state.index: state for state in sorted(leaves, key=lambda s: s.index)}

    @property
    def parent(self) -> InitialState:
        return self.decay.parent

    @property
    def nodes(self) -> tuple[IsobarNode, ...]:
        """All `IsobarNode` instances in the chain, starting with the production node."""
        return _collect_nodes(self.decay)

    @property
    def resonances(self) -> tuple[Particle, ...]:
        """Intermediate particles in the decay chain."""
        return tuple(
            node.parent for node in self.nodes if not isinstance(node.parent, State)
        )


def _collect_leaves(node: IsobarNode) -> list[State]:
    """Collect the final `State` leaves of an isobar tree.

    The returned list may contain plain `Particle` instances if the tree is invalid,
    which is checked by `DecayChain.__attrs_post_init__`.
    """
    leaves: list[State] = []
    for child in node.children:
        if isinstance(child, IsobarNode):
            leaves.extend(_collect_leaves(child))
        else:
            leaves.append(child)
    return leaves


def _collect_nodes(node: IsobarNode) -> tuple[IsobarNode, ...]:
    nodes = [node]
    for child in node.children:
        if isinstance(child, IsobarNode):
            nodes.extend(_collect_nodes(child))
    return tuple(nodes)


def get_final_state_ids(node: IsobarNode | State) -> tuple[int, ...]:
    """Get the sorted IDs of the final states that are attached to this node.

    Each edge in an isobar tree is uniquely identified by the final states that it
    eventually decays into, so the returned tuple can be used as an identifier for the
    edge that flows into :code:`node`.
    """
    if isinstance(node, IsobarNode):
        return tuple(sorted(state.index for state in _collect_leaves(node)))
    return (node.index,)


def to_edge_id(state_id: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize a state ID to an edge identifier over a `DecayChain`.

    >>> to_edge_id(2)
    (2,)
    >>> to_edge_id((3, 1, 2))
    (1, 2, 3)
    """
    if isinstance(state_id, int):
        return (state_id,)
    return tuple(sorted(state_id))


def get_edge_ids(chain: DecayChain) -> tuple[tuple[int, ...], ...]:
    """Get the identifiers of all edges in a `DecayChain`.

    Each edge is identified by its attached final-state IDs (see
    :func:`get_final_state_ids`). The first entry is the initial state edge, followed by
    the incoming edges of each child in the chain, ordered depth-first.
    """
    edge_ids = [get_final_state_ids(chain.decay)]
    for node in chain.nodes:
        edge_ids.extend(get_final_state_ids(child) for child in node.children)
    return tuple(edge_ids)


def generate_helicity_assignments(
    chain: DecayChain,
) -> list[dict[tuple[int, ...], sp.Rational]]:
    r"""Enumerate the allowed helicity combinations of a `DecayChain`.

    The helicity values of each particle in the decay chain are determined from its
    spin magnitude and mass (see :func:`.get_spin_projections`) and each two-body decay
    node :math:`p \to c_1 c_2` imposes the constraint
    :math:`|\lambda_{c_1} - \lambda_{c_2}| \le s_p`. Nodes that carry an `LSCoupling`
    restrict the combinations further to those with non-vanishing Clebsch-Gordan
    coefficients (see :func:`.formulate_isobar_cg_coefficients`). Each returned
    combination maps the decay edges, identified by their attached final-state IDs (see
    :func:`get_final_state_ids`), to a helicity value. This provides the same
    information as the spin projections in a collection of
    `qrules.topology.FrozenTransition` objects, without requiring QRules.
    """
    from ampform.decay.spin import get_spin_projections

    edges: dict[tuple[int, ...], Particle] = {
        get_final_state_ids(chain.decay): chain.parent
    }
    node_constraints = []
    for node in chain.nodes:
        child1_key, child2_key = (get_final_state_ids(c) for c in node.children)
        edges[child1_key] = to_particle(node.child1)
        edges[child2_key] = to_particle(node.child2)
        node_constraints.append((node, child1_key, child2_key))
    keys = list(edges)
    helicity_ranges = [get_spin_projections(edges[key]) for key in keys]
    assignments = []
    for helicities in product(*helicity_ranges):
        assignment = dict(zip(keys, helicities, strict=True))
        if all(
            _is_allowed_node_helicity(
                node, assignment[child1_key], assignment[child2_key]
            )
            for node, child1_key, child2_key in node_constraints
        ):
            assignments.append(assignment)
    return assignments


def _is_allowed_node_helicity(
    node: IsobarNode, λ1: sp.Rational, λ2: sp.Rational
) -> bool:
    from sympy.physics.quantum.cg import CG

    parent_spin = node.parent.spin
    difference = λ1 - λ2
    if abs(difference) > parent_spin:
        return False
    if node.interaction is None:
        return True
    L, S = node.interaction.L, node.interaction.S
    if abs(difference) > S:
        return False
    return (
        CG(L, 0, S, difference, parent_spin, difference).doit() != 0
        and CG(
            to_particle(node.child1).spin,
            λ1,
            to_particle(node.child2).spin,
            -λ2,
            S,
            difference,
        ).doit()
        != 0
    )


ChainType = TypeVar("ChainType", bound=DecayChain)
"""Type of the `DecayChain` instances in a `Decay`."""


@frozen
class Decay(Generic[ChainType]):
    """Collection of `DecayChain` instances that share the same initial and final state."""

    states: dict[int, State]
    chains: tuple[ChainType, ...] = field(converter=to_chains)

    def __attrs_post_init__(self) -> None:
        expected_initial_state = self.initial_state
        expected_final_state = set(self.final_state.values())
        for i, chain in enumerate(self.chains):
            if chain.parent != expected_initial_state:
                msg = dedent(f"""
                    Chain {i} has initial state
                      {chain.parent.index}: {chain.parent.name}
                    but should have
                      {expected_initial_state.index}: {expected_initial_state.name}
                """).strip()
                raise ValueError(msg)
            final_state = set(chain.final_state.values())
            if final_state != expected_final_state:

                def to_str(s: set[State]) -> str:
                    return ", ".join(
                        f"{p.index}: {p.name}" for p in sorted(s, key=lambda x: x.index)
                    )

                msg = dedent(f"""
                    Chain {i} has final state
                       {to_str(final_state)}
                    but should have
                       {to_str(expected_final_state)}
                """).strip()
                raise ValueError(msg)

    @property
    def initial_state(self) -> InitialState:
        return self.states[0]

    @property
    def final_state(self) -> dict[int, State]:
        return {s.index: s for s in self.states.values() if s.index != 0}

    def find_chain(self, resonance_name: str) -> ChainType:
        for chain in self.chains:
            resonance_names = [p.name for p in chain.resonances]
            if resonance_name in resonance_names:
                return chain
        msg = f"No decay chain found for resonance {resonance_name}"
        raise KeyError(msg)


@frozen
class ThreeBodyDecay(Decay["ThreeBodyDecayChain"]):
    """A `Decay` with exactly three final state particles."""

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if set(self.final_state) != {1, 2, 3}:
            final_state_ids = ", ".join(str(i) for i in sorted(self.final_state))
            msg = f"Final state IDs have to be 1, 2, 3, but got {final_state_ids}"
            raise ValueError(msg)
        for chain in self.chains:
            if not isinstance(chain, ThreeBodyDecayChain):
                msg = (
                    f"Chains have to be of type {ThreeBodyDecayChain.__name__}, but"
                    f" got a {type(chain).__name__}"
                )
                raise TypeError(msg)

    @property
    def final_state(self) -> dict[FinalStateID, FinalState]:
        return {s.index: s for s in self.states.values() if s.index != 0}

    def get_subsystem(self, subsystem_id: FinalStateID) -> ThreeBodyDecay:
        filtered_chains = [c for c in self.chains if c.spectator.index == subsystem_id]
        if not filtered_chains:
            decay_description = _get_decay_description(self)
            subsystems = ", ".join(sorted(str(i) for i in _get_subsystem_ids(self)))
            msg = f"Decay {decay_description} only has subsystems {subsystems}, not {subsystem_id}"
            warn(msg, category=UserWarning)
        return ThreeBodyDecay(self.states, filtered_chains)


def _get_decay_description(decay: Decay) -> str:
    initial_state = decay.initial_state.name
    final_state = ", ".join(f"{i}: {s.name}" for i, s in decay.final_state.items())
    return f"{initial_state} → {final_state}"


def _get_subsystem_ids(decay: ThreeBodyDecay) -> set[FinalStateID]:
    return {c.spectator.index for c in decay.chains}


def get_decay_product_ids(
    spectator_id: FinalStateID,
) -> tuple[FinalStateID, FinalStateID]:
    if spectator_id == 1:
        return 2, 3
    if spectator_id == 2:  # ruff: ignore[magic-value-comparison]
        return 3, 1
    if spectator_id == 3:  # ruff: ignore[magic-value-comparison]
        return 1, 2
    msg = f"Spectator ID has to be one of 1, 2, 3, not {spectator_id}"
    raise ValueError(msg)


@frozen(order=True)
class ThreeBodyDecayChain(DecayChain):
    """A `DecayChain` with exactly three final state particles."""

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        n_final_states = len(self.final_state)
        if n_final_states != 3:  # ruff: ignore[magic-value-comparison]
            msg = f"Expected 3 final state particles, but got {n_final_states}"
            raise ValueError(msg)

    @property
    def resonance(self) -> Particle:
        return to_particle(self.decay_node)

    @property
    def production_node(self) -> ProductionNode:
        return self.decay

    @property
    def decay_node(self) -> DecayNode:
        return self._get_child_of_type(IsobarNode)

    @property
    def decay_products(self) -> tuple[FinalState, FinalState]:
        return (
            to_particle(self.decay_node.child1),
            to_particle(self.decay_node.child2),
        )  # ty:ignore[invalid-return-type]

    @property
    def spectator(self) -> FinalState:
        return self._get_child_of_type(State)

    @cache  # ruff: ignore[cached-instance-method]
    def _get_child_of_type(self, typ: type[T]) -> T:
        for child in self.decay.children:
            if isinstance(child, typ):
                return child
        msg = f"The production node does not have any children that are of type {typ.__name__}"
        raise ValueError(msg)

    @property
    def incoming_ls(self) -> LSCoupling | None:
        return self.decay.interaction

    @property
    def outgoing_ls(self) -> LSCoupling | None:
        return self.decay_node.interaction


T = TypeVar("T", IsobarNode, Particle, InitialState, FinalState)


@frozen(order=True)
class LSCoupling:
    L: int
    S: sp.Rational = field(converter=to_rational, validator=assert_spin_value)


@overload
def to_particle(isobar: IsobarNode[ParentType]) -> ParentType: ...
@overload
def to_particle(isobar: State[StateIDTemplate]) -> State[StateIDTemplate]: ...
@overload
def to_particle(isobar: Particle) -> Particle: ...
def to_particle(isobar):
    if isinstance(isobar, IsobarNode):
        return isobar.parent
    return isobar
