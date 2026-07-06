from __future__ import annotations

import attrs
import pytest
import sympy as sp

from ampform.decay import (
    Decay,
    DecayChain,
    IsobarNode,
    Particle,
    State,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
    generate_helicity_assignments,
    get_edge_ids,
    get_final_state_ids,
)

# https://compwa-org--129.org.readthedocs.build/report/018.html#resonances-and-ls-scheme
dummy_args = dict(mass=0, width=0)
Λc = Particle("Λc", latex=R"\Lambda_c^+", spin=0.5, parity=+1, **dummy_args)
p = Particle("p", latex="p", spin=0.5, parity=+1, **dummy_args)
π = Particle("π+", latex=R"\pi^+", spin=0, parity=-1, **dummy_args)
K = Particle("K-", latex="K^-", spin=0, parity=-1, **dummy_args)
Λ1520 = Particle("Λ(1520)", latex=R"\Lambda(1520)", spin=1.5, parity=-1, **dummy_args)


def to_state(particle: Particle, index: int) -> State:
    return State(**attrs.asdict(particle, recurse=False), index=index)


class TestIsobarNode:
    def test_children(self):
        decay = IsobarNode(Λ1520, p, K)
        assert decay.children == (p, K)

    def test_ls(self):
        L, S = 2, 1
        node = IsobarNode(Λ1520, p, K, interaction=(L, S))
        assert node.interaction is not None
        assert node.interaction.L == L
        assert node.interaction.S == S


class TestDecayChain:
    """Tests with a four-body decay chain Λc → (Λ(1520) → (Λ(1520) → p K) π) π."""

    chain = DecayChain(
        decay=IsobarNode(
            parent=to_state(Λc, index=0),
            child1=IsobarNode(
                parent=Λ1520,
                child1=IsobarNode(
                    parent=Λ1520,
                    child1=to_state(p, index=1),
                    child2=to_state(K, index=2),
                ),
                child2=to_state(π, index=3),
            ),
            child2=to_state(π, index=4),
        )
    )

    def test_final_state(self):
        assert list(self.chain.final_state) == [1, 2, 3, 4]
        final_state_names = [s.name for s in self.chain.final_state.values()]
        assert final_state_names == ["p", "K-", "π+", "π+"]

    def test_initial_state(self):
        assert self.chain.initial_state.name == "Λc"
        assert self.chain.initial_state.index == 0

    def test_nodes(self):
        nodes = self.chain.nodes
        assert len(nodes) == 3
        assert nodes[0] is self.chain.decay

    def test_resonances(self):
        assert self.chain.resonances == (Λ1520, Λ1520)

    def test_duplicate_state_ids(self):
        with pytest.raises(ValueError, match="particles with the same ID"):
            DecayChain(
                decay=IsobarNode(
                    parent=to_state(Λc, index=0),
                    child1=IsobarNode(
                        parent=Λ1520,
                        child1=to_state(p, index=1),
                        child2=to_state(K, index=1),
                    ),
                    child2=to_state(π, index=2),
                )
            )

    def test_leaves_must_be_states(self):
        with pytest.raises(TypeError, match="the following are not: p, K-"):
            DecayChain(
                decay=IsobarNode(
                    parent=to_state(Λc, index=0),
                    child1=IsobarNode(parent=Λ1520, child1=p, child2=K),
                    child2=to_state(π, index=3),
                )
            )


def test_get_final_state_ids():
    chain = TestDecayChain.chain
    assert get_final_state_ids(chain.decay) == (1, 2, 3, 4)
    node1, node2 = chain.decay.child1, chain.decay.child2
    assert isinstance(node1, IsobarNode)
    assert get_final_state_ids(node1) == (1, 2, 3)
    assert get_final_state_ids(node2) == (4,)


def test_get_edge_ids():
    chain = TestDecayChain.chain
    assert get_edge_ids(chain) == (
        (1, 2, 3, 4),
        (1, 2, 3),
        (4,),
        (1, 2),
        (3,),
        (1,),
        (2,),
    )


class TestGenerateHelicityAssignments:
    def test_massless_states_have_no_zero_helicity(self):
        ψ = to_state(
            Particle("psi", latex=R"\psi", spin=1, parity=-1, mass=3.1, width=0.0),
            index=0,
        )
        γ = to_state(
            Particle("gamma", latex=R"\gamma", spin=1, parity=-1, mass=0.0, width=0.0),
            index=1,
        )
        f0 = Particle("f(0)(980)", latex="f_0(980)", spin=0, parity=+1, **dummy_args)
        chain = DecayChain(
            decay=IsobarNode(
                parent=ψ,
                child1=IsobarNode(f0, to_state(π, index=2), to_state(π, index=3)),
                child2=γ,
            )
        )
        assignments = generate_helicity_assignments(chain)
        assert len(assignments) == 6
        assert {a[1,] for a in assignments} == {-1, +1}
        assert {a[1, 2, 3] for a in assignments} == {-1, 0, +1}
        assert all(a[2, 3] == 0 for a in assignments)

    def test_helicity_coupling_constraint(self):
        chain = DecayChain(
            decay=IsobarNode(
                parent=to_state(Λc, index=0),
                child1=IsobarNode(
                    parent=Λ1520,
                    child1=to_state(p, index=1),
                    child2=to_state(K, index=2),
                ),
                child2=to_state(π, index=3),
            )
        )
        assignments = generate_helicity_assignments(chain)
        assert len(assignments) == 8
        resonance_helicities = {a[1, 2] for a in assignments}
        assert resonance_helicities == {-sp.Rational(1, 2), +sp.Rational(1, 2)}


class TestDecay:
    def test_find_chain(self):
        chain = TestDecayChain.chain
        decay = Decay(
            states={
                s.index: s for s in [chain.initial_state, *chain.final_state.values()]
            },
            chains=[chain],
        )
        assert decay.find_chain("Λ(1520)") is chain
        with pytest.raises(KeyError, match="No decay chain found"):
            decay.find_chain("N(1440)")

    def test_mismatched_final_state(self):
        chain = TestDecayChain.chain
        with pytest.raises(ValueError, match="Chain 0 has final state"):
            Decay(
                states={
                    0: chain.initial_state,
                    1: to_state(p, index=1),
                    2: to_state(K, index=2),
                    3: to_state(π, index=3),
                    4: to_state(K, index=4),
                },
                chains=[chain],
            )


class TestThreeBodyDecayChain:
    def test_number_of_final_states(self):
        with pytest.raises(
            ValueError, match="Expected 3 final state particles, but got 4"
        ):
            ThreeBodyDecayChain(decay=TestDecayChain.chain.decay)


class TestThreeBodyDecay:
    def test_final_state_ids(self):
        chain = ThreeBodyDecayChain(
            decay=IsobarNode(
                parent=to_state(Λc, index=0),
                child1=IsobarNode(
                    parent=Λ1520,
                    child1=to_state(p, index=1),
                    child2=to_state(K, index=2),
                ),
                child2=to_state(π, index=4),
            )
        )
        states = {
            s.index: s for s in [chain.initial_state, *chain.final_state.values()]
        }
        with pytest.raises(
            ValueError, match="Final state IDs have to be 1, 2, 3, but got 1, 2, 4"
        ):
            ThreeBodyDecay(states=states, chains=[chain])
