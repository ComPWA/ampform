from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ampform import get_builder
from ampform.decay import DecayChain, IsobarNode, Particle, State
from ampform.helicity.naming import (
    CanonicalAmplitudeNameGenerator,
    HelicityAmplitudeNameGenerator,
    _render_float,
    generate_transition_label,
    get_boost_chain_suffix,
    get_topology_identifier,
)

if TYPE_CHECKING:
    from qrules import ReactionInfo

    from ampform.helicity import HelicityModel


def test_generate_transition_label(reaction: ReactionInfo):
    for transition in reaction.transitions:
        label = generate_transition_label(transition)
        jpsi_spin = _render_float(transition.states[-1].spin_projection)
        gamma_spin = _render_float(transition.states[0].spin_projection)
        assert label == (
            Rf"J/\psi(1S)_{{{jpsi_spin}}} \to \gamma_{{{gamma_spin}}}"
            R" \pi^{0}_{0} \pi^{0}_{0}"
        )


@pytest.mark.parametrize("parent_helicities", [False, True])
@pytest.mark.parametrize("child_helicities", [False, True])
@pytest.mark.parametrize("ls_combinations", [False, True])
def test_coefficient_names(
    reaction: ReactionInfo,
    parent_helicities,
    child_helicities,
    ls_combinations,
):
    builder = get_builder(reaction)
    assert isinstance(builder.naming, HelicityAmplitudeNameGenerator)
    builder.naming.insert_parent_helicities = parent_helicities
    builder.naming.insert_child_helicities = child_helicities
    if ls_combinations and reaction.formalism == "helicity":
        pytest.skip("No LS-combinations if using helicity formalism")
    if isinstance(builder.naming, CanonicalAmplitudeNameGenerator):
        builder.naming.insert_ls_combinations = ls_combinations
    model = builder.formulate()

    coefficients = get_coefficients(model)
    n_resonances = len(reaction.get_intermediate_particles())
    multiplicity = 1
    if parent_helicities:
        multiplicity *= 2
    if ls_combinations:
        multiplicity *= 2
    if child_helicities and (parent_helicities or ls_combinations):
        multiplicity *= 2
    assert len(coefficients) == multiplicity * n_resonances

    coefficient_name = coefficients[0]
    if parent_helicities:
        assert R"J/\psi(1S)_{-1}" in coefficient_name
    else:
        assert R"J/\psi(1S) " in coefficient_name

    if child_helicities:
        assert R"\gamma_{" in coefficient_name
    else:
        assert R"\gamma;" in coefficient_name

    if ls_combinations:
        assert R"\xrightarrow[S=1]{L=0}" in coefficient_name
    else:
        assert R"\to" in coefficient_name


def get_coefficients(model: HelicityModel) -> list[str]:
    return [
        str(symbol)
        for symbol in model.parameter_defaults
        if str(symbol).startswith("C_")
    ]


def _create_five_body_chain() -> DecayChain:
    """Create a dummy chain X → (R₁₄₅ → 1 (R₄₅ → 4 5)) (R₂₃ → 2 3)."""
    dummy_args = dict(spin=0, parity=None, mass=0.0, width=0.0)
    resonance = Particle("R", latex="R", **dummy_args)

    def create_state(index: int) -> State:
        return State(f"f{index}", latex=f"f_{index}", index=index, **dummy_args)

    return DecayChain(
        decay=IsobarNode(
            parent=create_state(0),
            child1=IsobarNode(
                parent=resonance,
                child1=create_state(1),
                child2=IsobarNode(resonance, create_state(4), create_state(5)),
            ),
            child2=IsobarNode(resonance, create_state(2), create_state(3)),
        )
    )


def test_get_boost_chain_suffix_of_decay_chain():
    """Boost-chain suffixes over a `.DecayChain` follow the `.Topology` conventions."""
    chain = _create_five_body_chain()
    expected = {
        (1,): "_1^145",
        (2,): "_2^23",
        (3,): "_3^23",
        (4,): "_4^45,145",
        (5,): "_5^45,145",
        (1, 4, 5): "_145",
        (2, 3): "_23",
        (4, 5): "_45^145",
        (1, 2, 3, 4, 5): "_12345",
    }
    assert {edge: get_boost_chain_suffix(chain, edge) for edge in expected} == expected
    assert get_boost_chain_suffix(chain, 3) == "_3^23"
    with pytest.raises(KeyError, match=r"no edge with final states \(1, 2\)"):
        get_boost_chain_suffix(chain, (1, 2))


def test_get_topology_identifier_of_decay_chain():
    chain = _create_five_body_chain()
    assert get_topology_identifier(chain) == "23,45,145"
