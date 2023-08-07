from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ampform import get_builder
from ampform.helicity.naming import (
    CanonicalAmplitudeNameGenerator,
    HelicityAmplitudeNameGenerator,
    _render_float,
    generate_transition_label,
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
def test_coefficient_names(  # noqa: C901, PLR0912, PLR0915
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
    if reaction.formalism == "helicity":
        if parent_helicities:
            if child_helicities:
                assert len(coefficients) == 4 * n_resonances
            else:
                assert len(coefficients) == 2 * n_resonances
        else:  # noqa: PLR5501
            if child_helicities:
                assert len(coefficients) == n_resonances
            else:
                assert len(coefficients) == n_resonances
    elif reaction.formalism == "canonical-helicity":
        if ls_combinations:
            if parent_helicities:
                if child_helicities:
                    assert len(coefficients) == 8 * n_resonances
                else:
                    assert len(coefficients) == 4 * n_resonances
            else:  # noqa: PLR5501
                if child_helicities:
                    assert len(coefficients) == 4 * n_resonances
                else:
                    assert len(coefficients) == 2 * n_resonances
        else:  # noqa: PLR5501
            if parent_helicities:
                if child_helicities:
                    assert len(coefficients) == 4 * n_resonances
                else:
                    assert len(coefficients) == 2 * n_resonances
            else:
                assert len(coefficients) == n_resonances

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
