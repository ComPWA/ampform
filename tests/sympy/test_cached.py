from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ampform.sympy import cached

if TYPE_CHECKING:
    import sympy as sp

    from ampform.helicity import HelicityModel


def test_doit(amplitude_model: tuple[str, HelicityModel]):
    _, model = amplitude_model
    expected_expr = model.expression.doit()
    assert expected_expr != model.expression

    unfolded_expr_1 = cached.doit(model.expression)
    assert unfolded_expr_1 == expected_expr
    unfolded_expr_2 = cached.doit(model.expression)
    assert unfolded_expr_2 == expected_expr


@pytest.mark.parametrize(
    "substitution_name", ["parameter_defaults", "kinematic_variables"]
)
def test_xreplace(amplitude_model: tuple[str, HelicityModel], substitution_name: str):
    _, model = amplitude_model
    full_expression = model.expression.doit()
    substitutions: dict[sp.Symbol, sp.Basic] = getattr(model, substitution_name)
    expected_expr = full_expression.xreplace(substitutions)
    assert expected_expr != full_expression

    substituted_expr_1 = cached.xreplace(full_expression, substitutions)
    assert substituted_expr_1 == expected_expr
    substituted_expr_2 = cached.xreplace(full_expression, substitutions)
    assert substituted_expr_2 == expected_expr


def test_unfold(amplitude_model: tuple[str, HelicityModel]):
    _, model = amplitude_model
    amplitudes = {k: v.doit() for k, v in model.amplitudes.items()}
    intensity_expr_direct = model.intensity.doit().xreplace(amplitudes)
    intensity_expr_unfold = cached.unfold(model.intensity, model.amplitudes)
    assert intensity_expr_direct == intensity_expr_unfold
