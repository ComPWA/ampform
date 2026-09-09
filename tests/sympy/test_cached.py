# cspell:ignore pbarksigma
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp
from frozendict import frozendict

from ampform.sympy import cached
from ampform.sympy._cache import to_bytes
from ampform.sympy.cached import _sorted_frozendict

if TYPE_CHECKING:
    from ampform.helicity import HelicityModel


def test_doit(amplitude_model: tuple[str, HelicityModel]):
    _, model = amplitude_model
    expected_expr = model.expression.doit()
    assert expected_expr != model.expression

    unfolded_expr_1 = cached.doit(model.expression)
    assert unfolded_expr_1 == expected_expr
    unfolded_expr_2 = cached.doit(model.expression)
    assert unfolded_expr_2 == expected_expr


def describe_simplify():
    def it_matches_uncached_simplification():
        a, b, c, d, x, y, z = sp.symbols("a b c d x y z")
        expr = (
            (a * x + b * y + c * z + d) ** 2
            - (a * x) ** 2
            - (b * y) ** 2
            - (c * z) ** 2
            - 2 * a * b * x * y
            - 2 * a * c * x * z
            - 2 * b * c * y * z
            - 2 * d * (a * x + b * y + c * z)
        )

        simplified = expr.simplify()
        assert simplified != expr
        assert simplified == d**2

        cached_simplified = cached.simplify(expr)
        assert simplified == cached_simplified

    @pytest.mark.parametrize("amplitude_idx", list(range(4)))
    def it_simplifies_amplitude_model(
        amplitude_model: tuple[str, HelicityModel], amplitude_idx: int
    ):
        _, model = amplitude_model
        _, amplitude_expr = min(model.amplitudes.items(), key=lambda item: str(item[0]))

        simplified = amplitude_expr.simplify()
        assert simplified != amplitude_expr

        cached_simplified = cached.simplify(amplitude_expr)
        assert simplified == cached_simplified


def test_trigsimp():
    x, y = sp.symbols("x y")
    expr = (sp.sin(x) * sp.cos(y) + sp.cos(x) * sp.sin(y)) ** 2 + (
        sp.cos(x) * sp.cos(y) - sp.sin(x) * sp.sin(y)
    ) ** 2
    simplified = sp.trigsimp(expr)
    assert expr != simplified
    assert simplified == 1
    cached_simplified = cached.trigsimp(expr)
    assert simplified == cached_simplified


def test_unfold(amplitude_model: tuple[str, HelicityModel]):
    _, model = amplitude_model
    amplitudes = {k: v.doit() for k, v in model.amplitudes.items()}
    intensity_expr_direct = model.intensity.doit().xreplace(amplitudes)
    intensity_expr_unfold = cached.unfold(model.intensity, model.amplitudes)
    assert intensity_expr_direct == intensity_expr_unfold


@pytest.mark.parametrize("substitute", ["subs", "xreplace"])
@pytest.mark.parametrize(
    "substitution_name", ["parameter_defaults", "kinematic_variables"]
)
def test_xreplace(
    amplitude_model: tuple[str, HelicityModel],
    substitute: str,
    substitution_name: str,
):
    cached_func = getattr(cached, substitute)
    _, model = amplitude_model
    full_expression = model.expression.doit()
    substitutions: dict[sp.Symbol, sp.Basic] = getattr(model, substitution_name)
    expected_expr = full_expression.xreplace(substitutions)
    assert expected_expr != full_expression

    substituted_expr_1 = cached_func(full_expression, substitutions)
    assert substituted_expr_1 == expected_expr
    substituted_expr_2 = cached_func(full_expression, substitutions)
    assert substituted_expr_2 == expected_expr


def test_xreplace_ignores_substitution_order():
    """Substitutions built in a different order must map to the same cache key."""
    expr = sp.sympify("a*sin(b) + c**2 + d/e + f")
    forward = {s: sp.Symbol(f"{s}_new") for s in sorted(expr.free_symbols, key=str)}
    backward = dict(reversed(list(forward.items())))
    assert tuple(forward) != tuple(backward)
    assert to_bytes(frozendict(forward)) != to_bytes(frozendict(backward))
    assert to_bytes(_sorted_frozendict(forward)) == to_bytes(
        _sorted_frozendict(backward)
    )


def test_xreplace_orders_symbols_with_equal_sort_key():
    """Symbols that differ only in their assumptions must still get a fixed order."""
    x = sp.Symbol("x")
    x_real = sp.Symbol("x", real=True)
    assert sp.default_sort_key(x) == sp.default_sort_key(x_real)
    forward = {x: 1, x_real: 2}
    backward = {x_real: 2, x: 1}
    assert to_bytes(_sorted_frozendict(forward)) == to_bytes(
        _sorted_frozendict(backward)
    )
