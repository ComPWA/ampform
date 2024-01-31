import pytest
import sympy as sp

from ampform.sympy import UnevaluatableIntegral


class TestUnevaluatableIntegral:
    def test_real_value_function(self):
        x = sp.symbols("x")
        integral_expr = UnevaluatableIntegral(x**2, (x, 1, 3))
        func = sp.lambdify(args=[], expr=integral_expr)
        assert func() == 26 / 3

    @pytest.mark.parametrize(
        "p_value,expected",
        [
            (2, 26 / 3),
            (1, 4),
        ],
    )
    def test_array_value_parameter_function(self, p_value, expected):
        x, p = sp.symbols("x,p")
        integral_expr = UnevaluatableIntegral(x**p, (x, 1, 3))
        func = sp.lambdify(args=[p], expr=integral_expr)
        assert pytest.approx(func(p=p_value)) == expected
