import pytest
import sympy as sp

from ampform.sympy import UnevaluatableIntegral


class TestUnevaluatableIntegral:
    def test_real_value_function(self):
        x = sp.symbols("x")
        integral_expr = UnevaluatableIntegral(x**2, (x, 1, 3))
        func = sp.lambdify(args=[], expr=integral_expr)
        assert func() == 26 / 3

    def test_array_value_parameter_function(self):
        x, p = sp.symbols("x,p")
        integral_expr = UnevaluatableIntegral(x**p, (x, 1, 3))
        func = sp.lambdify(args=[p], expr=integral_expr)
        assert func(p=2) == 26 / 3
        assert pytest.approx(func(p=1)) == 4
