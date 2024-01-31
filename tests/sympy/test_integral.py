import sympy as sp

from ampform.sympy import UnevaluatableIntegral


class TestUnevaluatableIntegral:
    def test_real_value_function(self):
        x = sp.symbols("x")
        integral_expr = UnevaluatableIntegral(x**2, (x, 1, 3))
        func = sp.lambdify(args=[], expr=integral_expr)
        assert func() == 26 / 3
