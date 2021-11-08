# pylint: disable=no-self-use
import numpy as np
import pytest
import sympy as sp

from ampform.dynamics import ComplexSqrt


class TestComplexSqrt:
    @pytest.mark.parametrize("real", [False, True])
    def test_evaluate(self, real: bool):
        x = sp.Symbol("x", real=real)
        expr = ComplexSqrt(x).evaluate()
        if real:
            assert expr == sp.Piecewise(
                (sp.I * sp.sqrt(-x), x < 0),
                (sp.sqrt(x), True),
            )
        else:
            assert expr == sp.sqrt(x)

    def test_latex(self):
        x = sp.Symbol("x")
        expr = ComplexSqrt(x)
        assert sp.latex(expr) == R"\sqrt[\mathrm{c}]{x}"

    @pytest.mark.parametrize("real", [False, True])
    @pytest.mark.parametrize("backend", ["math", "numpy"])
    def test_lambdify(self, backend: str, real: bool):
        x = sp.Symbol("x", real=real)
        expression = ComplexSqrt(x)
        lambdified = sp.lambdify(x, expression, backend)
        assert lambdified(np.array(-1)) == 1j

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        [
            (sp.Symbol("x", real=True), "ComplexSqrt(x)"),
            (sp.Symbol("x"), "ComplexSqrt(x)"),
            (+4, "2"),
            (-4, "2*I"),
        ],
    )
    def test_new(self, input_value, expected: str):
        expr = ComplexSqrt(input_value)
        assert str(expr) == expected
