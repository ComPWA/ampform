import numpy as np
import pytest
import sympy as sp

from ampform.sympy.math import ComplexSqrt

a, b = sp.symbols("a b")


def describe_ComplexSqrt():
    @pytest.mark.parametrize(
        "arg",
        [
            sp.Symbol("x"),
            sp.Symbol("x", real=True),
            sp.Symbol("x", positive=True),
            a + b**2,
        ],
    )
    def it_leaves_symbolic_arguments_unevaluated(arg):
        assert ComplexSqrt(arg).doit() == ComplexSqrt(arg)

    def it_defines_real_and_imaginary_branches_piecewise():
        x = sp.Symbol("x")
        expr = ComplexSqrt(x).get_definition()
        assert expr == sp.Piecewise(
            (sp.I * sp.sqrt(-x), x < 0),
            (sp.sqrt(x), True),
        )

    def it_renders_as_latex():
        x = sp.Symbol("x")
        expr = ComplexSqrt(x)
        assert sp.latex(expr) == R"\sqrt[\mathrm{c}]{x}"

    @pytest.mark.parametrize("real", [False, True])
    @pytest.mark.parametrize("backend", ["math", "numpy"])
    def it_returns_imaginary_roots_with_each_backend(backend: str, real: bool):
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
    def it_evaluates_numeric_roots_and_preserves_symbolic_arguments(
        input_value, expected: str
    ):
        expr = ComplexSqrt(input_value)
        assert str(expr) == expected
