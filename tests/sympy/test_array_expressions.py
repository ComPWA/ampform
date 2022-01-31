# pylint: disable=no-self-use
# cspell:ignore doprint
import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from ampform.sympy._array_expressions import (
    ArrayMultiplication,
    ArraySum,
    ArraySymbol,
)


class TestArrayMultiplication:
    def test_numpy_str(self):
        n_events = 3
        momentum = sp.MatrixSymbol("p", m=n_events, n=4)
        beta = sp.Symbol("beta")
        theta = sp.Symbol("theta")
        expr = ArrayMultiplication(beta, theta, momentum)
        numpy_code = _generate_numpy_code(expr)
        assert numpy_code == 'einsum("...ij,...jk,...k->...i", beta, theta, p)'


class TestArraySum:
    def test_latex(self):
        x, y = sp.symbols("x y")
        array_sum = ArraySum(x**2, sp.cos(y))
        assert sp.latex(array_sum) == R"x^{2} + \cos{\left(y \right)}"

    def test_latex_array_symbols(self):
        p0, p1, p2, p3 = sp.symbols("p:4", cls=ArraySymbol)
        array_sum = ArraySum(p0, p1, p2, p3)
        assert sp.latex(array_sum) == "{p}_{0123}"

    def test_numpy(self):
        expr = ArraySum(*sp.symbols("x y"))
        numpy_code = _generate_numpy_code(expr)
        assert numpy_code == "x + y"


def _generate_numpy_code(expr: sp.Expr) -> str:
    printer = NumPyPrinter()
    return printer.doprint(expr)
