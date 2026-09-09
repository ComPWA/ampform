from functools import partial

import sympy as sp
from sympy.printing.numpy import NumPyPrinter

from ampform.sympy._array_expressions import ArrayMultiplication, ArraySum, ArraySymbol


def describe_ArrayMultiplication():
    def it_prints_array_contractions_as_einsum():
        n_events = 3
        momentum = sp.MatrixSymbol("p", m=n_events, n=4)
        beta = sp.Symbol("beta")
        theta = sp.Symbol("theta")
        expr = ArrayMultiplication(beta, theta, momentum)
        numpy_code = _generate_numpy_code(expr)
        assert numpy_code == 'einsum("...ij,...jk,...k->...i", beta, theta, p)'


def describe_ArraySum():
    def it_renders_as_latex():
        x, y = sp.symbols("x y")
        array_sum = ArraySum(x**2, sp.cos(y))
        assert sp.latex(array_sum) == R"x^{2} + \cos{\left(y \right)}"

    def it_combines_array_indices_in_latex():
        p0, p1, p2, p3 = sp.symbols("p:4", cls=partial(ArraySymbol, shape=[]))
        array_sum = ArraySum(p0, p1, p2, p3)
        assert sp.latex(array_sum) == "{p}_{0123}"

    def it_prints_addition_as_numpy_code():
        expr = ArraySum(*sp.symbols("x y"))
        numpy_code = _generate_numpy_code(expr)
        assert numpy_code == "x + y"


def _generate_numpy_code(expr: sp.Expr) -> str:
    # cspell:ignore doprint
    printer = NumPyPrinter()
    return printer.doprint(expr)
