import operator
from functools import reduce

import sympy as sp

from ampform.sympy import partial_doit


class TestFunction:
    def test_hash(self):
        x = sp.Symbol("x")
        f = sp.Function("h")
        g = sp.Function("h")
        assert f is g
        assert f(x) is g(x)
        f = sp.Function("h")(x)
        g = sp.Function("h")(x)
        assert f is g


class TestSymbol:
    def test_hash(self):
        x = sp.Symbol("a")
        y = sp.Symbol("a")
        y_real = sp.Symbol("a", real=True)
        assert x == y
        assert x is y
        assert y != y_real
        assert hash(x) == hash(y)

    def test_name(self):
        x = sp.Symbol("x; weird-spacing\t.,")
        f = sp.Function("  f.^")
        g = sp.Function("g")(x)
        assert x.name == "x; weird-spacing	.,"
        # cspell:ignore srepr
        assert sp.srepr(x) == "Symbol('x; weird-spacing\\t.,')"
        assert f.name == "  f.^"
        assert g.name == "g"
        x.name = "x"
        assert x.name == "x"

    def test_product(self):
        symbols = [
            sp.Symbol("x"),
            sp.Symbol("y"),
            sp.Symbol("z"),
        ]
        reduce(operator.mul, symbols)


def test_partial_doit():
    x, m, n = sp.symbols("x m n")
    expr = (
        sp.Integral(sp.Sum(sp.sin(x) / n, (n, 1, 3)), x)
        + sp.Derivative(sp.Product(sp.cos(x) * sp.exp(-x) / n, (n, 1, 3)), x)
        + sp.Sum(sp.Integral(1 / n**2, (n, 1, 10)), (n, 1, 3))
        + sp.Sum(sp.sin(sp.Sum(1 / (n * m), (m, 1, 5))), (n, 1, 5))
    )
    assert expr is partial_doit(expr, types=())

    unfolded_expr = expr.doit()
    assert unfolded_expr is not expr
    n_ops = sp.count_ops(expr)
    n_ops_unfolded = sp.count_ops(unfolded_expr)

    n_ops_doit_sum = sp.count_ops(partial_doit(expr, sp.Sum))
    assert n_ops_doit_sum != n_ops
    assert n_ops_doit_sum != n_ops_unfolded

    n_ops_doit_sum_recursive = sp.count_ops(partial_doit(expr, sp.Sum, recursive=True))
    assert n_ops_doit_sum_recursive != n_ops
    assert n_ops_doit_sum_recursive != n_ops_doit_sum
    assert n_ops_doit_sum_recursive != n_ops_unfolded

    n_ops_doit_sum_integral = sp.count_ops(partial_doit(expr, (sp.Integral, sp.Sum)))
    assert n_ops_doit_sum_integral != n_ops
    assert n_ops_doit_sum_integral != n_ops_doit_sum
    assert n_ops_doit_sum_integral != n_ops_doit_sum_recursive
    assert n_ops_doit_sum_integral != n_ops_unfolded

    all_unfolded_expr = partial_doit(
        expr,
        types=(sp.Derivative, sp.Integral, sp.Product, sp.Sum),
        recursive=True,
    ).simplify()
    assert all_unfolded_expr == unfolded_expr.simplify()

    wrong_type_unfolded_expr = partial_doit(expr, sp.sin)
    assert wrong_type_unfolded_expr == expr


def test_partial_doit_top_node():
    x, n = sp.symbols("x n")
    expr = sp.Sum(sp.sin(x) / n, (n, 1, 3))
    doit_expr = expr.doit()
    partial_doit_expr = partial_doit(expr, sp.Sum)
    assert doit_expr == partial_doit_expr
