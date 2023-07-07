import operator
from functools import reduce

import sympy as sp


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
