from __future__ import annotations

import inspect
from typing import Any

import pytest
import sympy as sp

from ampform.sympy._decorator import (
    _check_has_implementation,
    _implement_latex_repr,
    _implement_new_method,
    _insert_args_in_signature,
    _set_assumptions,
    unevaluated_expression,
)


def test_check_implementation():
    with pytest.raises(ValueError, match=r"must have an evaluate\(\) method"):

        @_check_has_implementation
        class MyExpr1:  # pyright: ignore[reportUnusedClass]
            pass

    with pytest.raises(TypeError, match=r"evaluate\(\) must be a callable method"):

        @_check_has_implementation
        class MyExpr2:  # pyright: ignore[reportUnusedClass]
            evaluate = "test"


def test_implement_latex_repr():
    @_implement_latex_repr
    @_implement_new_method
    class MyExpr(sp.Expr):
        a: sp.Symbol
        b: sp.Symbol
        _latex_repr_ = R"f\left({a}, {b}\right)"

    alpha, phi = sp.symbols("alpha phi")
    expr = MyExpr(alpha, sp.cos(phi))
    assert sp.latex(expr) == R"f\left(\alpha, \cos{\left(\phi \right)}\right)"


def test_implement_new_method():
    @_implement_new_method
    class MyExpr(sp.Expr):
        a: int
        b: int
        c: int

    with pytest.raises(
        ValueError, match=r"^Expecting 3 positional arguments \(a, b, c\), but got 4$"
    ):
        MyExpr(1, 2, 3, 4)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match=r"^Missing constructor arguments: c$"):
        MyExpr(1, 2)  # type: ignore[call-arg]
    expr = MyExpr(1, 2, 3)
    assert expr.args == (1, 2, 3)
    expr = MyExpr(1, b=2, c=3)
    assert expr.args == (1, 2, 3)


def test_insert_args_in_signature():
    parameters = ["a", "b"]

    @_insert_args_in_signature(parameters)
    def my_func(*args, **kwargs) -> int:
        return 1

    signature = inspect.signature(my_func)
    assert list(signature.parameters) == [*parameters, "args", "kwargs"]
    assert signature.return_annotation == "int"


def test_unevaluated_expression():
    @unevaluated_expression
    class BreakupMomentum(sp.Expr):
        r"""Breakup momentum of a two-body decay :math:`a \to 1+2`."""

        s: sp.Basic
        m1: sp.Basic
        m2: sp.Basic
        _latex_repr_ = R"q\left({s}\right)"

        def evaluate(self) -> sp.Expr:
            s, m1, m2 = self.args
            return sp.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2))  # type: ignore[operator]

    m0, ma, mb = sp.symbols("m0 m_a m_b")
    expr = BreakupMomentum(m0**2, ma, mb)
    args_str = list(inspect.signature(expr.__new__).parameters)
    assert args_str == ["s", "m1", "m2", "args", "evaluate", "kwargs"]
    latex = sp.latex(expr)
    assert latex == R"q\left(m_{0}^{2}\right)"


def test_unevaluated_expression_callable():
    @unevaluated_expression(implement_doit=False)
    class Squared(sp.Expr):
        x: Any

        def evaluate(self) -> sp.Expr:
            return self.x**2

    sqrt = Squared(2)
    assert str(sqrt) == "Squared(2)"
    assert str(sqrt.doit()) == "Squared(2)"

    @unevaluated_expression(complex=True, implement_doit=False)
    class MySqrt(sp.Expr):
        x: Any

    expr = MySqrt(-1)
    assert expr.is_commutative
    assert expr.is_complex  # type: ignore[attr-defined]


def test_set_assumptions():
    @_set_assumptions(commutative=True, negative=False, real=True)
    class MySqrt: ...

    expr = MySqrt()
    assert expr.is_commutative  # type: ignore[attr-defined]
    assert not expr.is_negative  # type: ignore[attr-defined]
    assert expr.is_real  # type: ignore[attr-defined]
