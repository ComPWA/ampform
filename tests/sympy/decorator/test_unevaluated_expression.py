from __future__ import annotations

import inspect
from typing import Any, ClassVar

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


def test_set_assumptions():
    @_set_assumptions(commutative=True, negative=False, real=True)
    class MySqrt: ...

    expr = MySqrt()
    assert expr.is_commutative  # type: ignore[attr-defined]
    assert not expr.is_negative  # type: ignore[attr-defined]
    assert expr.is_real  # type: ignore[attr-defined]


def test_unevaluated_expression():
    @unevaluated_expression
    class BreakupMomentum(sp.Expr):
        s: Any
        m1: Any
        m2: Any
        _latex_repr_ = R"q\left({s}\right)"

        def evaluate(self) -> sp.Expr:
            s, m1, m2 = self.args
            return sp.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2))  # type: ignore[operator]

    m0, ma, mb = sp.symbols("m0 m_a m_b")
    expr = BreakupMomentum(m0**2, ma, mb)
    assert expr.s is m0**2
    assert expr.m1 is ma
    assert expr.m2 is mb
    assert expr.is_commutative is True
    args_str = list(inspect.signature(expr.__new__).parameters)
    assert args_str == ["s", "m1", "m2", "args", "evaluate", "kwargs"]
    latex = sp.latex(expr)
    assert latex == R"q\left(m_{0}^{2}\right)"

    q_value = BreakupMomentum(1, m1=0.2, m2=0.4)
    assert isinstance(q_value.s, sp.Integer)
    assert isinstance(q_value.m1, sp.Float)
    assert isinstance(q_value.m2, sp.Float)


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


def test_unevaluated_expression_classvar():
    @unevaluated_expression
    class MyExpr(sp.Expr):
        x: float
        m: ClassVar[int] = 2

        def evaluate(self) -> sp.Expr:
            return self.x**self.m  # type: ignore[return-value]

    x_expr = MyExpr(4)
    assert x_expr.x is sp.Integer(4)
    assert x_expr.m is 2  # noqa: F632

    y_expr = MyExpr(5)
    assert x_expr.doit() == 4**2
    assert y_expr.doit() == 5**2
    MyExpr.m = 3
    assert x_expr.doit() == 4**3
    assert y_expr.doit() == 5**3


def test_unevaluated_expression_default_arg_with_classvar():
    @unevaluated_expression
    class FunkyPower(sp.Expr):
        x: Any
        m: int = 1
        default_return: ClassVar[float | None] = None

        def evaluate(self) -> sp.Expr:
            if self.default_return is None:
                return self.x**self.m
            return sp.sympify(self.default_return)

    x = sp.Symbol("x")
    exprs = (
        FunkyPower(x),
        FunkyPower(x, 2),
        FunkyPower(x, m=3),
    )
    assert exprs[0].doit() == x
    assert exprs[1].doit() == x**2
    assert exprs[2].doit() == x**3
    for expr in exprs:
        assert expr.x is x
        assert isinstance(expr.m, sp.Integer)
        assert expr.default_return is None

    half = sp.Rational(1, 2)
    FunkyPower.default_return = half
    assert exprs[0].doit() == half
    assert exprs[1].doit() == half
    assert exprs[2].doit() == half
    for expr in exprs:
        assert expr.x is x
        assert isinstance(expr.m, sp.Integer)
        assert expr.default_return is half


def test_unevaluated_expression_default_args():
    @unevaluated_expression
    class MyExpr(sp.Expr):
        x: Any
        m: int = 2

        def evaluate(self) -> sp.Expr:
            return self.x**self.m

    expr1 = MyExpr(x=5)
    assert str(expr1) == "MyExpr(5, 2)"
    assert expr1.doit() == 5**2

    expr2 = MyExpr(4, 3)
    assert expr2.doit() == 4**3
