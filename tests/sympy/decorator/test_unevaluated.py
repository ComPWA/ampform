from __future__ import annotations

import inspect
from typing import Any, ClassVar

import pytest
import sympy as sp

from ampform.sympy._decorator import argument, unevaluated


def test_classvar_behavior():
    @unevaluated
    class MyExpr(sp.Expr):
        x: float
        m: ClassVar[int] = 2
        class_name = "MyExpr"

        def evaluate(self) -> sp.Expr:
            return self.x**self.m  # type: ignore[return-value]

    x_expr = MyExpr(4)
    assert x_expr.x is sp.Integer(4)
    assert x_expr.m is 2  # noqa: F632

    y_expr = MyExpr(5)
    assert x_expr.doit() == 4**2
    assert y_expr.doit() == 5**2
    assert x_expr.class_name == "MyExpr"
    assert y_expr.class_name == "MyExpr"
    MyExpr.m = 3
    new_name = "different name"
    MyExpr.class_name = new_name
    assert x_expr.doit() == 4**3
    assert y_expr.doit() == 5**3
    assert x_expr.class_name == new_name
    assert y_expr.class_name == new_name


def test_construction_non_sympy_attributes():
    class CannotBeSympified: ...

    @unevaluated(implement_doit=False)
    class MyExpr(sp.Expr):
        sympifiable: Any
        non_sympy: CannotBeSympified = argument(sympify=False)

    obj = CannotBeSympified()
    expr = MyExpr(
        sympifiable=3,
        non_sympy=obj,
    )
    assert expr.sympifiable is not 3  # noqa: F632
    assert expr.sympifiable is sp.Integer(3)
    assert expr.non_sympy is obj


def test_default_argument():
    @unevaluated
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


def test_default_argument_with_classvar():
    @unevaluated
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


def test_hashable_with_classes():
    class CannotBeSympified: ...

    @unevaluated(implement_doit=False)
    class MyExpr(sp.Expr):
        x: Any
        typ: type[CannotBeSympified] = argument(sympify=False)

    x = sp.Symbol("x")
    expr = MyExpr(x, typ=CannotBeSympified)
    assert expr._hashable_content() == (
        x,
        f"{CannotBeSympified.__module__}.{CannotBeSympified.__qualname__}",
    )


def test_latex_repr_typo_warning():
    with pytest.warns(
        UserWarning,
        match=r"Class defines a _latex_repr attribute, but it should be _latex_repr_",
    ):

        @unevaluated(real=False)
        class MyExpr(sp.Expr):  # pyright: ignore[reportUnusedClass]
            x: sp.Symbol
            _latex_repr = "<The attribute name is wrong>"

            def evaluate(self) -> sp.Expr:
                return self.x


def test_no_implement_doit():
    @unevaluated(implement_doit=False)
    class Squared(sp.Expr):
        x: Any

        def evaluate(self) -> sp.Expr:
            return self.x**2

    sqrt = Squared(2)
    assert str(sqrt) == "Squared(2)"
    assert str(sqrt.doit()) == "Squared(2)"

    @unevaluated(complex=True, implement_doit=False)
    class MySqrt(sp.Expr):
        x: Any

    expr = MySqrt(-1)
    assert expr.is_commutative
    assert expr.is_complex  # type: ignore[attr-defined]


def test_non_symbols_construction():
    @unevaluated
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


def test_subs_with_non_sympy_attributes():
    class Protocol: ...

    @unevaluated(implement_doit=False)
    class MyExpr(sp.Expr):
        x: Any
        protocol: type[Protocol] = argument(default=Protocol, sympify=False)

    x, y = sp.symbols("x y")
    expr = MyExpr(x)
    replaced_expr: MyExpr = expr.subs(x, y)
    assert replaced_expr.x is not x
    assert replaced_expr.x is y
    assert replaced_expr.protocol is Protocol


def test_xreplace_with_non_sympy_attributes():
    class Protocol: ...

    class Protocol1(Protocol): ...

    class Protocol2(Protocol): ...

    @unevaluated(implement_doit=False)
    class MyExpr(sp.Expr):
        x: Any
        protocol: type[Protocol] = argument(default=Protocol1, sympify=False)

    x, y = sp.symbols("x y")
    expr = MyExpr(x)
    replaced_expr: MyExpr = expr.xreplace({x: y, Protocol1: Protocol2})
    assert replaced_expr.x is not x
    assert replaced_expr.x is y
    assert replaced_expr.protocol is not Protocol1
    assert replaced_expr.protocol is Protocol2
