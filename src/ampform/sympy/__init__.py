# cspell:ignore mhash
# pylint: disable=invalid-getnewargs-ex-returned, protected-access
"""Tools that facilitate in building :mod:`sympy` expressions."""

import functools
import itertools
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Type, TypeVar

import sympy as sp
from sympy.printing.latex import LatexPrinter
from sympy.printing.precedence import PRECEDENCE


class UnevaluatedExpression(sp.Expr):
    """Base class for classes that expressions with an ``evaluate()`` method.

    Derive from this class when decorating a class with :func:`implement_expr`
    or :func:`implement_doit_method`. It is important to derive from
    `UnevaluatedExpression`, because an :code:`evaluate()` method has to be
    implemented.
    """

    # https://github.com/sympy/sympy/blob/1.8/sympy/core/basic.py#L74-L77
    __slots__: Tuple[str] = ("_name",)
    _name: Optional[str]
    """Optional instance attribute that can be used in LaTeX representations."""

    def __new__(  # pylint: disable=unused-argument
        cls, *args: Any, name: Optional[str] = None, **hints: Any
    ) -> "UnevaluatedExpression":
        # https://github.com/sympy/sympy/blob/1.8/sympy/core/basic.py#L113-L119
        obj = object.__new__(cls)
        obj._args = args
        obj._assumptions = cls.default_assumptions
        obj._mhash = None
        obj._name = name
        return obj

    def __getnewargs_ex__(self) -> Tuple[tuple, dict]:
        # Pickling support, see
        # https://github.com/sympy/sympy/blob/1.8/sympy/core/basic.py#L124-L126
        args = tuple(self.args)
        kwargs = {"name": self._name}
        return args, kwargs

    @abstractmethod
    def evaluate(self) -> sp.Expr:
        pass

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        """Provide a mathematical Latex representation for notebooks."""
        args = tuple(map(printer._print, self.args))
        name = self.__class__.__name__
        if self._name is not None:
            name = self._name
        return f"{name}{args}"


DecoratedClass = TypeVar("DecoratedClass")
"""`~typing.TypeVar` for decorators like `make_commutative`."""


def implement_expr(
    n_args: int,
) -> Callable[[DecoratedClass], DecoratedClass]:
    """Decorator for classes that derive from `UnevaluatedExpression`.

    Implement a `~object.__new__` and `~sympy.core.basic.Basic.doit` method for
    a class that derives from `~sympy.core.expr.Expr` (via
    `UnevaluatedExpression`).
    """

    def decorator(decorated_class: DecoratedClass) -> DecoratedClass:
        decorated_class = implement_new_method(n_args)(decorated_class)
        decorated_class = implement_doit_method(decorated_class)
        return decorated_class

    return decorator


def implement_new_method(
    n_args: int,
) -> Callable[[DecoratedClass], DecoratedClass]:
    """Implement ``__new__()`` method for an `UnevaluatedExpression` class.

    Implement a `~object.__new__` method for a class that derives from
    `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """

    def decorator(decorated_class: DecoratedClass) -> DecoratedClass:
        def new_method(  # pylint: disable=unused-argument
            cls: Type,
            *args: sp.Symbol,
            evaluate: bool = False,
            **hints: Any,
        ) -> bool:
            if len(args) != n_args:
                raise ValueError(
                    f"{n_args} parameters expected, got {len(args)}"
                )
            args = sp.sympify(args)
            expr = UnevaluatedExpression.__new__(cls, *args)
            if evaluate:
                return expr.evaluate()
            return expr

        decorated_class.__new__ = new_method  # type: ignore[assignment]
        return decorated_class

    return decorator


def implement_doit_method(decorated_class: DecoratedClass) -> DecoratedClass:
    """Implement ``doit()`` method for an `UnevaluatedExpression` class.

    Implement a `~sympy.core.basic.Basic.doit` method for a class that derives
    from `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """

    @functools.wraps(decorated_class.doit)  # type: ignore[attr-defined]
    def doit_method(self: UnevaluatedExpression, deep: bool = True) -> sp.Expr:
        expr = self.evaluate()
        if deep:
            return expr.doit()
        return expr

    decorated_class.doit = doit_method  # type: ignore[attr-defined]
    return decorated_class


def _implement_latex_subscript(  # pyright: reportUnusedFunction=false
    subscript: str,
) -> Callable[[Type[UnevaluatedExpression]], Type[UnevaluatedExpression]]:
    def decorator(
        decorated_class: Type[UnevaluatedExpression],
    ) -> Type[UnevaluatedExpression]:
        # pylint: disable=protected-access, unused-argument
        @functools.wraps(decorated_class.doit)
        def _latex(self: sp.Expr, printer: LatexPrinter, *args: Any) -> str:
            momentum = printer._print(self._momentum)
            if printer._needs_mul_brackets(self._momentum):
                momentum = Rf"\left({momentum}\right)"
            else:
                momentum = Rf"{{{momentum}}}"
            return f"{momentum}_{subscript}"

        decorated_class._latex = _latex  # type: ignore[assignment]
        return decorated_class

    return decorator


def make_commutative(decorated_class: DecoratedClass) -> DecoratedClass:
    decorated_class.is_commutative = True  # type: ignore[attr-defined]
    decorated_class.is_extended_real = True  # type: ignore[attr-defined]
    return decorated_class


def create_expression(
    cls: Type[UnevaluatedExpression],
    *args: Any,
    evaluate: bool = False,
    name: Optional[str] = None,
    **kwargs: Any,
) -> sp.Expr:
    """Helper function for implementing :code:`Expr.__new__`.

    See e.g. source code of `.BlattWeisskopfSquared`.
    """
    args = sp.sympify(args)
    expr = UnevaluatedExpression.__new__(cls, *args, name=name, **kwargs)
    if evaluate:
        return expr.evaluate()
    return expr


def create_symbol_matrix(name: str, m: int, n: int) -> sp.Matrix:
    """Create a `~sympy.matrices.dense.Matrix` with symbols as elements.

    The `~sympy.matrices.expressions.MatrixSymbol` has some issues when one is
    interested in the elements of the matrix. This function instead creates a
    `~sympy.matrices.dense.Matrix` where the elements are
    `~sympy.tensor.indexed.Indexed` instances.

    To convert these `~sympy.tensor.indexed.Indexed` instances to a
    `~sympy.core.symbol.Symbol`, use
    :func:`symplot.substitute_indexed_symbols`.

    >>> create_symbol_matrix("A", m=2, n=3)
    Matrix([
    [A[0, 0], A[0, 1], A[0, 2]],
    [A[1, 0], A[1, 1], A[1, 2]]])
    """
    symbol = sp.IndexedBase(name, shape=(m, n))
    return sp.Matrix([[symbol[i, j] for j in range(n)] for i in range(m)])


@implement_doit_method
class PoolSum(UnevaluatedExpression):
    # pylint: disable=line-too-long
    r"""Sum over indices where the values are taken from a domain set.

    >>> i, j, m, n = sp.symbols("i j m n")
    >>> expr = PoolSum(i**m + j**n, (i, [-0.5, +0.5]), (j, [2, 4, 5]))
    >>> expr
    PoolSum(i**m + j**n, (i, [-0.5, 0.5]), (j, [2, 4, 5]))
    >>> sp.latex(expr)
    '\\sum_{\\substack{i\\in\\left\\{-0.5,0.5\\right\\}\\\\j\\in\\left\\{2,4,5\\right\\}}}i^{m} + j^{n}'
    >>> expr.doit()
    3*(-0.5)**m + 3*0.5**m + 2*2**n + 2*4**n + 2*5**n
    """

    precedence = PRECEDENCE["Mul"]

    def __new__(
        cls,
        expression: sp.Expr,
        *indices: Tuple[sp.Symbol, List[float]],
        **hints: Any,
    ) -> "PoolSum":
        return create_expression(cls, expression, *indices, **hints)

    @property
    def expression(self) -> sp.Expr:
        return self.args[0]

    @property
    def indices(self) -> List[Tuple[sp.Symbol, List[float]]]:
        return self.args[1:]

    def evaluate(self) -> sp.Expr:
        indices = dict(self.indices)
        return sp.Add(
            *[
                self.expression.subs(zip(indices, combi))
                for combi in itertools.product(*indices.values())
            ]
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        expression = printer._print(self.expression)
        indices = dict(self.indices)
        index_ranges: List[str] = []
        for idx, values in indices.items():
            idx_values = ",".join(map(printer._print, values))
            index_ranges.append(
                Rf"{printer._print(idx)}\in\left\{{{idx_values}\right\}}"
            )
        substack = R"\\".join(index_ranges)
        return Rf"\sum_{{\substack{{{substack}}}}}{expression}"
