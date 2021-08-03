"""Tools for defining lineshapes with `sympy`."""

from abc import abstractmethod
from typing import Any, Callable, Optional, Type

import sympy as sp
from sympy.printing.latex import LatexPrinter


class UnevaluatedExpression(sp.Expr):
    """Base class for classes that expressions with an ``evaluate()`` method.

    Derive from this class when decorating a class with :func:`implement_expr`
    or :func:`implement_doit_method`. It is important to derive from
    `UnevaluatedExpression`, because an :code:`evaluate()` method has to be
    implemented.
    """

    @abstractmethod
    def evaluate(self) -> sp.Expr:
        pass

    @abstractmethod
    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        """Provide a mathematical Latex representation for notebooks."""
        args = tuple(map(printer._print, self.args))
        return f"{self.__class__.__name__}{args}"


def implement_expr(
    n_args: int,
) -> Callable[[Type[UnevaluatedExpression]], Type[UnevaluatedExpression]]:
    """Decorator for classes that derive from `UnevaluatedExpression`.

    Implement a `~object.__new__` and `~sympy.core.basic.Basic.doit` method for
    a class that derives from `~sympy.core.expr.Expr` (via
    `UnevaluatedExpression`).
    """

    def decorator(
        decorated_class: Type[UnevaluatedExpression],
    ) -> Type[UnevaluatedExpression]:
        decorated_class = implement_new_method(n_args)(decorated_class)
        decorated_class = implement_doit_method()(decorated_class)
        return decorated_class

    return decorator


def implement_new_method(
    n_args: int,
) -> Callable[[Type[UnevaluatedExpression]], Type[UnevaluatedExpression]]:
    """Implement ``__new__()`` method for an `UnevaluatedExpression` class.

    Implement a `~object.__new__` method for a class that derives from
    `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """

    def decorator(
        decorated_class: Type[UnevaluatedExpression],
    ) -> Type[UnevaluatedExpression]:
        def new_method(
            cls: Type,
            *args: sp.Symbol,
            **hints: Any,
        ) -> bool:
            if len(args) != n_args:
                raise ValueError(
                    f"{n_args} parameters expected, got {len(args)}"
                )
            args = sp.sympify(args)
            evaluate = hints.get("evaluate", False)
            if evaluate:
                return sp.Expr.__new__(cls, *args).evaluate()  # type: ignore  # pylint: disable=no-member
            return sp.Expr.__new__(cls, *args)

        decorated_class.__new__ = new_method  # type: ignore
        return decorated_class

    return decorator


def implement_doit_method() -> Callable[
    [Type[UnevaluatedExpression]], Type[UnevaluatedExpression]
]:
    """Implement ``doit()`` method for an `UnevaluatedExpression` class.

    Implement a `~sympy.core.basic.Basic.doit` method for a class that derives
    from `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """

    def decorator(
        decorated_class: Type[UnevaluatedExpression],
    ) -> Type[UnevaluatedExpression]:
        def doit_method(self: Any, **hints: Any) -> sp.Expr:
            return type(self)(*self.args, **hints, evaluate=True)

        decorated_class.doit = doit_method
        return decorated_class

    return decorator


def create_expression(
    cls: Type[UnevaluatedExpression], evaluate: bool, *args: Any, **kwargs: Any
) -> sp.Expr:
    """Helper function for implementing :code:`Expr.__new__`.

    See e.g. source code of `.BlattWeisskopfSquared`.
    """
    # pylint: disable=no-member
    deep: Optional[bool] = kwargs.pop("deep", None)
    expr = sp.Expr.__new__(cls, *args, **kwargs)
    if evaluate:
        expr = expr.evaluate()
    if deep:
        expr = expr.doit(deep=deep)
    return expr
