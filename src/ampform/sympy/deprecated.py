"""Deprecated classes and functions for constructing unevaluated expressions.

.. deprecated:: 0.15.0
"""

from __future__ import annotations

import functools
import sys
from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, TypeVar
from warnings import warn

import sympy as sp

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter


class UnevaluatedExpression(sp.Expr):
    """Base class for expression classes with an :meth:`evaluate` method.

    Deriving from `~sympy.core.expr.Expr` allows us to keep expression trees condense
    before unfolding them with their `~sympy.core.basic.Basic.doit` method. This allows
    us to:

    1. condense the LaTeX representation of an expression tree by providing a custom
       :meth:`_latex` method.
    2. overwrite its printer methods (see `.NumPyPrintable` and e.g.
       :doc:`compwa-report:001/index`).

    The `UnevaluatedExpression` base class makes implementations of its derived classes
    more secure by enforcing the developer to provide implementations for these methods,
    so that SymPy mechanisms work correctly. Decorators like :func:`implement_expr` and
    :func:`implement_doit_method` provide convenient means to implement the missing
    methods.

    .. autolink-preface::

        import sympy as sp
        from ampform.sympy import UnevaluatedExpression, create_expression

    .. automethod:: __new__
    .. automethod:: evaluate
    .. automethod:: _latex
    """

    # https://github.com/sympy/sympy/blob/1.8/sympy/core/basic.py#L74-L77
    __slots__: tuple[str] = ("_name",)
    _name: str | None
    """Optional instance attribute that can be used in LaTeX representations."""

    def __init_subclass__(cls, **kwargs):
        warn(
            f"{cls.__name__} is deprecated, use the"
            " @ampform.sympy.unevaluated_expression decorator instead",
            category=DeprecationWarning,
            stacklevel=1,
        )
        super().__init_subclass__(**kwargs)

    @override
    def __new__(
        cls: type[DecoratedClass],
        *args,
        name: str | None = None,
        **hints,
    ) -> DecoratedClass:
        """Constructor for a class derived from `UnevaluatedExpression`.

        This :meth:`~object.__new__` method correctly sets the
        `~sympy.core.basic.Basic.args`, assumptions etc. Overwrite it in order to
        further specify its signature. The function :func:`create_expression` can be
        used in its implementation, like so:

        >>> class MyExpression(UnevaluatedExpression):
        ...     def __new__(
        ...         cls, x: sp.Symbol, y: sp.Symbol, n: int, **hints
        ...     ) -> "MyExpression":
        ...         return create_expression(cls, x, y, n, **hints)
        ...
        ...     def evaluate(self) -> sp.Expr:
        ...         x, y, n = self.args
        ...         return (x + y) ** n
        >>> x, y = sp.symbols("x y")
        >>> expr = MyExpression(x, y, n=3)
        >>> expr
        MyExpression(x, y, 3)
        >>> expr.evaluate()
        (x + y)**3
        """
        # https://github.com/sympy/sympy/blob/1.8/sympy/core/basic.py#L113-L119
        obj = object.__new__(cls)
        obj._args = args  # noqa: SLF001
        obj._assumptions = cls.default_assumptions  # type: ignore[attr-defined]  # noqa: SLF001
        obj._mhash = None  # cspell:ignore mhash  # noqa: SLF001
        obj._name = name  # noqa: SLF001
        return obj

    def __getnewargs_ex__(self) -> tuple[tuple, dict]:
        # Pickling support, see
        # https://github.com/sympy/sympy/blob/1.8/sympy/core/basic.py#L124-L126
        args = tuple(self.args)
        kwargs = {"name": self._name}
        return args, kwargs

    @override  # type:ignore[misc]
    def _hashable_content(self) -> tuple:
        # https://github.com/sympy/sympy/blob/1.10/sympy/core/basic.py#L157-L165
        # name is converted to string because unstable hash for None
        return (*super()._hashable_content(), str(self._name))

    @abstractmethod
    def evaluate(self) -> sp.Expr:
        """Evaluate and 'unfold' this `UnevaluatedExpression` by one level.

        >>> from ampform.dynamics import BreakupMomentumSquared
        >>> s, m1, m2 = sp.symbols("s m1 m2")
        >>> expr = BreakupMomentumSquared(s, m1, m2)
        >>> expr
        BreakupMomentumSquared(s, m1, m2)
        >>> expr.evaluate()
        (s - (m1 - m2)**2)*(s - (m1 + m2)**2)/(4*s)
        >>> expr.doit(deep=False)
        (s - (m1 - m2)**2)*(s - (m1 + m2)**2)/(4*s)

        .. note:: When decorating this class with :func:`implement_doit_method`,
            its :meth:`evaluate` method is equivalent to
            :meth:`~sympy.core.basic.Basic.doit` with :code:`deep=False`.
        """

    def _latex(self, printer: LatexPrinter, *args) -> str:
        r"""Provide a mathematical Latex representation for pretty printing.

        >>> from ampform.dynamics import BreakupMomentumSquared
        >>> s, m1 = sp.symbols("s m1")
        >>> expr = BreakupMomentumSquared(s, m1, m1)
        >>> print(sp.latex(expr))
        q^2\left(s\right)
        >>> print(sp.latex(expr.doit()))
        - m_{1}^{2} + \frac{s}{4}
        """
        args = tuple(map(printer._print, self.args))
        name = type(self).__name__
        if self._name is not None:
            name = self._name
        return f"{name}{args}"


DecoratedClass = TypeVar("DecoratedClass", bound=UnevaluatedExpression)
"""`~typing.TypeVar` for decorators like :func:`implement_doit_method`."""


def implement_expr(
    n_args: int,
) -> Callable[[type[DecoratedClass]], type[DecoratedClass]]:
    """Decorator for classes that derive from `UnevaluatedExpression`.

    Implement a :meth:`~object.__new__` and :meth:`~sympy.core.basic.Basic.doit` method
    for a class that derives from `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """
    warn(
        "@implement_expr is deprecated, use the @ampform.sympy.unevaluated_expression"
        " decorator instead",
        category=DeprecationWarning,
        stacklevel=1,
    )

    def decorator(
        decorated_class: type[DecoratedClass],
    ) -> type[DecoratedClass]:
        decorated_class = implement_new_method(n_args)(decorated_class)
        return implement_doit_method(decorated_class)

    return decorator


def implement_new_method(
    n_args: int,
) -> Callable[[type[DecoratedClass]], type[DecoratedClass]]:
    """Implement :meth:`UnevaluatedExpression.__new__` on a derived class.

    Implement a :meth:`~object.__new__` method for a class that derives from
    `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """
    warn(
        "@implement_new_method is deprecated, use the"
        " @ampform.sympy.unevaluated_expression decorator instead",
        category=DeprecationWarning,
        stacklevel=1,
    )

    def decorator(
        decorated_class: type[DecoratedClass],
    ) -> type[DecoratedClass]:
        def new_method(
            cls: type[DecoratedClass],
            *args: sp.Symbol,
            evaluate: bool = False,
            **hints,
        ) -> DecoratedClass:
            if len(args) != n_args:
                msg = f"{n_args} parameters expected, got {len(args)}"
                raise ValueError(msg)
            args = sp.sympify(args)
            expr = UnevaluatedExpression.__new__(cls, *args)
            if evaluate:
                return expr.evaluate()  # type: ignore[return-value]
            return expr

        decorated_class.__new__ = new_method  # type: ignore[assignment]
        return decorated_class

    return decorator


def implement_doit_method(
    decorated_class: type[DecoratedClass],
) -> type[DecoratedClass]:
    """Implement ``doit()`` method for an `UnevaluatedExpression` class.

    Implement a :meth:`~sympy.core.basic.Basic.doit` method for a class that derives
    from `~sympy.core.expr.Expr` (via `UnevaluatedExpression`). A
    :meth:`~sympy.core.basic.Basic.doit` method is an extension of an
    :meth:`~.UnevaluatedExpression.evaluate` method in the sense that it can work
    recursively on deeper expression trees.
    """
    warn(
        "@implement_doit_method is deprecated, use the"
        " @ampform.sympy.unevaluated_expression decorator instead",
        category=DeprecationWarning,
        stacklevel=1,
    )

    @functools.wraps(decorated_class.doit)  # type: ignore[attr-defined]
    def doit_method(self: UnevaluatedExpression, deep: bool = True) -> sp.Expr:
        expr = self.evaluate()
        if deep:
            return expr.doit()
        return expr

    decorated_class.doit = doit_method  # type: ignore[assignment]
    return decorated_class


DecoratedExpr = TypeVar("DecoratedExpr", bound=sp.Expr)
"""`~typing.TypeVar` for decorators like :func:`make_commutative`."""


def make_commutative(
    decorated_class: type[DecoratedExpr],
) -> type[DecoratedExpr]:
    """Set commutative and 'extended real' assumptions on expression class.

    .. seealso:: :doc:`sympy:guides/assumptions`
    """
    warn(
        "@make_commutative is deprecated, use the @ampform.sympy.unevaluated_expression"
        " decorator instead with commutative=True",
        category=DeprecationWarning,
        stacklevel=1,
    )
    decorated_class.is_commutative = True  # type: ignore[attr-defined]
    decorated_class.is_extended_real = True  # type: ignore[attr-defined]
    return decorated_class


def create_expression(
    cls: type[DecoratedExpr],
    *args,
    evaluate: bool = False,
    name: str | None = None,
    **kwargs,
) -> DecoratedExpr:
    """Helper function for implementing `UnevaluatedExpression.__new__`."""
    warn(
        "create_expression() is deprecated, construct the class with the"
        " @ampform.sympy.unevaluated_expression decorator instead",
        category=DeprecationWarning,
        stacklevel=1,
    )
    args = sp.sympify(args)
    if issubclass(cls, UnevaluatedExpression):
        expr = UnevaluatedExpression.__new__(cls, *args, name=name, **kwargs)
        if evaluate:
            return expr.evaluate()  # type: ignore[return-value]
        return expr  # type: ignore[return-value]
    return sp.Expr.__new__(cls, *args, **kwargs)  # type: ignore[return-value]
