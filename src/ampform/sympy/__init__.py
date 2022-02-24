# cspell:ignore mhash
# pylint: disable=invalid-getnewargs-ex-returned
"""Tools that facilitate in building :mod:`sympy` expressions."""

import functools
from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

import sympy as sp
from sympy.printing.latex import LatexPrinter
from sympy.printing.numpy import NumPyPrinter


class UnevaluatedExpression(sp.Expr):
    """Base class for expression classes with an :meth:`evaluate` method.

    Deriving from `~sympy.core.expr.Expr` allows us to keep expression trees
    condense before unfolding them with their `~sympy.core.basic.Basic.doit`
    method. This allows us to:

    1. condense the LaTeX representation of an expression tree by providing a
       custom :meth:`_latex` method.
    2. overwrite its printer methods (see `NumPyPrintable` and e.g.
       :doc:`compwa-org:report/001`).

    The `UnevaluatedExpression` base class makes implementations of its derived
    classes more secure by enforcing the developer to provide implementations
    for these methods, so that SymPy mechanisms work correctly. Decorators like
    :func:`implement_expr` and :func:`implement_doit_method` provide convenient
    means to implement the missing methods.

    .. autolink-preface::

        import sympy as sp
        from ampform.sympy import UnevaluatedExpression, create_expression

    .. automethod:: __new__
    .. automethod:: evaluate
    .. automethod:: _latex
    """

    # https://github.com/sympy/sympy/blob/1.8/sympy/core/basic.py#L74-L77
    __slots__: Tuple[str] = ("_name",)
    _name: Optional[str]
    """Optional instance attribute that can be used in LaTeX representations."""

    def __new__(  # pylint: disable=unused-argument
        cls: Type["DecoratedClass"],
        *args: Any,
        name: Optional[str] = None,
        **hints: Any,
    ) -> "DecoratedClass":
        """Constructor for a class derived from `UnevaluatedExpression`.

        This :meth:`~object.__new__` method correctly sets the
        `~sympy.core.basic.Basic.args`, assumptions etc. Overwrite it in order
        to further specify its signature. The function
        :func:`create_expression` can be used in its implementation, like so:

        >>> class MyExpression(UnevaluatedExpression):
        ...    def __new__(
        ...        cls, x: sp.Symbol, y: sp.Symbol, n: int, **hints
        ...    ) -> "MyExpression":
        ...        return create_expression(cls, x, y, n, **hints)
        ...
        ...    def evaluate(self) -> sp.Expr:
        ...        x, y, n = self.args
        ...        return (x + y)**n
        ...
        >>> x, y = sp.symbols("x y")
        >>> expr = MyExpression(x, y, n=3)
        >>> expr
        MyExpression(x, y, 3)
        >>> expr.evaluate()
        (x + y)**3
        """
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
        """Evaluate and 'unfold' this `UnevaluatedExpression` by one level.

        >>> from ampform.dynamics import BreakupMomentumSquared
        >>> issubclass(BreakupMomentumSquared, UnevaluatedExpression)
        True
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

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        r"""Provide a mathematical Latex representation for pretty printing.

        >>> from ampform.dynamics import BreakupMomentumSquared
        >>> issubclass(BreakupMomentumSquared, UnevaluatedExpression)
        True
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


class NumPyPrintable(sp.Expr):
    r"""`~sympy.core.expr.Expr` class that can lambdify to NumPy code.

    This interface for classes that derive from `sympy.Expr
    <sympy.core.expr.Expr>` enforce the implementation of a :meth:`_numpycode`
    method in case the class does not correctly
    :func:`~sympy.utilities.lambdify.lambdify` to NumPy code. For more info on
    SymPy printers, see :doc:`sympy:modules/printing`.

    Several computational frameworks try to converge their interface to that of
    NumPy. See for instance `TensorFlow's NumPy API
    <https://www.tensorflow.org/guide/tf_numpy>`_ and `jax.numpy
    <https://jax.readthedocs.io/en/latest/jax.numpy.html>`_. This fact is used
    in `TensorWaves <https://tensorwaves.rtfd.io>`_ to
    :func:`~sympy.utilities.lambdify.lambdify` SymPy expressions to these
    different backends with the same lambdification code.

    .. note:: This interface differs from `UnevaluatedExpression` in that it
        **should not** implement an :meth:`.evaluate` (and therefore a
        :meth:`~sympy.core.basic.Basic.doit`) method.


    .. warning:: The implemented :meth:`_numpycode` method should countain as
        little SymPy computations as possible. Instead, it should get most
        information from its construction `~sympy.core.basic.Basic.args`, so
        that SymPy can use printer tricks like
        :func:`~sympy.simplify.cse_main.cse`, prior expanding with
        :meth:`~sympy.core.basic.Basic.doit`, and other simplifications that
        can make the generated code shorter. An example is the `.BoostZMatrix`
        class, which takes :math:`\beta` as input instead of the
        `.FourMomentumSymbol` from which :math:`\beta` is computed.

    .. automethod:: _numpycode
    """

    @abstractmethod
    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        """Lambdify this `NumPyPrintable` class to NumPy code."""


DecoratedClass = TypeVar("DecoratedClass", bound=UnevaluatedExpression)
"""`~typing.TypeVar` for decorators like :func:`make_commutative`."""


def implement_expr(
    n_args: int,
) -> Callable[[Type[DecoratedClass]], Type[DecoratedClass]]:
    """Decorator for classes that derive from `UnevaluatedExpression`.

    Implement a :meth:`~object.__new__` and
    :meth:`~sympy.core.basic.Basic.doit` method for a class that derives from
    `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """

    def decorator(
        decorated_class: Type[DecoratedClass],
    ) -> Type[DecoratedClass]:
        decorated_class = implement_new_method(n_args)(decorated_class)
        decorated_class = implement_doit_method(decorated_class)
        return decorated_class

    return decorator


def implement_new_method(
    n_args: int,
) -> Callable[[Type[DecoratedClass]], Type[DecoratedClass]]:
    """Implement :meth:`UnevaluatedExpression.__new__` on a derived class.

    Implement a :meth:`~object.__new__` method for a class that derives from
    `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """

    def decorator(
        decorated_class: Type[DecoratedClass],
    ) -> Type[DecoratedClass]:
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


def implement_doit_method(
    decorated_class: Type[DecoratedClass],
) -> Type[DecoratedClass]:
    """Implement ``doit()`` method for an `UnevaluatedExpression` class.

    Implement a :meth:`~sympy.core.basic.Basic.doit` method for a class that
    derives from `~sympy.core.expr.Expr` (via `UnevaluatedExpression`). A
    :meth:`~sympy.core.basic.Basic.doit` method is an extension of an
    :meth:`~.UnevaluatedExpression.evaluate` method in the sense that it can
    work recursively on deeper expression trees.
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


def make_commutative(
    decorated_class: Type[DecoratedClass],
) -> Type[DecoratedClass]:
    """Set commutative and 'extended real' assumptions on expression class.

    .. seealso:: :doc:`sympy:guides/assumptions`
    """
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
    """Helper function for implementing `UnevaluatedExpression.__new__`."""
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
