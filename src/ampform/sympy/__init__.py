# cspell:ignore mhash
# pylint: disable=invalid-getnewargs-ex-returned, protected-access
"""Tools that facilitate in building :mod:`sympy` expressions."""

import functools
import itertools
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import sympy as sp
from sympy.printing.latex import LatexPrinter
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.precedence import PRECEDENCE


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


@implement_doit_method
class PoolSum(UnevaluatedExpression):
    # pylint: disable=line-too-long
    r"""Sum over indices where the values are taken from a domain set.

    >>> i, j, m, n = sp.symbols("i j m n")
    >>> expr = PoolSum(i**m + j**n, (i, (-1, 0, +1)), (j, (2, 4, 5)))
    >>> expr
    PoolSum(i**m + j**n, (i, (-1, 0, 1)), (j, (2, 4, 5)))
    >>> print(sp.latex(expr))
    \sum_{i=-1}^{1} \sum_{j\in\left\{2,4,5\right\}}{i^{m} + j^{n}}
    >>> expr.doit()
    3*(-1)**m + 3*0**m + 3*2**n + 3*4**n + 3*5**n + 3
    """

    precedence = PRECEDENCE["Mul"]

    def __new__(
        cls,
        expression: sp.Expr,
        *indices: Tuple[sp.Symbol, Iterable[sp.Float]],
        **hints: Any,
    ) -> "PoolSum":
        converted_indices = []
        for idx_symbol, values in indices:
            values = tuple(values)
            if len(values) == 0:
                raise ValueError(f"No values provided for index {idx_symbol}")
            converted_indices.append((idx_symbol, values))
        return create_expression(cls, expression, *converted_indices, **hints)

    @property
    def expression(self) -> sp.Expr:
        return self.args[0]

    @property
    def indices(self) -> List[Tuple[sp.Symbol, Tuple[sp.Float, ...]]]:
        return self.args[1:]

    @property
    def free_symbols(self) -> Set[sp.Symbol]:
        return super().free_symbols - {s for s, _ in self.indices}

    def evaluate(self) -> sp.Expr:
        indices = {symbol: tuple(values) for symbol, values in self.indices}
        return sp.Add(
            *[
                self.expression.subs(zip(indices, combi))
                for combi in itertools.product(*indices.values())
            ]
        )

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        indices = dict(self.indices)
        sum_symbols: List[str] = []
        for idx, values in indices.items():
            sum_symbols.append(_render_sum_symbol(printer, idx, values))
        expression = printer._print(self.expression)
        return R" ".join(sum_symbols) + f"{{{expression}}}"

    def cleanup(self) -> Union[sp.Expr, "PoolSum"]:
        """Remove redundant summations, like indices with one or no value.

        >>> x, i = sp.symbols("x i")
        >>> PoolSum(x**i, (i, [0, 1, 2])).cleanup().doit()
        x**2 + x + 1
        >>> PoolSum(x, (i, [0, 1, 2])).cleanup()
        x
        >>> PoolSum(x).cleanup()
        x
        >>> PoolSum(x**i, (i, [0])).cleanup()
        1
        """
        substitutions = {}
        new_indices = []
        for idx, values in self.indices:
            if idx not in self.expression.free_symbols:
                continue
            if len(values) == 0:
                continue
            if len(values) == 1:
                substitutions[idx] = values[0]
            else:
                new_indices.append((idx, values))
        new_expression = self.expression.xreplace(substitutions)
        if len(new_indices) == 0:
            return new_expression
        return PoolSum(new_expression, *new_indices)


def _render_sum_symbol(
    printer: LatexPrinter, idx: sp.Symbol, values: Sequence[float]
) -> str:
    if len(values) == 0:
        return ""
    idx = printer._print(idx)
    if len(values) == 1:
        value = values[0]
        return Rf"\sum_{{{idx}={value}}}"
    if _is_regular_series(values):
        sorted_values = sorted(values)
        first_value = sorted_values[0]
        last_value = sorted_values[-1]
        return Rf"\sum_{{{idx}={first_value}}}^{{{last_value}}}"
    idx_values = ",".join(map(printer._print, values))
    return Rf"\sum_{{{idx}\in\left\{{{idx_values}\right\}}}}"


def _is_regular_series(values: Sequence[float]) -> bool:
    """Check whether a set of values is a series with unit distances.

    >>> _is_regular_series([0, 1, 2])
    True
    >>> _is_regular_series([-0.5, +0.5])
    True
    >>> _is_regular_series([+0.5, -0.5, 1.5])
    True
    >>> _is_regular_series([-1, +1])
    False
    >>> _is_regular_series([1])
    False
    >>> _is_regular_series([])
    False
    """
    if len(values) <= 1:
        return False
    sorted_values = sorted(values)
    for val, next_val in zip(sorted_values, sorted_values[1:]):
        difference = float(next_val - val)
        if difference != 1.0:
            return False
    return True
