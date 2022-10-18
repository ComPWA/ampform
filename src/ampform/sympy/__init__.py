# cspell:ignore mhash
# pylint: disable=invalid-getnewargs-ex-returned, protected-access, W0223
# https://stackoverflow.com/a/22224042
"""Tools that facilitate in building :mod:`sympy` expressions."""
from __future__ import annotations

import functools
import hashlib
import itertools
import logging
import os
import pickle
from abc import abstractmethod
from os.path import abspath, dirname, expanduser
from textwrap import dedent
from typing import Callable, Iterable, Sequence, SupportsFloat, TypeVar

import sympy as sp
from sympy.printing.latex import LatexPrinter
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.precedence import PRECEDENCE

_LOGGER = logging.getLogger(__name__)


class UnevaluatedExpression(sp.Expr):
    """Base class for expression classes with an :meth:`evaluate` method.

    Deriving from `~sympy.core.expr.Expr` allows us to keep expression trees condense
    before unfolding them with their `~sympy.core.basic.Basic.doit` method. This allows
    us to:

    1. condense the LaTeX representation of an expression tree by providing a custom
       :meth:`_latex` method.
    2. overwrite its printer methods (see `NumPyPrintable` and e.g.
       :doc:`compwa-org:report/001`).

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

    def __new__(  # pylint: disable=unused-argument
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
        obj._assumptions = cls.default_assumptions  # type: ignore[attr-defined]
        obj._mhash = None
        obj._name = name
        return obj

    def __getnewargs_ex__(self) -> tuple[tuple, dict]:
        # Pickling support, see
        # https://github.com/sympy/sympy/blob/1.8/sympy/core/basic.py#L124-L126
        args = tuple(self.args)
        kwargs = {"name": self._name}
        return args, kwargs

    def _hashable_content(self) -> tuple:
        # https://github.com/sympy/sympy/blob/1.10/sympy/core/basic.py#L157-L165
        # name is converted to string because unstable hash for None
        return (*super()._hashable_content(), str(self._name))

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

    def _latex(self, printer: LatexPrinter, *args) -> str:
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

    This interface for classes that derive from `sympy.Expr <sympy.core.expr.Expr>`
    enforce the implementation of a :meth:`_numpycode` method in case the class does not
    correctly :func:`~sympy.utilities.lambdify.lambdify` to NumPy code. For more info on
    SymPy printers, see :doc:`sympy:modules/printing`.

    Several computational frameworks try to converge their interface to that of NumPy.
    See for instance `TensorFlow's NumPy API
    <https://www.tensorflow.org/guide/tf_numpy>`_ and `jax.numpy
    <https://jax.readthedocs.io/en/latest/jax.numpy.html>`_. This fact is used in
    `TensorWaves <https://tensorwaves.rtfd.io>`_ to
    :func:`~sympy.utilities.lambdify.lambdify` SymPy expressions to these different
    backends with the same lambdification code.

    .. note:: This interface differs from `UnevaluatedExpression` in that it **should
        not** implement an :meth:`.evaluate` (and therefore a
        :meth:`~sympy.core.basic.Basic.doit`) method.


    .. warning:: The implemented :meth:`_numpycode` method should countain as little
        SymPy computations as possible. Instead, it should get most information from its
        construction `~sympy.core.basic.Basic.args`, so that SymPy can use printer
        tricks like :func:`~sympy.simplify.cse_main.cse`, prior expanding with
        :meth:`~sympy.core.basic.Basic.doit`, and other simplifications that can make
        the generated code shorter. An example is the `.BoostZMatrix` class, which takes
        :math:`\beta` as input instead of the `.FourMomentumSymbol` from which
        :math:`\beta` is computed.

    .. automethod:: _numpycode
    """

    @abstractmethod
    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        """Lambdify this `NumPyPrintable` class to NumPy code."""


DecoratedClass = TypeVar("DecoratedClass", bound=UnevaluatedExpression)
"""`~typing.TypeVar` for decorators like :func:`implement_doit_method`."""


def implement_expr(
    n_args: int,
) -> Callable[[type[DecoratedClass]], type[DecoratedClass]]:
    """Decorator for classes that derive from `UnevaluatedExpression`.

    Implement a :meth:`~object.__new__` and :meth:`~sympy.core.basic.Basic.doit` method
    for a class that derives from `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """

    def decorator(
        decorated_class: type[DecoratedClass],
    ) -> type[DecoratedClass]:
        decorated_class = implement_new_method(n_args)(decorated_class)
        decorated_class = implement_doit_method(decorated_class)
        return decorated_class

    return decorator


def implement_new_method(
    n_args: int,
) -> Callable[[type[DecoratedClass]], type[DecoratedClass]]:
    """Implement :meth:`UnevaluatedExpression.__new__` on a derived class.

    Implement a :meth:`~object.__new__` method for a class that derives from
    `~sympy.core.expr.Expr` (via `UnevaluatedExpression`).
    """

    def decorator(
        decorated_class: type[DecoratedClass],
    ) -> type[DecoratedClass]:
        def new_method(  # pylint: disable=unused-argument
            cls: type[DecoratedClass],
            *args: sp.Symbol,
            evaluate: bool = False,
            **hints,
        ) -> DecoratedClass:
            if len(args) != n_args:
                raise ValueError(f"{n_args} parameters expected, got {len(args)}")
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

    @functools.wraps(decorated_class.doit)  # type: ignore[attr-defined]
    def doit_method(self: UnevaluatedExpression, deep: bool = True) -> sp.Expr:
        expr = self.evaluate()
        if deep:
            return expr.doit()
        return expr

    decorated_class.doit = doit_method  # type: ignore[assignment]
    return decorated_class


def _implement_latex_subscript(  # pyright: reportUnusedFunction=false
    subscript: str,
) -> Callable[[type[UnevaluatedExpression]], type[UnevaluatedExpression]]:
    def decorator(
        decorated_class: type[UnevaluatedExpression],
    ) -> type[UnevaluatedExpression]:
        # pylint: disable=protected-access, unused-argument
        @functools.wraps(decorated_class.doit)
        def _latex(self: sp.Expr, printer: LatexPrinter, *args) -> str:
            momentum = printer._print(self._momentum)  # type: ignore[attr-defined]
            if printer._needs_mul_brackets(self._momentum):  # type: ignore[attr-defined]
                momentum = Rf"\left({momentum}\right)"
            else:
                momentum = Rf"{{{momentum}}}"
            return f"{momentum}_{subscript}"

        decorated_class._latex = _latex  # type: ignore[assignment]
        return decorated_class

    return decorator


DecoratedExpr = TypeVar("DecoratedExpr", bound=sp.Expr)
"""`~typing.TypeVar` for decorators like :func:`make_commutative`."""


def make_commutative(
    decorated_class: type[DecoratedExpr],
) -> type[DecoratedExpr]:
    """Set commutative and 'extended real' assumptions on expression class.

    .. seealso:: :doc:`sympy:guides/assumptions`
    """
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
    args = sp.sympify(args)
    if issubclass(cls, UnevaluatedExpression):
        expr = UnevaluatedExpression.__new__(cls, *args, name=name, **kwargs)
        if evaluate:
            return expr.evaluate()  # type: ignore[return-value]
        return expr  # type: ignore[return-value]
    return sp.Expr.__new__(cls, *args, **kwargs)  # type: ignore[return-value]


def create_symbol_matrix(name: str, m: int, n: int) -> sp.MutableDenseMatrix:
    """Create a `~sympy.matrices.dense.Matrix` with symbols as elements.

    The `~sympy.matrices.expressions.MatrixSymbol` has some issues when one is
    interested in the elements of the matrix. This function instead creates a
    `~sympy.matrices.dense.Matrix` where the elements are
    `~sympy.tensor.indexed.Indexed` instances.

    To convert these `~sympy.tensor.indexed.Indexed` instances to a
    `~sympy.core.symbol.Symbol`, use :func:`symplot.substitute_indexed_symbols`.

    >>> create_symbol_matrix("A", m=2, n=3)
    Matrix([
    [A[0, 0], A[0, 1], A[0, 2]],
    [A[1, 0], A[1, 1], A[1, 2]]])
    """
    symbol = sp.IndexedBase(name, shape=(m, n))
    return sp.Matrix([[symbol[i, j] for j in range(n)] for i in range(m)])


@implement_doit_method
class PoolSum(UnevaluatedExpression):
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
        expression,
        *indices: tuple[sp.Symbol, Iterable[sp.Basic]],
        **hints,
    ) -> PoolSum:
        converted_indices = []
        for idx_symbol, values in indices:
            values = tuple(values)
            if len(values) == 0:
                raise ValueError(f"No values provided for index {idx_symbol}")
            converted_indices.append((idx_symbol, values))
        return create_expression(cls, expression, *converted_indices, **hints)

    @property
    def expression(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    @property
    def indices(self) -> list[tuple[sp.Symbol, tuple[sp.Float, ...]]]:
        return self.args[1:]  # type: ignore[return-value]

    @property
    def free_symbols(self) -> set[sp.Basic]:
        return super().free_symbols - {s for s, _ in self.indices}

    def evaluate(self) -> sp.Expr:
        indices = {symbol: tuple(values) for symbol, values in self.indices}
        return sp.Add(
            *[
                self.expression.subs(zip(indices, combi))
                for combi in itertools.product(*indices.values())
            ]
        )

    def _latex(self, printer: LatexPrinter, *args) -> str:
        indices = dict(self.indices)
        sum_symbols: list[str] = []
        for idx, values in indices.items():
            sum_symbols.append(_render_sum_symbol(printer, idx, values))
        expression = printer._print(self.expression)
        return R" ".join(sum_symbols) + f"{{{expression}}}"

    def cleanup(self) -> sp.Expr | PoolSum:
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
    printer: LatexPrinter, idx: sp.Symbol, values: Sequence[SupportsFloat]
) -> str:
    if len(values) == 0:
        return ""
    idx_latex = printer._print(idx)
    if len(values) == 1:
        value = values[0]
        return Rf"\sum_{{{idx_latex}={value}}}"
    if _is_regular_series(values):
        sorted_values = sorted(values, key=float)
        first_value = sorted_values[0]
        last_value = sorted_values[-1]
        return Rf"\sum_{{{idx_latex}={first_value}}}^{{{last_value}}}"
    idx_values = ",".join(map(printer._print, values))
    return Rf"\sum_{{{idx_latex}\in\left\{{{idx_values}\right\}}}}"


def _is_regular_series(values: Sequence[SupportsFloat]) -> bool:
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
    sorted_values = sorted(values, key=float)
    for val, next_val in zip(sorted_values, sorted_values[1:]):
        difference = float(next_val) - float(val)
        if difference != 1.0:
            return False
    return True


def perform_cached_doit(
    unevaluated_expr: sp.Expr, directory: str | None = None
) -> sp.Expr:
    """Perform :meth:`~sympy.core.basic.Basic.doit` cache the result to disk.

    The cached result is fetched from disk if the hash of the original expression is the
    same as the hash embedded in the filename.

    Args:
        unevaluated_expr: A `sympy.Expr <sympy.core.expr.Expr>` on which to call
            :meth:`~sympy.core.basic.Basic.doit`.
        directory: The directory in which to cache the result. If `None`, the cache
            directory will be put under the home directory.

    .. tip:: For a faster cache, set `PYTHONHASHSEED
        <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>`_ to a
        fixed value.
    """
    if directory is None:
        home_directory = expanduser("~")
        directory = abspath(f"{home_directory}/.sympy-cache")
    h = get_readable_hash(unevaluated_expr)
    filename = f"{directory}/{h}.pkl"
    os.makedirs(dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    _LOGGER.warning(
        f"Cached expression file {filename} not found, performing doit()..."
    )
    unfolded_expr = unevaluated_expr.doit()
    with open(filename, "wb") as f:
        pickle.dump(unfolded_expr, f)
    return unfolded_expr


def get_readable_hash(obj) -> str:
    python_hash_seed = _get_python_hash_seed()
    if python_hash_seed is not None:
        return f"pythonhashseed-{python_hash_seed}{hash(obj):+}"
    b = _to_bytes(obj)
    return hashlib.sha256(b).hexdigest()


def _to_bytes(obj) -> bytes:
    if isinstance(obj, sp.Expr):
        # Using the str printer is slower and not necessarily unique,
        # but pickle.dumps() does not always result in the same bytes stream.
        _warn_about_unsafe_hash()
        return str(obj).encode()
    return pickle.dumps(obj)


def _get_python_hash_seed() -> int | None:
    python_hash_seed = os.environ.get("PYTHONHASHSEED", "")
    if python_hash_seed is not None and python_hash_seed.isdigit():
        return int(python_hash_seed)
    return None


@functools.lru_cache(maxsize=None)  # warn once
def _warn_about_unsafe_hash():
    message = """
    PYTHONHASHSEED has not been set. For faster and safer hashing of SymPy expressions,
    set the PYTHONHASHSEED environment variable to a fixed value and rerun the program.
    See https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    """
    message = dedent(message).replace("\n", " ").strip()
    _LOGGER.warning(message)
