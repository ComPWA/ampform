"""Tools that facilitate in building :mod:`sympy` expressions.

.. autodecorator:: unevaluated
.. autofunction:: argument

.. dropdown:: SymPy assumptions

    .. autodata:: ExprClass
    .. autoclass:: SymPyAssumptions

"""

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
from typing import TYPE_CHECKING, Iterable, Sequence, SupportsFloat

import sympy as sp
from sympy.printing.precedence import PRECEDENCE

from ._decorator import (
    ExprClass,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    SymPyAssumptions,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    argument,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    unevaluated,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)
from .deprecated import (
    UnevaluatedExpression,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    create_expression,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    implement_doit_method,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    implement_expr,  # pyright: ignore[reportUnusedImport]  # noqa: F401
    make_commutative,  # pyright: ignore[reportUnusedImport]  # noqa: F401
)

if TYPE_CHECKING:
    from sympy.printing.latex import LatexPrinter
    from sympy.printing.numpy import NumPyPrinter

_LOGGER = logging.getLogger(__name__)


class NumPyPrintable(sp.Expr):
    r"""`~sympy.core.expr.Expr` class that can lambdify to NumPy code.

    This interface is for classes that derive from `sympy.Expr <sympy.core.expr.Expr>`
    and that require a :meth:`_numpycode` method in case the class does not correctly
    :func:`~sympy.utilities.lambdify.lambdify` to NumPy code. For more info on SymPy
    printers, see :doc:`sympy:modules/printing`.

    Several computational frameworks try to converge their interface to that of NumPy.
    See for instance `TensorFlow's NumPy API
    <https://www.tensorflow.org/guide/tf_numpy>`_ and `jax.numpy
    <https://jax.readthedocs.io/en/latest/jax.numpy.html>`_. This fact is used in
    `TensorWaves <https://tensorwaves.rtfd.io>`_ to
    :func:`~sympy.utilities.lambdify.lambdify` SymPy expressions to these different
    backends with the same lambdification code.

    .. warning:: If you decorate this class with :func:`unevaluated`, you usually want
        to do so with :code:`implement_doit=False`, because you do not want the class
        to be 'unfolded' with :meth:`~sympy.core.basic.Basic.doit` before lambdification.


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


class PoolSum(sp.Expr):
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
        evaluate: bool = False,
        **hints,
    ) -> PoolSum:
        converted_indices = []
        for idx_symbol, values in indices:
            values = tuple(values)
            if len(values) == 0:
                msg = f"No values provided for index {idx_symbol}"
                raise ValueError(msg)
            converted_indices.append((idx_symbol, values))
        args = sp.sympify((expression, *converted_indices))
        expr: PoolSum = sp.Expr.__new__(cls, *args, **hints)
        if evaluate:
            return expr.evaluate()  # type: ignore[return-value]
        return expr

    @property
    def expression(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    @property
    def indices(self) -> list[tuple[sp.Symbol, tuple[sp.Float, ...]]]:
        return self.args[1:]  # type: ignore[return-value]

    @property
    def free_symbols(self) -> set[sp.Basic]:
        return super().free_symbols - {s for s, _ in self.indices}

    def doit(self, deep: bool = True) -> sp.Expr:  # type: ignore[override]
        expr = self.evaluate()
        if deep:
            return expr.doit()
        return expr

    def evaluate(self) -> sp.Expr:
        indices = {symbol: tuple(values) for symbol, values in self.indices}
        return sp.Add(*[
            self.expression.subs(zip(indices, combi))
            for combi in itertools.product(*indices.values())
        ])

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
        if difference != 1.0:  # noqa: PLR2004
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
            return pickle.load(f)  # noqa: S301
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
