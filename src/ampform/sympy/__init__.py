"""Tools that facilitate in building :mod:`sympy` expressions.

.. autodecorator:: unevaluated
.. autofunction:: argument

.. dropdown:: SymPy assumptions

    .. autodata:: ExprClass
    .. autoclass:: SymPyAssumptions

"""

from __future__ import annotations

import itertools
import logging
import re
import sys
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, SupportsFloat

import sympy as sp
from sympy.printing.conventions import split_super_sub
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.pycode import _unpack_integral_limits  # noqa: PLC2701

from ampform.sympy._cache import cache_to_disk

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

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

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

    @override
    def __new__(
        cls,
        expression,
        *indices: tuple[sp.Symbol, Iterable[sp.Basic]],
        evaluate: bool = False,
        **hints,
    ) -> Self:
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

    @override
    def doit(self, deep: bool = True) -> sp.Expr:  # type: ignore[misc]
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
        if difference != 1.0:
            return False
    return True


def determine_indices(symbol: sp.Basic) -> list[int]:
    r"""Extract any indices if available from a `~sympy.core.symbol.Symbol`.

    >>> determine_indices(sp.Symbol("m1"))
    [1]
    >>> determine_indices(sp.Symbol("m_12"))
    [12]
    >>> determine_indices(sp.Symbol("m_a2"))
    [2]
    >>> determine_indices(sp.Symbol(R"\alpha_{i2, 5}"))
    [2, 5]
    >>> determine_indices(sp.Symbol("m"))
    []

    `~sympy.tensor.indexed.Indexed` instances can also be handled:
    >>> m_a = sp.IndexedBase("m_a")
    >>> determine_indices(m_a[0])
    [0]
    """
    _, _, subscripts = split_super_sub(sp.latex(symbol))
    if not subscripts:
        return []
    subscript: str = subscripts[-1]
    subscript = re.sub(r"[^0-9^\,]", "", subscript)
    subscript = f"[{subscript}]"
    try:
        indices = eval(subscript)  # noqa: S307
    except SyntaxError:
        return []
    return list(indices)


class UnevaluatableIntegral(sp.Integral):
    """See :ref:`usage/sympy:Numerical integrals`.

    .. versionadded:: 0.14.10
    """

    abs_tolerance = 1e-5
    rel_tolerance = 1e-5
    limit = 50
    dummify = True

    @override
    def doit(self, **hints):
        args = [arg.doit(**hints) for arg in self.args]
        return self.func(*args)

    @override  # type:ignore[misc]
    def _numpycode(self, printer, *args) -> str:
        _warn_if_scipy_not_installed()
        integration_vars, limits = _unpack_integral_limits(self)
        if len(limits) != 1 or len(integration_vars) != 1:
            msg = f"Cannot handle {len(limits)}-dimensional integrals"
            raise ValueError(msg)
        x = integration_vars[0]
        a, b = limits[0]
        expr = self.args[0]
        if self.dummify:
            dummy = sp.Dummy()
            expr = expr.xreplace({x: dummy})
            x = dummy
        integrate_func = "quad_vec"
        printer.module_imports["scipy.integrate"].add(integrate_func)
        return (
            f"{integrate_func}(lambda {printer._print(x)}: {printer._print(expr)},"
            f" {printer._print(a)}, {printer._print(b)},"
            f" epsabs={self.abs_tolerance}, epsrel={self.abs_tolerance},"
            f" limit={self.limit})[0]"
        )


def _warn_if_scipy_not_installed() -> None:
    try:
        import scipy  # noqa: F401, PLC0415  # pyright: ignore[reportUnusedImport, reportMissingImports]
    except ImportError:
        warnings.warn(
            "Scipy is not installed. Install with 'pip install scipy' or with 'pip"
            " install ampform[scipy]'",
            stacklevel=1,
        )


@cache_to_disk
def perform_cached_doit(unevaluated_expr: sp.Expr) -> sp.Expr:
    """Perform :meth:`~sympy.core.basic.Basic.doit` and cache the result to disk.

    The cached result is fetched from disk if the hash of the original expression is the
    same as the hash embedded in the filename (see :func:`.get_readable_hash`).

    Args:
        unevaluated_expr: A `sympy.Expr <sympy.core.expr.Expr>` on which to call
            :meth:`~sympy.core.basic.Basic.doit`.

    .. versionadded:: 0.14.4
    .. automodule:: ampform.sympy._cache
    """
    return unevaluated_expr.doit()


@cache_to_disk
def perform_cached_substitution(
    expr: sp.Expr,
    substitutions: Mapping[sp.Basic, sp.Basic],
) -> sp.Expr:
    return expr.xreplace(substitutions)
