"""Tools that facilitate in building :mod:`sympy` expressions.

.. autodecorator:: unevaluated
.. autofunction:: argument

.. dropdown:: SymPy assumptions

    .. autodata:: ExprClass
    .. autoclass:: SymPyAssumptions

"""

from __future__ import annotations

import itertools
import re
import sys
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast

import sympy as sp
from sympy.printing.conventions import split_super_sub
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.pycode import _unpack_integral_limits  # noqa: PLC2701

from ._decorator import ExprClass as ExprClass
from ._decorator import SymPyAssumptions as SymPyAssumptions
from ._decorator import argument as argument
from ._decorator import get_non_sympy_fields
from ._decorator import unevaluated as unevaluated
from .cached import doit as perform_cached_doit  # noqa: F401
from .cached import xreplace as perform_cached_substitution  # noqa: F401
from .deprecated import UnevaluatedExpression as UnevaluatedExpression
from .deprecated import create_expression as create_expression
from .deprecated import implement_doit_method as implement_doit_method
from .deprecated import implement_expr as implement_expr
from .deprecated import make_commutative as make_commutative

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override
if TYPE_CHECKING:
    from collections.abc import Callable

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
    from collections.abc import Iterable, Sequence
    from typing import SupportsFloat, TypeVar

    from sympy.printing.latex import LatexPrinter
    from sympy.printing.numpy import NumPyPrinter

    T = TypeVar("T", bound=sp.Basic)


def partial_doit(
    expr: T,
    types: type[sp.Basic] | tuple[type[sp.Basic], ...],
    recursive: bool = False,
) -> T:
    if recursive:
        while substitutions := _get_substitutions(expr, types):
            expr = expr.xreplace(substitutions)
        return expr
    substitutions = _get_substitutions(expr, types)
    return expr.xreplace(substitutions)


def _get_substitutions(
    expr: sp.Basic, types: type[sp.Basic] | tuple[type[sp.Basic], ...]
) -> dict[sp.Basic, sp.Basic]:
    return {
        node: node.doit(deep=False)
        for node in sp.preorder_traversal(expr)
        if isinstance(node, types)
    }


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
    `~sympy.core.symbol.Symbol`, use :func:`.substitute_indexed_symbols`.

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
        expr: PoolSum = sp.Expr.__new__(cls, *args, **hints)  # ty:ignore[not-iterable]
        if evaluate:
            return expr.evaluate()
        return expr

    @property
    def expression(self) -> sp.Expr:
        return self.args[0]  # ty:ignore[invalid-return-type]

    @property
    def indices(self) -> list[tuple[sp.Symbol, tuple[sp.Float, ...]]]:
        return self.args[1:]

    @property
    def free_symbols(self) -> set[sp.Basic]:
        return super().free_symbols - {s for s, _ in self.indices}

    @override
    def doit(self, deep: bool = True) -> sp.Expr:  # ty:ignore[invalid-method-override]
        expr = self.evaluate()
        if deep:
            return expr.doit()
        return expr

    def evaluate(self) -> sp.Expr:
        indices = {symbol: tuple(values) for symbol, values in self.indices}
        return sp.Add(*[
            self.expression.subs(zip(indices, combi, strict=True))
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
    for val, next_val in itertools.pairwise(sorted_values):
        difference = float(next_val) - float(val)
        if difference != 1.0:  # noqa: RUF069
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


def rename_symbols(
    expression: sp.Expr, renames: Callable[[str], str] | dict[str, str]
) -> sp.Expr:
    r"""Rename symbols in an expression.

    >>> a, b, x = sp.symbols(R"a \beta x")
    >>> expr = a + b * x
    >>> rename_symbols(expr, renames={"a": "A", R"\beta": "B"})
    A + B*x
    >>> rename_symbols(expr, renames=lambda s: s.replace("\\", ""))
    a + beta*x
    >>> rename_symbols(expr, renames={"non-existent": "c"})
    Traceback (most recent call last):
        ...
    KeyError: "No symbol with name 'non-existent' in expression"
    """
    substitutions: dict[sp.Symbol, sp.Symbol] = {}
    free_symbols = cast("set[sp.Symbol]", expression.free_symbols)
    if callable(renames):
        for old_symbol in free_symbols:
            new_name = renames(old_symbol.name)
            new_symbol = sp.Symbol(new_name, **old_symbol.assumptions0)
            substitutions[old_symbol] = new_symbol
    elif isinstance(renames, dict):
        for old_name, new_name in renames.items():
            matches = (s for s in free_symbols if s.name == old_name)
            try:
                old_symbol = next(matches)
            except StopIteration as e:
                msg = f"No symbol with name '{old_name}' in expression"
                raise KeyError(msg) from e
            new_symbol = sp.Symbol(new_name, **old_symbol.assumptions0)
            substitutions[old_symbol] = new_symbol
    else:
        msg = f"Cannot rename from type {type(renames).__name__}"
        raise TypeError(msg)
    return expression.xreplace(substitutions)


@unevaluated(implement_doit=False)
class NumericalIntegral(sp.Integral):
    """Expression class representing an integral that should be evaluated numerically.

    This class inherits from `sympy.Integral <sympy.integrals.integrals.Integral>`, but
    is blocked from evaluating symbolically. Instead, it should be lambdified to a
    numerical integration function and evaluated numerically.

    .. seealso:: :ref:`usage/sympy:Numerical integrals`

    .. version-added:: 0.14.10

    .. version-changed:: 0.16.0

        * Renamed from :code:`UnevaluatableIntegral` to `NumericalIntegral`.
        * The integration algorithm is configured through class constructor arguments
          rather than class variables.
    """

    function: sp.Expr
    """Integrand of the integral."""
    limits: tuple[sp.Symbol, sp.Basic, sp.Basic]
    """Integration variable and its limits (can be `~sympy.core.numbers.Infinity`)."""
    algorithm: str | None = argument(default=None, kw_only=True, sympify=False)
    """Name of the numerical integration algorithm to use when lambdifying this integral.

    The algorithm should be in the format :code:`module.function`, for instance
    :func:`scipy.integrate.quad_vec` or :func:`quadax.quadgk`. By default, the algorithm
    is :func:`quadax.romberg` when lambdifying to JAX and
    :func:`scipy.integrate.quad_vec` when lambdifying to NumPy.
    """
    configuration: dict[str, Any] | None = argument(
        default=None, sympify=False, kw_only=True
    )
    """Keyword arguments for the numerical integration algorithm.

    For example, for :func:`scipy.integrate.quad_vec`, one can set the relative
    tolerance with :code:`configuration={'epsrel': 1e-5}`.
    """
    dummify: bool = argument(default=True, sympify=False, kw_only=True)
    """Replace the integration variable with a dummy symbol before lambdification.

    The integrand expression is lambdified to a :code:`lambda` function. Therefore, when
    the integrand expresssion contains the integration variable in a non-trivial way,
    and the expression is lambdified using common sub-expressions, it is better to
    replace it with a unique `~sympy.core.symbol.Dummy` symbol that does not appear
    anywhere else in the expression tree, so that is not pulled out of the
    :code:`lambda` function.
    """

    @override
    def doit(self, **hints):
        args = [arg.doit(**hints) for arg in self.args]
        kwargs = {
            field.name: getattr(self, field.name)
            for field in get_non_sympy_fields(self)
        }
        return self.func(*args, **kwargs)

    @override
    def _jaxcode(self, printer, *args) -> str:  # ty:ignore[invalid-explicit-override]
        algorithm = self.algorithm or "quadax.romberg"
        if algorithm.startswith("quadax"):
            return self.__to_quadax_like(printer, algorithm)
        return self.__to_scipy_like(printer, algorithm)

    @override
    def _numpycode(self, printer, *args) -> str:  # ty:ignore[invalid-explicit-override]
        algorithm = self.algorithm or "scipy.integrate.quad_vec"
        if algorithm.startswith("quadax"):
            return self.__to_quadax_like(printer, algorithm)
        return self.__to_scipy_like(printer, algorithm)

    def __to_quadax_like(self, printer, algorithm: str) -> str:
        """https://quadax.readthedocs.io."""
        integrate, integrand, x, a, b = self.__prepare_components(printer, algorithm)
        src = _generate_function_call(
            integrate,
            fun=f"lambda {x}: {integrand}",
            interval=f"({a}, {b})",
            **self.configuration or {},
        )
        return f"{src}[0]"

    def __to_scipy_like(self, printer, algorithm: str) -> str:
        """https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad_vec.html."""
        integrate, integrand, x, a, b = self.__prepare_components(printer, algorithm)
        kwargs = self.configuration or {}
        src = _generate_function_call(
            integrate, f"lambda {x}: {integrand}", a, b, **kwargs
        )
        return f"{src}[0]"

    def __prepare_components(
        self, printer, algorithm: str
    ) -> tuple[str, str, str, str, str]:
        integration_vars, limits = _unpack_integral_limits(self)
        if len(limits) != 1 or len(integration_vars) != 1:
            msg = f"Cannot handle {len(limits)}-dimensional integrals"
            raise ValueError(msg)
        x = integration_vars[0]
        a, b = limits[0]
        integrand = self.function
        if self.dummify:
            dummy = sp.Dummy()
            integrand = integrand.xreplace({x: dummy})
            x = dummy
        parts = algorithm.split(".")
        if len(parts) < 2:  # noqa: PLR2004
            msg = f"Algorithm should be in format 'module.function', got '{algorithm}'"
            raise ValueError(msg)
        module_name = ".".join(parts[:-1])
        algorithm_name = parts[-1]
        printer.module_imports[module_name].add(algorithm_name)
        return (
            algorithm_name,
            printer._print(integrand),
            printer._print(x),
            printer._print(a),
            printer._print(b),
        )


def _generate_function_call(func_name: str, /, *args, **kwargs) -> str:
    """Generate a function call string with the given function name, arguments, and keyword arguments.

    >>> _generate_function_call("quad_vec", "f", 0, 1, epsabs=1e-5)
    'quad_vec(f, 0, 1, epsabs=1e-05)'
    >>> _generate_function_call("quadgk", fun="lambda x: x**2", interval=(0, 1))
    'quadgk(fun=lambda x: x**2, interval=(0, 1))'
    """
    src = f"{func_name}("
    src += ", ".join(map(str, args))
    if args:
        src += ", "
    src += ", ".join(f"{key}={value}" for key, value in kwargs.items())
    src += ")"
    return src
