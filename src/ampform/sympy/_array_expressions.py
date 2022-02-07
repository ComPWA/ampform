# cspell:ignore sympified
# pylint: disable=arguments-differ, line-too-long, protected-access
# pylint: disable=singleton-comparison, unused-argument
"""Temporary module for SymPy :code:`ArraySlice` and related classes.

This module can be removed once `sympy/sympy#22265
<https://github.com/sympy/sympy/pull/22265>`_ is merged and released.
"""

import string
from collections import abc
from itertools import zip_longest
from typing import Any, Iterable, List, Optional, Tuple, Type, Union, overload

import sympy as sp
from sympy.codegen.ast import none
from sympy.core.sympify import _sympify
from sympy.functions.elementary.integers import floor
from sympy.printing.conventions import split_super_sub
from sympy.printing.latex import LatexPrinter
from sympy.printing.numpy import NumPyPrinter
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.printer import Printer
from sympy.printing.str import StrPrinter
from sympy.tensor.array.expressions.array_expressions import (
    ArraySymbol,
    _ArrayExpr,
    get_shape,
)

from . import create_expression, make_commutative


class ArrayElement(_ArrayExpr):
    parent: sp.Expr = property(lambda self: self._args[0])  # type: ignore[assignment]
    indices: sp.Tuple = property(lambda self: self._args[1])

    def __new__(cls, parent: sp.Expr, indices: Iterable) -> "ArrayElement":
        sympified_indices = sp.Tuple(*map(_sympify, indices))
        parent_shape = get_shape(parent)
        if any(
            (i >= s) == True  # noqa: E712
            for i, s in zip(sympified_indices, parent_shape)
        ):
            raise ValueError("shape is out of bounds")
        if len(parent_shape):
            if len(sympified_indices) > len(parent_shape):
                raise IndexError(
                    f"Too many indices for {cls.__name__}: parent"
                    f" {type(parent).__name__} is"
                    f" {len(parent_shape)}-dimensional, but"
                    f" {len(sympified_indices)} indices were given"
                )
            normalized_indices = [
                _normalize_index(i, axis_size)
                for i, axis_size in zip(indices, parent_shape)
            ]
        else:
            normalized_indices = list(indices)
        return sp.Expr.__new__(cls, parent, sp.Tuple(*normalized_indices))


_ArrayExpr._iterable = False  # required for lambdify


@overload
def _array_symbol_getitem(
    self: Type[ArraySymbol], key: Union[sp.Basic, int]
) -> "ArrayElement":
    ...


@overload
def _array_symbol_getitem(self: Type[ArraySymbol], key: slice) -> "ArraySlice":
    ...


@overload
def _array_symbol_getitem(
    self: Type[ArraySymbol], key: Tuple[Union[sp.Basic, int], ...]
) -> "ArrayElement":
    ...


@overload
def _array_symbol_getitem(
    self: Type[ArraySymbol], key: Tuple[Union[sp.Basic, int, slice], ...]
) -> "ArraySlice":
    ...


def _array_symbol_getitem(self, key):  # type: ignore[no-untyped-def]
    if isinstance(key, abc.Iterable):
        indices = tuple(key)
    else:
        indices = (key,)
    if any(isinstance(i, slice) for i in indices):
        return ArraySlice(self, indices)
    return ArrayElement(self, indices)


ArraySymbol.__getitem__ = _array_symbol_getitem


def _normalize_index(idx: Any, axis_size: Optional[sp.Expr]) -> Any:
    if (
        axis_size
        and axis_size.is_Integer
        and (-axis_size <= idx) == True  # noqa: E712
        and (idx < 0) == True  # noqa: E712
    ):
        return idx + axis_size
    return idx


class ArraySlice(_ArrayExpr):
    parent: sp.Expr = property(lambda self: self.args[0])  # type: ignore[assignment]
    indices: Tuple[sp.Tuple, ...] = property(lambda self: tuple(self.args[1]))  # type: ignore[assignment]

    def __new__(
        cls,
        parent: sp.Expr,
        indices: Tuple[Union[sp.Basic, int, slice], ...],
    ) -> "ArraySlice":
        parent_shape = get_shape(parent)
        normalized_indices = []
        for idx, axis_size in zip_longest(indices, parent_shape):
            if idx is None:
                break
            if isinstance(idx, slice):
                new_idx = sp.Tuple(*normalize(idx, axis_size))
            else:
                new_idx = _sympify(_normalize_index(idx, axis_size))
            normalized_indices.append(new_idx)
        return sp.Expr.__new__(cls, parent, sp.Tuple(*normalized_indices))

    @property
    def shape(self) -> Tuple[Union[sp.Basic, int], ...]:
        parent_shape = get_shape(self.parent)
        shape = [
            _compute_slice_size(idx, axis_size)
            for idx, axis_size in zip_longest(self.indices, parent_shape)
        ]
        return tuple(shape)


def _compute_slice_size(idx: Any, axis_size: Any) -> Any:  # noqa: R701
    if idx is None:
        return axis_size
    if not isinstance(idx, sp.Tuple):
        return 1
    start, stop, step = idx
    if stop is None and axis_size is None:
        return None
    size = stop - start
    size = size if step == 1 or step is None else floor(size / step)
    if axis_size is not None and (size > axis_size) == True:  # noqa: E712
        return axis_size
    return size


def normalize(  # noqa: R701
    i: Any, parentsize: Any
) -> Tuple[sp.Basic, sp.Basic, sp.Basic]:
    if isinstance(i, slice):
        i = (i.start, i.stop, i.step)
    if not isinstance(i, (tuple, list, sp.Tuple)):
        if (i < 0) == True:  # noqa: E712
            i += parentsize
        i = (i, i + 1, 1)
    i = list(i)
    if len(i) == 2:
        i.append(1)
    start, stop, step = i
    start = start or 0
    if parentsize is not None:
        if stop is None:
            stop = parentsize
        if (start < 0) == True:  # noqa: E712
            start += parentsize
        if (stop < 0) == True:  # noqa: E712
            stop += parentsize
        step = step or 1

        if ((stop - start) * step < 1) == True:  # noqa: E712
            raise IndexError()

    start, stop, step = tuple(
        map(
            lambda i: none if i is None else i,
            (start, stop, step),
        )
    )
    return start, stop, step


# pylint: disable=invalid-name
def _print_latex_ArrayElement(  # noqa: N802
    self: LatexPrinter, expr: ArrayElement
) -> str:
    parent = self.parenthesize(expr.parent, PRECEDENCE["Func"], True)
    indices = ", ".join(self._print(i) for i in expr.indices)
    return f"{{{parent}}}_{{{indices}}}"


def _print_latex_ArraySlice(  # noqa: N802
    self: LatexPrinter, expr: ArraySlice
) -> str:
    shape = getattr(expr.parent, "shape", ())
    stringified_indices = []
    for idx, axis_size in zip_longest(expr.indices, shape):
        if idx is None:
            break
        stringified_indices.append(_slice_to_str(self, idx, axis_size))
    parent = self.parenthesize(expr.parent, PRECEDENCE["Func"], strict=True)
    indices = ", ".join(stringified_indices)
    return Rf"{parent}\left[{indices}\right]"


def _print_str_ArrayElement(  # noqa: N802
    self: StrPrinter, expr: ArrayElement
) -> str:
    parent = self.parenthesize(expr.parent, PRECEDENCE["Func"], True)
    indices = ", ".join(self._print(i) for i in expr.indices)
    return f"{parent}[{indices}]"


def _print_str_ArraySlice(  # noqa: N802
    self: StrPrinter, expr: ArraySlice
) -> str:
    shape = getattr(expr.parent, "shape", ())
    stringified_indices = []
    for idx, axis_size in zip_longest(expr.indices, shape):
        if idx is None:
            break
        stringified_indices.append(_slice_to_str(self, idx, axis_size))
    parent = self.parenthesize(expr.parent, PRECEDENCE["Func"], strict=True)
    indices = ", ".join(stringified_indices)
    return f"{parent}[{indices}]"


def _slice_to_str(self: LatexPrinter, x: Any, dim: Any) -> str:
    if not isinstance(x, abc.Iterable):
        return self._print(x)
    x = list(x)
    if x[2] == 1 or x[2] in {none, None}:
        del x[2]
    if x[0] == 0:
        x[0] = None
    if x[1] == dim:
        x[1] = None
    return ":".join("" if xi in {none, None} else self._print(xi) for xi in x)


LatexPrinter._print_ArrayElement = _print_latex_ArrayElement
LatexPrinter._print_ArraySlice = _print_latex_ArraySlice
StrPrinter._print_ArrayElement = _print_str_ArrayElement
StrPrinter._print_ArraySlice = _print_str_ArraySlice


class ArraySum(sp.Expr):
    precedence = PRECEDENCE["Add"]

    def __new__(cls, *terms: sp.Basic, **hints: Any) -> "ArraySum":
        return create_expression(cls, *terms, **hints)

    @property
    def terms(self) -> Tuple[sp.Basic, ...]:
        return self.args

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        if all(
            map(lambda i: isinstance(i, (sp.Symbol, ArraySymbol)), self.terms)
        ):
            names = set(map(_strip_subscript_superscript, self.terms))
            if len(names) == 1:
                name = next(iter(names))
                subscript = "".join(map(_get_subscript, self.terms))
                return f"{{{name}}}_{{{subscript}}}"
        return printer._print_ArraySum(self)


def _print_array_sum(self: Printer, expr: ArraySum) -> str:
    terms = map(self._print, expr.terms)
    return " + ".join(terms)


Printer._print_ArraySum = _print_array_sum


def _get_subscript(symbol: sp.Symbol) -> str:
    """Collect subscripts from a `sympy.core.symbol.Symbol`.

    >>> _get_subscript(sp.Symbol("p1"))
    '1'
    >>> _get_subscript(sp.Symbol("p^2_{0,0}"))
    '0,0'
    """
    if isinstance(symbol, sp.Basic):
        text = sp.latex(symbol)
    else:
        text = symbol
    _, _, subscripts = split_super_sub(text)
    stripped_subscripts: Iterable[str] = map(
        lambda s: s.strip("{").strip("}"), subscripts
    )
    return " ".join(stripped_subscripts)


def _strip_subscript_superscript(symbol: sp.Symbol) -> str:
    """Collect subscripts from a `sympy.core.symbol.Symbol`.

    >>> _strip_subscript_superscript(sp.Symbol("p1"))
    'p'
    >>> _strip_subscript_superscript(sp.Symbol("p^2_{0,0}"))
    'p'
    """
    if isinstance(symbol, sp.Basic):
        text = sp.latex(symbol)
    else:
        text = symbol
    name, _, _ = split_super_sub(text)
    return name


@make_commutative
class ArrayAxisSum(sp.Expr):
    def __new__(
        cls, array: ArraySymbol, axis: Optional[int] = None, **hints: Any
    ) -> "ArrayAxisSum":
        if axis is not None and not isinstance(axis, (int, sp.Integer)):
            raise TypeError("Only single digits allowed for axis")
        return create_expression(cls, array, axis, **hints)

    @property
    def array(self) -> ArraySymbol:
        return self.args[0]

    @property
    def axis(self) -> Optional[int]:
        return self.args[1]

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        array = printer._print(self.array)
        if self.axis is None:
            return Rf"\sum{{{array}}}"
        axis = printer._print(self.axis)
        return Rf"\sum_{{\mathrm{{axis{axis}}}}}{{{array}}}"

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        printer.module_imports[printer._module].add("sum")
        array = printer._print(self.array)
        axis = printer._print(self.axis)
        return f"sum({array}, axis={axis})"


class ArrayMultiplication(sp.Expr):
    r"""Contract rank-:math:`n` arrays and a rank-:math`n-1` array.

    This class is particularly useful to create a tensor product of rank-3
    matrix array classes, such as `.BoostZ`, `.RotationY`, and `.RotationZ`,
    with a rank-2 `.FourMomentumSymbol`. In that case, if :math:`n` is the
    number of events, you would get a contraction of arrays of shape
    :math:`n\times\times4\times4` (:math:`n` Lorentz matrices) to
    :math:`n\times\times4` (:math:`n` four-momentum tuples).
    """

    def __new__(cls, *tensors: sp.Expr, **hints: Any) -> "ArrayMultiplication":
        return create_expression(cls, *tensors, **hints)

    @property
    def tensors(self) -> List[sp.Expr]:
        return self.args

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        tensors = map(printer._print, self.tensors)
        return " ".join(tensors)

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        printer.module_imports[printer._module].add("einsum")
        tensors = list(map(printer._print, self.args))
        if len(tensors) == 0:
            return ""
        if len(tensors) == 1:
            return tensors[0]
        contraction = self._create_einsum_subscripts(len(tensors))
        return f'einsum("{contraction}", {", ".join(tensors)})'

    @staticmethod
    def _create_einsum_subscripts(n_arrays: int) -> str:
        """Create the contraction path for `ArrayMultiplication`.

        >>> ArrayMultiplication._create_einsum_subscripts(1)
        '...i->...i'
        >>> ArrayMultiplication._create_einsum_subscripts(2)
        '...ij,...j->...i'
        >>> ArrayMultiplication._create_einsum_subscripts(3)
        '...ij,...jk,...k->...i'
        """
        letters = string.ascii_lowercase[8 : 8 + n_arrays]
        contraction = ""
        for i, j in zip_longest(letters, letters[1:]):
            if j is None:
                contraction += f"...{i}"
            else:
                contraction += f"...{i}{j},"
        contraction += "->...i"
        return contraction


class MatrixMultiplication(sp.Expr):
    r"""Contract rank-:math:`n` arrays and a rank-:math`n` array.

    This class is particularly useful to create a tensor product of rank-3
    matrix array classes, such as `.BoostZ`, `.RotationY`, and `.RotationZ`,
    with a rank-3 `.FourMomentumSymbol`. In that case, if :math:`n` is the
    number of events, you would get a contraction of arrays of shape
    :math:`n\times\times4\times4` (:math:`n` Lorentz matrices) to
    :math:`n\times\times4\times4` (:math:`n` four-momentum tuples).
    """

    def __new__(cls, *tensors: sp.Expr, **hints: Any) -> "ArrayMultiplication":
        return create_expression(cls, *tensors, **hints)

    @property
    def tensors(self) -> List[sp.Expr]:
        return self.args

    def _latex(self, printer: LatexPrinter, *args: Any) -> str:
        tensors = map(printer._print, self.tensors)
        return " ".join(tensors)

    def _numpycode(self, printer: NumPyPrinter, *args: Any) -> str:
        printer.module_imports[printer._module].add("einsum")
        tensors = list(map(printer._print, self.args))
        if len(tensors) == 0:
            return ""
        if len(tensors) == 1:
            return tensors[0]
        contraction = self._create_einsum_subscripts(len(tensors))
        return f'einsum("{contraction}", {", ".join(tensors)})'

    @staticmethod
    def _create_einsum_subscripts(n_arrays: int) -> str:
        """Create the contraction path for `MatrixMultiplication`.

        >>> MatrixMultiplication._create_einsum_subscripts(1)
        '...ij->...ij'
        >>> MatrixMultiplication._create_einsum_subscripts(2)
        '...ij,...jk->...ik'
        >>> MatrixMultiplication._create_einsum_subscripts(3)
        '...ij,...jk,...kl->...il'
        """
        letters = string.ascii_lowercase[8 : 8 + n_arrays + 1]
        groups = []
        for i, j in zip(letters, letters[1:]):
            groups.append(f"...{i}{j}")
        return f"{','.join(groups)}->...i{letters[-1]}"
