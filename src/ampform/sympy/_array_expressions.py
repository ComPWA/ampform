"""Temporary module for SymPy :code:`ArraySlice` and related classes.

This module can be removed once `sympy/sympy#22265
<https://github.com/sympy/sympy/pull/22265>`_ is merged and released.
"""

from __future__ import annotations

import string
import sys
from collections import abc
from itertools import zip_longest
from typing import TYPE_CHECKING, overload

import sympy as sp
from sympy.codegen.ast import none
from sympy.core.sympify import _sympify  # noqa: PLC2701
from sympy.functions.elementary.integers import floor
from sympy.printing.conventions import split_super_sub
from sympy.printing.latex import LatexPrinter
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.printer import Printer
from sympy.printing.str import StrPrinter
from sympy.tensor.array.expressions.array_expressions import (
    ArraySymbol,
    _ArrayExpr,  # noqa: PLC2701
    get_shape,
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
    from collections.abc import Iterable

    from sympy.printing.numpy import NumPyPrinter


class ArrayElement(_ArrayExpr):
    @override
    def __new__(cls, parent: sp.Expr, indices: Iterable) -> Self:
        # cspell:ignore sympified
        sympified_indices = sp.Tuple(*map(_sympify, indices))
        parent_shape = get_shape(parent)
        if any(
            (i >= s) == True  # noqa: E712
            for i, s in zip(sympified_indices, parent_shape)
        ):
            msg = "shape is out of bounds"
            raise ValueError(msg)
        if len(parent_shape):
            if len(sympified_indices) > len(parent_shape):
                msg = (
                    f"Too many indices for {cls.__name__}: parent"
                    f" {type(parent).__name__} is {len(parent_shape)}-dimensional, but"
                    f" {len(sympified_indices)} indices were given"
                )
                raise IndexError(msg)
            normalized_indices = [
                _normalize_index(i, axis_size)
                for i, axis_size in zip(indices, parent_shape)
            ]
        else:
            normalized_indices = list(indices)
        return sp.Expr.__new__(cls, parent, sp.Tuple(*normalized_indices))

    @property
    def parent(self) -> sp.Expr:
        return self._args[0]  # type: ignore[return-value]

    @property
    def indices(self) -> sp.Tuple:
        return self._args[1]  # type: ignore[return-value]


# required for lambdify
_ArrayExpr._iterable = False  # type: ignore[attr-defined]  # noqa: SLF001


@overload
def _array_symbol_getitem(
    self: type[ArraySymbol], key: sp.Basic | int
) -> ArrayElement: ...


@overload
def _array_symbol_getitem(self: type[ArraySymbol], key: slice) -> ArraySlice: ...


@overload
def _array_symbol_getitem(  # type: ignore[misc]
    self: type[ArraySymbol], key: tuple[sp.Basic | int, ...]
) -> ArrayElement: ...


@overload
def _array_symbol_getitem(
    self: type[ArraySymbol], key: tuple[sp.Basic | int | slice, ...]
) -> ArraySlice: ...


def _array_symbol_getitem(self, key):
    if isinstance(key, abc.Iterable):
        indices = tuple(key)
    else:
        indices = (key,)
    if any(isinstance(i, slice) for i in indices):
        return ArraySlice(self, indices)
    return ArrayElement(self, indices)


ArraySymbol.__getitem__ = _array_symbol_getitem  # type: ignore[assignment]


def _normalize_index(idx, axis_size: sp.Expr | None):
    if (
        axis_size
        and axis_size.is_Integer
        and (-axis_size <= idx) == True  # noqa: E712
        and (idx < 0) == True  # noqa: E712
    ):
        return idx + axis_size
    return idx


class ArraySlice(_ArrayExpr):
    parent: sp.Basic = property(lambda self: self.args[0])  # type: ignore[assignment]
    indices: tuple[sp.Tuple, ...] = property(lambda self: tuple(self.args[1]))  # type: ignore[assignment]
    is_commutative = True

    @override
    def __new__(
        cls,
        parent: sp.Basic,
        indices: tuple[sp.Basic | int | slice, ...],
    ) -> Self:
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
    def shape(self) -> tuple[sp.Basic | int, ...]:  # type: ignore[override]
        parent_shape = get_shape(self.parent)
        shape = [
            _compute_slice_size(idx, axis_size)
            for idx, axis_size in zip_longest(self.indices, parent_shape)
        ]
        return tuple(shape)


def _compute_slice_size(idx, axis_size):
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


def normalize(i, parentsize) -> tuple[sp.Basic, sp.Basic, sp.Basic]:
    if isinstance(i, slice):
        i = (i.start, i.stop, i.step)
    if not isinstance(i, (tuple, list, sp.Tuple)):
        if (i < 0) == True:  # noqa: E712
            i += parentsize
        i = (i, i + 1, 1)
    i = list(i)
    if len(i) == 2:  # noqa: PLR2004
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
            raise IndexError

    start, stop, step = tuple(none if i is None else i for i in (start, stop, step))
    return start, stop, step


def _print_latex_ArrayElement(  # noqa: N802
    self: LatexPrinter, expr: ArrayElement
) -> str:
    parent = self.parenthesize(expr.parent, PRECEDENCE["Func"], True)
    indices = ", ".join(self._print(i) for i in expr.indices)
    return f"{{{parent}}}_{{{indices}}}"


def _print_latex_ArraySlice(self: LatexPrinter, expr: ArraySlice) -> str:  # noqa: N802
    shape = getattr(expr.parent, "shape", ())
    stringified_indices = []
    for idx, axis_size in zip_longest(expr.indices, shape):
        if idx is None:
            break
        stringified_indices.append(_slice_to_str(self, idx, axis_size))
    parent = self.parenthesize(expr.parent, PRECEDENCE["Func"], strict=True)
    indices = ", ".join(stringified_indices)
    return Rf"{parent}\left[{indices}\right]"


def _print_str_ArrayElement(self: StrPrinter, expr: ArrayElement) -> str:  # noqa: N802
    parent = self.parenthesize(expr.parent, PRECEDENCE["Func"], True)
    indices = ", ".join(self._print(i) for i in expr.indices)
    return f"{parent}[{indices}]"


def _print_str_ArraySlice(self: StrPrinter, expr: ArraySlice) -> str:  # noqa: N802
    shape = getattr(expr.parent, "shape", ())
    stringified_indices = []
    for idx, axis_size in zip_longest(expr.indices, shape):
        if idx is None:
            break
        stringified_indices.append(_slice_to_str(self, idx, axis_size))
    parent = self.parenthesize(expr.parent, PRECEDENCE["Func"], strict=True)
    indices = ", ".join(stringified_indices)
    return f"{parent}[{indices}]"


def _slice_to_str(self: Printer, x, dim) -> str:
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


LatexPrinter._print_ArrayElement = _print_latex_ArrayElement  # type: ignore[assignment]  # noqa: SLF001
LatexPrinter._print_ArraySlice = _print_latex_ArraySlice  # type: ignore[attr-defined]  # noqa: SLF001
StrPrinter._print_ArrayElement = _print_str_ArrayElement  # type: ignore[assignment]  # noqa: SLF001
StrPrinter._print_ArraySlice = _print_str_ArraySlice  # type: ignore[attr-defined]  # noqa: SLF001


class ArraySum(sp.Expr):
    precedence = PRECEDENCE["Add"]

    @override
    def __new__(cls, *terms: sp.Basic, **hints) -> Self:
        terms = sp.sympify(terms)
        return sp.Expr.__new__(cls, *terms, **hints)

    @property
    def terms(self) -> tuple[sp.Basic, ...]:
        return self.args

    def _latex(self, printer: LatexPrinter, *args) -> str:
        if all(isinstance(i, (sp.Symbol, ArraySymbol)) for i in self.terms):
            names = set(map(_strip_subscript_superscript, self.terms))
            if len(names) == 1:
                name = next(iter(names))
                subscript = "".join(map(_get_subscript, self.terms))
                return f"{{{name}}}_{{{subscript}}}"
        return printer._print_ArraySum(self)  # type: ignore[attr-defined]  # noqa: SLF001


def _print_array_sum(self: Printer, expr: ArraySum) -> str:
    terms = map(self._print, expr.terms)
    return " + ".join(terms)


Printer._print_ArraySum = _print_array_sum  # type: ignore[attr-defined]  # noqa: SLF001


def _get_subscript(symbol: sp.Basic) -> str:
    """Collect subscripts from a `sympy.core.symbol.Symbol`.

    >>> _get_subscript(sp.Symbol("p1"))
    '1'
    >>> _get_subscript(sp.Symbol("p^2_{0,0}"))
    '0,0'
    """
    text = sp.latex(symbol) if isinstance(symbol, sp.Basic) else symbol
    _, _, subscripts = split_super_sub(text)
    stripped_subscripts = (s.strip("{").strip("}") for s in subscripts)
    return " ".join(stripped_subscripts)


def _strip_subscript_superscript(symbol: sp.Basic) -> str:
    """Collect subscripts from a `sympy.core.symbol.Symbol`.

    >>> _strip_subscript_superscript(sp.Symbol("p1"))
    'p'
    >>> _strip_subscript_superscript(sp.Symbol("p^2_{0,0}"))
    'p'
    """
    text = sp.latex(symbol) if isinstance(symbol, sp.Basic) else symbol
    name, _, _ = split_super_sub(text)
    return name


class ArrayAxisSum(sp.Expr):
    is_commutative = True

    @override
    def __new__(cls, array: sp.Expr, axis: int | None = None, **hints) -> Self:
        if axis is not None and not isinstance(axis, (int, sp.Integer)):
            msg = "Only single digits allowed for axis"
            raise TypeError(msg)
        args = sp.sympify((array, axis))
        return sp.Expr.__new__(cls, *args, **hints)

    @property
    def array(self) -> sp.Expr:
        return self.args[0]  # type: ignore[return-value]

    @property
    def axis(self) -> sp.Integer | None:
        return self.args[1]  # type: ignore[return-value]

    def _latex(self, printer: LatexPrinter, *args) -> str:
        array = printer._print(self.array)
        if self.axis is None:
            return Rf"\sum{{{array}}}"
        axis = printer._print(self.axis)
        return Rf"\sum_{{\mathrm{{axis{axis}}}}}{{{array}}}"

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
        printer.module_imports[printer._module].add("sum")
        array = printer._print(self.array)
        axis = printer._print(self.axis)
        return f"sum({array}, axis={axis})"


class ArrayMultiplication(sp.Expr):
    r"""Contract rank-:math:`n` arrays and a rank-:math`n-1` array.

    This class is particularly useful to create a tensor product of rank-3 matrix array
    classes, such as `.BoostZ`, `.RotationY`, and `.RotationZ`, with a rank-2
    `.FourMomentumSymbol`. In that case, if :math:`n` is the number of events, you would
    get a contraction of arrays of shape :math:`n\times\times4\times4` (:math:`n`
    Lorentz matrices) to :math:`n\times\times4` (:math:`n` four-momentum tuples).
    """

    @override
    def __new__(cls, *tensors: sp.Basic, **hints) -> Self:
        tensors = sp.sympify(tensors)
        return sp.Expr.__new__(cls, *tensors, **hints)

    @property
    def tensors(self) -> list[sp.Expr]:
        return self.args  # type: ignore[return-value]

    def _latex(self, printer: LatexPrinter, *args) -> str:
        tensors = map(printer._print, self.tensors)
        return " ".join(tensors)

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
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

    This class is particularly useful to create a tensor product of rank-3 matrix array
    classes, such as `.BoostZ`, `.RotationY`, and `.RotationZ`, with a rank-3
    `.FourMomentumSymbol`. In that case, if :math:`n` is the number of events, you would
    get a contraction of arrays of shape :math:`n\times\times4\times4` (:math:`n`
    Lorentz matrices) to :math:`n\times\times4\times4` (:math:`n` four-momentum tuples).
    """

    @override
    def __new__(cls, *tensors: sp.Basic, **hints) -> Self:
        tensors = sp.sympify(tensors)
        return sp.Expr.__new__(cls, *tensors, **hints)

    @property
    def tensors(self) -> tuple[sp.Basic, ...]:
        return self.args

    def _latex(self, printer: LatexPrinter, *args) -> str:
        tensors = map(printer._print, self.tensors)
        return " ".join(tensors)

    def _numpycode(self, printer: NumPyPrinter, *args) -> str:
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
