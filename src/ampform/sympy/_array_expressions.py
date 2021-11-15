# cspell:ignore sympified
# pylint: disable=arguments-differ, line-too-long, protected-access, singleton-comparison
"""Temporary module for SymPy :code:`ArraySlice` and related classes.

This module can be removed once `sympy/sympy#22265
<https://github.com/sympy/sympy/pull/22265>`_ is merged and released.
"""

from collections import abc
from itertools import zip_longest
from typing import Any, Iterable, Optional
from typing import Tuple as tTuple
from typing import Type
from typing import Union as tUnion
from typing import overload

from sympy.codegen.ast import none
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.sympify import _sympify
from sympy.functions.elementary.integers import floor
from sympy.printing.latex import LatexPrinter
from sympy.printing.precedence import PRECEDENCE
from sympy.printing.str import StrPrinter
from sympy.tensor.array.expressions.array_expressions import (
    ArraySymbol,
    _ArrayExpr,
    get_shape,
)


class ArrayElement(_ArrayExpr):
    parent: Expr = property(lambda self: self._args[0])  # type: ignore[assignment]
    indices: Tuple = property(lambda self: self._args[1])

    def __new__(cls, parent: Expr, indices: Iterable) -> "ArrayElement":
        sympified_indices = Tuple(*map(_sympify, indices))
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
        return Expr.__new__(cls, parent, Tuple(*normalized_indices))


_ArrayExpr._iterable = False  # required for lambdify


@overload
def _array_symbol_getitem(
    self: Type[ArraySymbol], key: tUnion[Basic, int]
) -> "ArrayElement":
    ...


@overload
def _array_symbol_getitem(self: Type[ArraySymbol], key: slice) -> "ArraySlice":
    ...


@overload
def _array_symbol_getitem(
    self: Type[ArraySymbol], key: tTuple[tUnion[Basic, int], ...]
) -> "ArrayElement":
    ...


@overload
def _array_symbol_getitem(
    self: Type[ArraySymbol], key: tTuple[tUnion[Basic, int, slice], ...]
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


def _normalize_index(idx: Any, axis_size: Optional[Expr]) -> Any:
    if (
        axis_size
        and axis_size.is_Integer
        and (-axis_size <= idx) == True  # noqa: E712
        and (idx < 0) == True  # noqa: E712
    ):
        return idx + axis_size
    return idx


class ArraySlice(_ArrayExpr):
    parent: Expr = property(lambda self: self.args[0])  # type: ignore[assignment]
    indices: tTuple[Tuple, ...] = property(lambda self: tuple(self.args[1]))  # type: ignore[assignment]

    def __new__(
        cls, parent: Expr, indices: tTuple[tUnion[Basic, int, slice], ...]
    ) -> "ArraySlice":
        parent_shape = get_shape(parent)
        normalized_indices = []
        for idx, axis_size in zip_longest(indices, parent_shape):
            if idx is None:
                break
            if isinstance(idx, slice):
                new_idx = Tuple(*normalize(idx, axis_size))
            else:
                new_idx = _sympify(_normalize_index(idx, axis_size))
            normalized_indices.append(new_idx)
        return Expr.__new__(cls, parent, Tuple(*normalized_indices))

    @property
    def shape(self) -> tTuple[tUnion[Basic, int], ...]:
        parent_shape = get_shape(self.parent)
        shape = [
            _compute_slice_size(idx, axis_size)
            for idx, axis_size in zip_longest(self.indices, parent_shape)
        ]
        return tuple(shape)


def _compute_slice_size(idx: Any, axis_size: Any) -> Any:  # noqa: R701
    if idx is None:
        return axis_size
    if not isinstance(idx, Tuple):
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
) -> tTuple[Basic, Basic, Basic]:
    if isinstance(i, slice):
        i = (i.start, i.stop, i.step)
    if not isinstance(i, (tuple, list, Tuple)):
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
    return fR"{parent}\left[{indices}\right]"


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
