"""Input-output functions for `ampform` and `sympy` objects.

.. tip:: This function are registered with :func:`functools.singledispatch` and can be
    extended as follows:

    >>> from ampform.io import aslatex
    >>> @aslatex.register(int)
    ... def _(obj: int) -> str:
    ...     return "my custom rendering"
    >>> aslatex(1)
    'my custom rendering'
    >>> aslatex(3.4 - 2j)
    '3.4-2i'
"""

from __future__ import annotations

from collections import abc
from functools import singledispatch
from typing import Iterable, Mapping

import sympy as sp


@singledispatch
def aslatex(obj) -> str:
    """Render objects as a LaTeX `str`.

    The resulting `str` can for instance be given to `IPython.display.Math`.
    """
    return str(obj)


@aslatex.register(complex)
def _(obj: complex) -> str:
    real = __downcast(obj.real)
    imag = __downcast(obj.imag)
    plus = "+" if imag >= 0 else ""
    return f"{real}{plus}{imag}i"


def __downcast(obj: float) -> float | int:
    if obj.is_integer():
        return int(obj)
    return obj


@aslatex.register(sp.Basic)
def _(obj: sp.Basic) -> str:
    return sp.latex(obj)


@aslatex.register(abc.Mapping)
def _(obj: Mapping) -> str:
    if len(obj) == 0:
        msg = "Need at least one dictionary item"
        raise ValueError(msg)
    latex = R"\begin{array}{rcl}" + "\n"
    for lhs, rhs in obj.items():
        latex += Rf"  {aslatex(lhs)} &=& {aslatex(rhs)} \\" + "\n"
    latex += R"\end{array}"
    return latex


@aslatex.register(abc.Iterable)
def _(obj: Iterable) -> str:
    obj = list(obj)
    if len(obj) == 0:
        msg = "Need at least one item to render as LaTeX"
        raise ValueError(msg)
    latex = R"\begin{array}{c}" + "\n"
    for item in map(aslatex, obj):
        latex += Rf"  {item} \\" + "\n"
    latex += R"\end{array}"
    return latex


def improve_latex_rendering() -> None:
    """Improve LaTeX rendering of an `~sympy.tensor.indexed.Indexed` object."""

    def _print_Indexed_latex(self, printer, *args):  # noqa: N802
        base = printer._print(self.base)
        indices = ", ".join(map(printer._print, self.indices))
        return f"{base}_{{{indices}}}"

    sp.Indexed._latex = _print_Indexed_latex  # type: ignore[attr-defined]
