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
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


@singledispatch
def aslatex(obj, **kwargs) -> str:  # noqa: D417
    """Render objects as a LaTeX `str`.

    The resulting `str` can for instance be given to `IPython.display.Math`.

    .. versionadded:: 0.14.1

    Args:
        terms_per_line: If set to a non-zero, positive number,
            `sp.Expr <sympy.core.expr.Expr>` objects on the right-hand-side with multiple
            terms are split over multiple lines. The terms are split at the addition.

            .. versionadded:: 0.15.2
    """
    return str(obj)


@aslatex.register(complex)
def _(obj: complex, **kwargs) -> str:
    real = __downcast(obj.real)
    imag = __downcast(obj.imag)
    plus = "+" if imag >= 0 else ""
    return f"{real}{plus}{imag}i"


def __downcast(obj: float, **kwargs) -> float:
    if obj.is_integer():
        return int(obj)
    return obj


@aslatex.register(str)
def _(obj: str, **kwargs) -> str:
    return obj


@aslatex.register(sp.Basic)
def _(obj: sp.Basic, **kwargs) -> str:
    return sp.latex(obj)


@aslatex.register(sp.Expr)
def _(obj: sp.Expr, *, terms_per_line: int = 0, **kwargs) -> str:
    terms = obj.as_ordered_terms()
    if terms_per_line > 0 and len(terms) > terms_per_line:
        return _render_broken_expression(terms, terms_per_line, **kwargs)
    return sp.latex(obj)


def _render_broken_expression(
    terms: Sequence[sp.Basic], terms_per_line: int, **kwargs
) -> str:
    n = terms_per_line
    groups = [sp.Add(*terms[i : i + n]) for i in range(0, len(terms), n)]
    latex = R"\begin{array}{l}" + "\n"
    latex += Rf"  {aslatex(groups[0], **kwargs)} \\" + "\n"
    for term in groups[1:]:
        latex += Rf"  \; + \; {aslatex(term, **kwargs)} \\" + "\n"
    latex += R"\end{array}"
    return latex


@aslatex.register(abc.Mapping)
def _(obj: Mapping, *, terms_per_line: int = 0, **kwargs) -> str:
    if len(obj) == 0:
        msg = "Need at least one dictionary item"
        raise ValueError(msg)
    latex = R"\begin{array}{rcl}" + "\n"
    for lhs, rhs in obj.items():
        latex += _render_row(lhs, rhs, terms_per_line, **kwargs)
    latex += R"\end{array}"
    return latex


def _render_row(lhs, rhs, terms_per_line: int, **kwargs) -> str:
    if terms_per_line > 0 and isinstance(rhs, sp.Expr):
        n = terms_per_line
        terms = rhs.as_ordered_terms()
        terms = [sum(terms[i : i + n]) for i in range(0, len(terms), n)]
        row = _render_row(lhs, terms[0], terms_per_line=False)
        for term in terms[1:]:
            row += Rf"    &+& {aslatex(term, **kwargs)} \\" + "\n"
        return row
    return Rf"  {aslatex(lhs)} &=& {aslatex(rhs, **kwargs)} \\" + "\n"


@aslatex.register(abc.Iterable)
def _(obj: Iterable, **kwargs) -> str:
    obj = list(obj)
    if len(obj) == 0:
        msg = "Need at least one item to render as LaTeX"
        raise ValueError(msg)
    latex = R"\begin{array}{c}" + "\n"
    for item in (aslatex(i, **kwargs) for i in obj):
        latex += Rf"  {item} \\" + "\n"
    latex += R"\end{array}"
    return latex


def improve_latex_rendering() -> None:
    """Improve LaTeX rendering of an `~sympy.tensor.indexed.Indexed` object.

    .. versionadded:: 0.14.2
    """

    def _print_Indexed_latex(self, printer, *args) -> str:  # noqa: N802
        base = printer._print(self.base)
        indices = ", ".join(map(printer._print, self.indices))
        return f"{base}_{{{indices}}}"

    sp.Indexed._latex = _print_Indexed_latex  # type: ignore[attr-defined]
