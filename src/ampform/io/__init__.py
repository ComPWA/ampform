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

import logging
import warnings
from collections import abc
from functools import singledispatch
from typing import TYPE_CHECKING

import sympy as sp

from ampform.decay import (
    IsobarNode,
    Particle,
    State,
    ThreeBodyDecay,
    ThreeBodyDecayChain,
)
from ampform.dynamics.builder import DefinedExpression

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


@singledispatch
def aslatex(obj, **kwargs) -> str:  # ruff: ignore[undocumented-param]
    """Render objects as a LaTeX `str`.

    The resulting `str` can for instance be given to `IPython.display.Math`.

    .. version-added:: 0.14.1

    Args:
        terms_per_line: If set to a non-zero, positive number,
            `sp.Expr <sympy.core.expr.Expr>` objects on the right-hand-side with multiple
            terms are split over multiple lines. The terms are split at the addition.

            .. version-added:: 0.15.2
    """
    return str(obj)


@aslatex.register(complex)
def _(obj: complex, **kwargs) -> str:
    real = __downcast(obj.real)
    imag = __downcast(obj.imag)
    plus = "+" if imag >= 0 else ""
    return f"{real}{plus}{imag}i"


def __downcast(obj: float, **kwargs) -> float:
    if isinstance(obj, float) and obj.is_integer():
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
    latex = R"\begin{aligned}" + "\n"
    latex += Rf"& {aslatex(groups[0], **kwargs)} \\" + "\n"
    for term in groups[1:]:
        latex += Rf"& \;+\; {aslatex(term, **kwargs)} \\" + "\n"
    latex += R"\end{aligned}"
    return latex


@aslatex.register(abc.Mapping)
def _(obj: Mapping, *, terms_per_line: int = 0, **kwargs) -> str:
    if len(obj) == 0:
        msg = "Need at least one dictionary item"
        raise ValueError(msg)
    latex = R"\begin{aligned}" + "\n"
    for lhs, rhs in obj.items():
        latex += _render_row(lhs, rhs, terms_per_line, **kwargs)
    latex += R"\end{aligned}"
    return latex


def _render_row(lhs, rhs, terms_per_line: int, **kwargs) -> str:
    if terms_per_line > 0 and isinstance(rhs, sp.Expr):
        n = terms_per_line
        terms = rhs.as_ordered_terms()
        terms = [sum(terms[i : i + n]) for i in range(0, len(terms), n)]
        row = _render_row(lhs, terms[0], terms_per_line=False)
        for term in terms[1:]:
            row += Rf"    \;&+\; {aslatex(term, **kwargs)} \\" + "\n"
        return row
    return Rf"  {aslatex(lhs)} \;&=\; {aslatex(rhs, **kwargs)} \\" + "\n"


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


@aslatex.register(IsobarNode)
def _(obj: IsobarNode, **kwargs) -> str:
    def render_arrow(node: IsobarNode) -> str:
        if node.interaction is None:
            return R"\to"
        return Rf"\xrightarrow[S={node.interaction.S}]{{L={node.interaction.L}}}"

    parent = aslatex(obj.parent, **kwargs)
    to = render_arrow(obj)
    child1 = aslatex(obj.child1, **kwargs)
    child2 = aslatex(obj.child2, **kwargs)
    latex = Rf"{parent} {to} {child1} {child2}"
    if isinstance(obj.parent, State):
        return latex
    return Rf"\left({latex}\right)"


@aslatex.register(ThreeBodyDecay)
def _(obj: ThreeBodyDecay, **kwargs) -> str:
    return aslatex(obj.chains, **kwargs)


@aslatex.register(ThreeBodyDecayChain)
def _(obj: ThreeBodyDecayChain, **kwargs) -> str:
    return aslatex(obj.decay, **kwargs)


@aslatex.register(Particle)
def _(obj: Particle, with_jp: bool = False, only_jp: bool = False, **kwargs) -> str:
    if only_jp:
        return _render_jp(obj)
    if with_jp:
        jp = _render_jp(obj)
        return Rf"{obj.latex}\left[{jp}\right]"
    return obj.latex


@aslatex.register(DefinedExpression)
def _(obj: DefinedExpression, **kwargs) -> str:
    latex = R"\begin{array}{rcl}" + "\n"
    expr = obj.expression
    unfolded = expr.doit(deep=False)
    if expr == unfolded:
        latex += Rf"  {aslatex(obj.expression, **kwargs)} \\" + "\n"
    else:
        latex += Rf"  {aslatex(expr)} &=& {aslatex(unfolded)} \\" + "\n"
    for lhs, rhs in obj.parameters.items():
        latex += Rf"  {aslatex(lhs)} &=& {aslatex(rhs)} \\" + "\n"
    latex += R"\end{array}"
    return latex


def _render_jp(particle: Particle) -> str:
    if particle.spin.denominator == 1:
        spin = sp.latex(particle.spin)
    else:
        spin = Rf"\frac{{{particle.spin.numerator}}}{{{particle.spin.denominator}}}"
    if particle.parity is None:
        return f"J={spin}"
    parity = "-" if particle.parity < 0 else "+"
    return f"{spin}^{parity}"


def as_markdown_table(
    obj: ThreeBodyDecay | ThreeBodyDecayChain | Sequence[Particle],
) -> str:
    """Render objects a `str` suitable for generating a table."""
    if isinstance(obj, ThreeBodyDecay):
        return _as_decay_markdown_table(obj.chains)
    item_type = _determine_item_type(obj)
    if item_type in {Particle, State}:
        return _as_resonance_markdown_table(obj)
    if item_type is ThreeBodyDecayChain:
        return _as_decay_markdown_table(obj)
    msg = (
        f"Cannot render a sequence with {item_type.__name__} items as a Markdown table"
    )
    raise NotImplementedError(msg)


def _determine_item_type(obj) -> type:
    """Determine the type of the items in a sequence.

    >>> _determine_item_type([1, 2, 3])
    <class 'int'>
    >>> _determine_item_type([True, False])
    <class 'bool'>
    >>> _determine_item_type([True, False, 1])
    <class 'int'>
    >>> _determine_item_type([3.14, 1 + 1j])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: Not all items are of the same type
    """
    if not isinstance(obj, abc.Sequence):
        return type(obj)
    if len(obj) < 1:
        msg = "Need at least one entry to render a table"
        raise ValueError(msg)
    existing_types = {type(i) for i in obj}
    existing_types = {
        typ
        for typ in existing_types
        if not any(
            typ is not other and issubclass(typ, other) for other in existing_types
        )
    }
    item_type = next(iter(existing_types))
    if len(existing_types) != 1:
        msg = f"Not all items are of type {item_type.__name__}"
        raise ValueError(msg)
    return item_type


def _as_resonance_markdown_table(items: Sequence[Particle]) -> str:
    column_names = [
        "name",
        "LaTeX",
        "$J^P$",
        "mass (MeV)",
        "width (MeV)",
    ]
    render_index = any(isinstance(i, State) for i in items)
    if render_index:
        column_names.insert(0, "index")
    src = _create_markdown_table_header(column_names)
    for particle in items:
        row_items = [
            f"`{particle.name}`",
            f"${particle.latex}$",
            Rf"${aslatex(particle, only_jp=True)}$",
            f"{int(1e3 * particle.mass):,.0f}",
            f"{int(1e3 * particle.width):,.0f}",
        ]
        if render_index:
            row_items.insert(0, particle.index if isinstance(particle, State) else " ")
        src += _create_markdown_table_row(row_items)
    return src


def _as_decay_markdown_table(decay_chains: Sequence[ThreeBodyDecayChain]) -> str:
    column_names = [
        "resonance",
        R"$J^P$",
        R"mass (MeV)",
        R"width (MeV)",
    ]
    if any(c.outgoing_ls is not None for c in decay_chains):
        column_names.append(R"$L_\mathrm{dec}^\mathrm{min}$")
    if any(c.incoming_ls is not None for c in decay_chains):
        column_names.append(R"$L_\mathrm{prod}^\mathrm{min}$")
    src = _create_markdown_table_header(column_names)
    for chain in decay_chains:
        child1, child2 = map(aslatex, chain.decay_products)
        row_items: list = [
            Rf"${chain.resonance.latex} \to {child1} {child2}$",
            Rf"${aslatex(chain.resonance, only_jp=True)}$",
            f"{int(1e3 * chain.resonance.mass):,.0f}",
            f"{int(1e3 * chain.resonance.width):,.0f}",
        ]
        if chain.outgoing_ls is not None:
            row_items.append(chain.outgoing_ls.L)
        if chain.incoming_ls is not None:
            row_items.append(chain.incoming_ls.L)
        src += _create_markdown_table_row(row_items)
    return src


def _create_markdown_table_header(column_names: list[str]):
    src = _create_markdown_table_row(column_names)
    src += _create_markdown_table_row(["---" for _ in column_names])
    return src


def _create_markdown_table_row(items: Iterable):
    return "| " + " | ".join(f"{i}" for i in items) + " |\n"


def improve_latex_rendering() -> None:
    """Improve LaTeX rendering of an `~sympy.tensor.indexed.Indexed` object.

    .. version-added:: 0.14.2
    """

    def _print_Indexed_latex(self, printer, *args) -> str:  # ruff: ignore[invalid-function-name]
        base = printer._print(self.base)
        indices = ", ".join(map(printer._print, self.indices))
        return f"{base}_{{{indices}}}"

    sp.Indexed._latex = _print_Indexed_latex  # ty: ignore[unresolved-attribute]


def mute_ampform_warnings() -> None:
    """Mute AmpForm logging and warnings about decay structures."""
    logging.getLogger("ampform.sympy._cache").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module="ampform.decay")
