"""Helper functions for working with `sympy` expressions and `ipywidgets`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp
from sympy.printing.latex import translate

if TYPE_CHECKING:  # pragma: no cover
    import ipywidgets as w

    Slider = w.FloatSlider | w.IntSlider
    """Allowed :doc:`ipywidgets <ipywidgets:index>` slider types."""


def create_slider(symbol: sp.Symbol, **kwargs) -> Slider:
    r"""Create an `int` or `float` slider, depending on Symbol assumptions.

    The description for the slider is rendered as LaTeX from the
    `~sympy.core.symbol.Symbol` name.

    >>> create_slider(sp.Symbol("a"))
    FloatSlider(value=0.0, description='\\(a\\)')
    >>> create_slider(sp.Symbol("n0", integer=True))
    IntSlider(value=0, description='\\(n_{0}\\)')
    """
    _assert_ipywidgets_installed()
    import ipywidgets as w  # noqa: PLC0415

    description = Rf"\({sp.latex(symbol)}\)"
    if symbol.is_integer:
        return w.IntSlider(description=description, **kwargs)
    return w.FloatSlider(description=description, **kwargs)


def _assert_ipywidgets_installed() -> None:
    try:
        import ipywidgets  # noqa: F401, PLC0415
    except ImportError as exc:
        msg = "Please install ipywidgets to use the ampform.sympy.slider module"
        raise ImportError(msg) from exc


def substitute_indexed_symbols(expression: sp.Expr) -> sp.Expr:
    """Substitute `~sympy.tensor.indexed.IndexedBase` with symbols.

    See :doc:`compwa-report:008/index` for more info.
    """
    return expression.xreplace({
        s: _indexed_to_symbol(s)
        for s in expression.free_symbols
        if isinstance(s, sp.Indexed)
    })


def _indexed_to_symbol(idx: sp.Indexed) -> sp.Symbol:
    base_name, _, _ = str(idx).rpartition("[")
    subscript = ",".join(map(str, idx.indices))
    if len(idx.indices) > 1:
        base_name = translate(base_name)
        subscript = "_{" + subscript + "}"
    return sp.Symbol(f"{base_name}{subscript}", **idx.assumptions0)
