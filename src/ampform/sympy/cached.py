"""Handy aliases for working with cached SymPy expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ampform.sympy._cache import cache_to_disk

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeVar

    import sympy as sp

    SympyObject = TypeVar("SympyObject", bound=sp.Basic)


@cache_to_disk
def doit(expr: SympyObject) -> SympyObject:
    """Perform :meth:`~sympy.core.basic.Basic.doit` and cache the result to disk.

    The cached result is fetched from disk if the hash of the original expression is the
    same as the hash embedded in the filename (see :func:`.get_readable_hash`).

    Args:
        expr: A `sympy.Expr <sympy.core.expr.Expr>` on which to call
            :meth:`~sympy.core.basic.Basic.doit`.

    .. versionadded:: 0.14.4
    .. automodule:: ampform.sympy._cache
    """
    return expr.doit()


@cache_to_disk
def xreplace(expr: sp.Expr, substitutions: Mapping[sp.Basic, sp.Basic]) -> sp.Expr:
    """Call :meth:`~sympy.core.basic.Basic.xreplace` and cache the result to disk."""
    return expr.xreplace(substitutions)


def unfold(model: Model) -> sp.Expr:
    """Insert the amplitude definitions into the intensity and unfold the expression."""
    unfolded_amplitudes = {
        symbol: doit(model.amplitudes[symbol])
        for symbol in sorted(model.amplitudes, key=str)
    }
    return doit(xreplace(model.intensity.doit(), unfolded_amplitudes))


class Model(Protocol):
    """Protocol for amplitude models of the form `.HelicityModel`."""

    @property
    def intensity(self) -> sp.Expr: ...
    @property
    def amplitudes(self) -> Mapping[sp.Basic, sp.Expr]: ...
