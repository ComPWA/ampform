"""Handy aliases for working with cached SymPy expressions.

.. autofunction:: doit
"""

# cspell:ignore srepr
from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

import sympy as sp
from frozendict import frozendict

from ampform.sympy._cache import cache_to_disk

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeVar

    SympyObject = TypeVar("SympyObject", bound=sp.Basic)


@cache
@cache_to_disk(dependencies=["sympy"])
def doit(expr: SympyObject) -> SympyObject:
    """Perform :meth:`~sympy.core.basic.Basic.doit` and cache the result to disk.

    The cached result is fetched from disk if the hash of the original expression is the
    same as the hash embedded in the filename (see :func:`.get_readable_hash`).

    Args:
        expr: A `sympy.Expr <sympy.core.expr.Expr>` on which to call
            :meth:`~sympy.core.basic.Basic.doit`.

    .. version-added:: 0.14.4
    .. automodule:: ampform.sympy._cache
    """
    return expr.doit()


@cache
@cache_to_disk
def simplify(expr: sp.Expr, *args, **kwargs) -> sp.Expr:
    """Perform :func:`~sympy.simplify.simplify.simplify` and cache the result to disk.

    .. version-added:: 0.15.7
    """
    return sp.simplify(expr, *args, **kwargs)


@cache
@cache_to_disk
def trigsimp(expr: sp.Expr, *args, **kwargs) -> sp.Expr:
    """Perform :func:`~sympy.simplify.trigsimp.trigsimp` and cache the result to disk.

    .. version-added:: 0.15.7
    """
    return sp.trigsimp(expr, *args, **kwargs)


def subs(expr: sp.Expr, substitutions: Mapping[sp.Basic, Any]) -> sp.Expr:
    """Call :meth:`~sympy.core.basic.Basic.subs` and cache the result to disk.

    The order of the substitutions does not affect the cache key.
    """
    return _subs_impl(expr, _sorted_frozendict(substitutions))


@cache
@cache_to_disk(function_name="subs", dependencies=["sympy"])
def _subs_impl(expr: sp.Expr, substitutions: frozendict[sp.Basic, Any]) -> sp.Expr:
    return expr.xreplace(substitutions)


def xreplace(expr: sp.Expr, substitutions: Mapping[sp.Basic, Any]) -> sp.Expr:
    """Call :meth:`~sympy.core.basic.Basic.xreplace` and cache the result to disk.

    The order of the substitutions does not affect the cache key.
    """
    return _xreplace_impl(expr, _sorted_frozendict(substitutions))


@cache
@cache_to_disk(function_name="xreplace", dependencies=["sympy"])
def _xreplace_impl(expr: sp.Expr, substitutions: frozendict[sp.Basic, Any]) -> sp.Expr:
    return expr.xreplace(substitutions)


@overload
def unfold(obj: Model) -> sp.Expr: ...
@overload
def unfold(obj: sp.Expr, substitutions: Mapping[sp.Basic, Any]) -> sp.Expr: ...
def unfold(
    obj: sp.Expr | Model, substitutions: Mapping[sp.Basic, Any] | None = None
) -> sp.Expr:
    """Efficiently perform both substitutions and :code:`doit()`."""
    if isinstance(obj, Model):
        return _unfold_impl(obj.intensity, obj.amplitudes)
    if substitutions is None:
        substitutions = {}
    return _unfold_impl(obj, substitutions)


@runtime_checkable
class Model(Protocol):
    @property
    def intensity(self) -> sp.Expr: ...
    @property
    def amplitudes(self) -> Mapping[sp.Basic, sp.Basic]: ...


def _unfold_impl(expr: sp.Expr, substitutions: Mapping[sp.Basic, Any]) -> sp.Expr:
    substitutions = _unfold_substitutions(_sorted_frozendict(substitutions))
    expr = doit(expr)
    return xreplace(expr, substitutions)


@cache
def _unfold_substitutions(
    substitutions: frozendict[sp.Basic, Any],
) -> frozendict[sp.Basic, Any]:
    return frozendict({k: doit(v) for k, v in substitutions.items()})


def _sorted_frozendict(
    substitutions: Mapping[sp.Basic, Any],
) -> frozendict[sp.Basic, Any]:
    """Freeze a substitution mapping in an order that does not depend on the caller.

    The disk cache is keyed on a pickle of the mapping, and a pickled mapping is written
    in iteration order. A mapping built by iterating over
    :attr:`~sympy.core.basic.Basic.free_symbols` inherits the order of that `set`, which
    depends on :code:`PYTHONHASHSEED`, so the same substitutions would otherwise get a
    different cache key in every process.

    `~sympy.core.sorting.default_sort_key` does not distinguish assumptions, which would
    leave the order of two otherwise identical symbols to the caller again, so
    :func:`~sympy.printing.repr.srepr` breaks those ties.
    """
    return frozendict(sorted(substitutions.items(), key=lambda kv: _sort_key(kv[0])))


def _sort_key(obj: Any) -> tuple[Any, str]:
    return sp.default_sort_key(obj), sp.srepr(obj)
