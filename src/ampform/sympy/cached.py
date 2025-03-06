"""Handy aliases for working with cached SymPy expressions."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

from frozendict import frozendict

from ampform.sympy._cache import cache_to_disk

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TypeVar

    import sympy as sp

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

    .. versionadded:: 0.14.4
    .. automodule:: ampform.sympy._cache
    """
    return expr.doit()


def xreplace(expr: sp.Expr, substitutions: Mapping[sp.Basic, sp.Basic]) -> sp.Expr:
    """Call :meth:`~sympy.core.basic.Basic.xreplace` and cache the result to disk."""
    return _xreplace_impl(expr, frozendict(substitutions))


@cache
@cache_to_disk(function_name="xreplace", dependencies=["sympy"])
def _xreplace_impl(
    expr: sp.Expr, substitutions: frozendict[sp.Basic, sp.Basic]
) -> sp.Expr:
    return expr.xreplace(substitutions)


@overload
def unfold(obj: Model) -> sp.Expr: ...
@overload
def unfold(obj: sp.Expr, substitutions: Mapping[sp.Basic, sp.Basic]) -> sp.Expr: ...
def unfold(
    obj: sp.Expr | Model, substitutions: Mapping[sp.Basic, sp.Basic] | None = None
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


def _unfold_impl(expr: sp.Expr, substitutions: Mapping[sp.Basic, sp.Basic]) -> sp.Expr:
    substitutions = _unfold_substitutions(frozendict(substitutions))
    expr = doit(expr)
    return xreplace(expr, substitutions)


@cache
def _unfold_substitutions(
    substitutions: frozendict[sp.Basic, sp.Basic],
) -> frozendict[sp.Basic, sp.Basic]:
    return frozendict({k: doit(v) for k, v in substitutions.items()})
