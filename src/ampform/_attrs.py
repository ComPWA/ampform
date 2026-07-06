"""Helper functions for constructing `attrs` decorated classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, SupportsFloat, TypeVar

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Iterable

    from attrs import Attribute

    from ampform.decay import DecayChain, LSCoupling

ChainType = TypeVar("ChainType", bound="DecayChain")


def assert_spin_value(instance, attribute: Attribute, value: sp.Rational) -> None:
    if value.denominator not in {1, 2}:
        msg = f"{attribute.name} value should be integer or half-integer, not {value}"
        raise ValueError(msg)


def to_ls(obj: LSCoupling | tuple[int, SupportsFloat] | None) -> LSCoupling | None:
    from ampform.decay import LSCoupling  # ruff: ignore[import-outside-top-level]

    if obj is None:
        return None
    if isinstance(obj, LSCoupling):
        return obj
    if isinstance(obj, tuple):
        L, S = obj
        return LSCoupling(L, S)
    msg = f"Cannot convert {type(obj).__name__} to {LSCoupling.__name__}"
    raise TypeError(msg)


def to_chains(obj: Iterable[ChainType]) -> tuple[ChainType, ...]:
    return tuple(obj)


def to_rational(obj: SupportsFloat) -> sp.Rational:
    return sp.Rational(obj)
