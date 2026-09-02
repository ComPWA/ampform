from __future__ import annotations

from functools import lru_cache
from importlib.metadata import version


@lru_cache(maxsize=1)
def get_qrules_version() -> tuple[int, ...]:
    """Get the version of qrules as a tuple of integers.

    >>> get_qrules_version() >= (0, 10)
    True
    >>> import pytest
    >>> from ampform._qrules import get_qrules_version
    >>> if get_qrules_version() < (0, 10):
    ...     pytest.skip("Doctest only works for qrules>=0.10")
    """
    v = version("qrules")
    return tuple(int(i) for i in v.split(".") if i.strip().isdigit())
