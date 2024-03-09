"""Helper functions for :func:`.perform_cached_doit`."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import pickle  # noqa: S403
import sys
from textwrap import dedent

import sympy as sp

_LOGGER = logging.getLogger(__name__)


def get_system_cache_directory() -> str:
    r"""Return the system cache directory for the current platform.

    >>> import sys
    >>> if sys.platform.startswith("darwin"):
    ...     assert get_system_cache_directory().endswith("/Library/Caches")
    >>> if sys.platform.startswith("linux"):
    ...     assert get_system_cache_directory().endswith("/.cache")
    >>> if sys.platform.startswith("win"):
    ...     assert get_system_cache_directory().endswith(R"\AppData\Local")
    """
    if sys.platform.startswith("linux"):
        cache_directory = os.getenv("XDG_CACHE_HOME")
        if cache_directory is not None:
            return cache_directory
    if sys.platform.startswith("darwin"):  # macos
        return os.path.expanduser("~/Library/Caches")
    if sys.platform.startswith("win"):
        cache_directory = os.getenv("LocalAppData")  # noqa: SIM112
        if cache_directory is not None:
            return cache_directory
        return os.path.expanduser("~/AppData/Local")
    return os.path.expanduser("~/.cache")


def get_readable_hash(obj, ignore_hash_seed: bool = False) -> str:
    """Get a human-readable hash of any hashable Python object.

    The algorithm is fastest if `PYTHONHASHSEED
    <https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED>`_ is set.
    Otherwise, it falls back to computing the hash with :func:`hashlib.sha256()`.

    Args:
        obj: Any hashable object, mutable or immutable, to be hashed.
        ignore_hash_seed: Ignore the :code:`PYTHONHASHSEED` environment variable. If
            :code:`True`, the hash seed is ignored and the hash is computed with
            :func:`hashlib.sha256`.
    """
    python_hash_seed = _get_python_hash_seed()
    if ignore_hash_seed or python_hash_seed is None:
        b = _to_bytes(obj)
        return hashlib.sha256(b).hexdigest()
    return f"pythonhashseed-{python_hash_seed}{hash(obj):+}"


def _to_bytes(obj) -> bytes:
    if isinstance(obj, sp.Expr):
        # Using the str printer is slower and not necessarily unique,
        # but pickle.dumps() does not always result in the same bytes stream.
        _warn_about_unsafe_hash()
        return str(obj).encode()
    return pickle.dumps(obj)


def _get_python_hash_seed() -> int | None:
    python_hash_seed = os.environ.get("PYTHONHASHSEED", "")
    if python_hash_seed is not None and python_hash_seed.isdigit():
        return int(python_hash_seed)
    return None


@functools.lru_cache(maxsize=None)  # warn once
def _warn_about_unsafe_hash():
    message = """
    PYTHONHASHSEED has not been set. For faster and safer hashing of SymPy expressions,
    set the PYTHONHASHSEED environment variable to a fixed value and rerun the program.
    See https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
    """
    message = dedent(message).replace("\n", " ").strip()
    _LOGGER.warning(message)
