"""Helper functions for :func:`.perform_cached_doit`."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle  # noqa: S403
import sys

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


def get_readable_hash(obj) -> str:
    """Get a human-readable hash of any hashable Python object.

    Args:
        obj: Any hashable object, mutable or immutable, to be hashed.
    """
    b = to_bytes(obj)
    h = hashlib.md5(b)  # noqa: S324
    return h.hexdigest()


def to_bytes(obj) -> bytes:
    if isinstance(obj, bytes | bytearray):
        return obj
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
