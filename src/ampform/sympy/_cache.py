"""Helper functions for :func:`.perform_cached_doit`."""

from __future__ import annotations

import hashlib
import logging
import os
import pickle  # noqa: S403
import sys
from functools import cache, wraps
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec
    from typing import Callable, TypeVar

    P = ParamSpec("P")
    T = TypeVar("T")

_LOGGER = logging.getLogger(__name__)


def cache_to_disk(func: Callable[P, T]) -> Callable[P, T]:
    if "NO_CACHE" in os.environ:
        return func

    @wraps(func)
    def wrapped_function(*args: P.args, **kwargs: P.kwargs) -> T:
        hashable_object = (
            args,
            tuple((k, kwargs[k]) for k in sorted(kwargs)),
        )
        h = get_readable_hash(hashable_object)
        cache_file = _get_cache_dir() / f"{h}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)  # noqa: S301
        result = func(*args, **kwargs)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        msg = f"Cached expression file {cache_file} not found, performing doit()..."
        _LOGGER.warning(msg)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        return result

    return wrapped_function


@cache
def _get_cache_dir() -> Path:
    system_cache_dir = get_system_cache_directory()
    sympy_version = version("sympy")
    cache_directory = Path(system_cache_dir) / "ampform" / f"sympy-v{sympy_version}"
    cache_directory.mkdir(exist_ok=True, parents=True)
    return cache_directory


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
