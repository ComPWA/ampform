"""Helper functions for :func:`.cached.doit` and related functions.

These methods are private, but can be imported from this module:

.. code-block:: python

   import ampform.sympy._cache
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle  # noqa: S403
import re
import sys
from collections import abc
from functools import cache, wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, overload

from frozendict import frozendict

if TYPE_CHECKING:
    from collections.abc import Hashable
    from io import BufferedReader

    from _typeshed import SupportsWrite

    if sys.version_info >= (3, 11):
        from typing import ParamSpec
    else:
        from typing_extensions import ParamSpec
    from typing import Any, Callable, ParamSpec, TypeVar

    P = ParamSpec("P")
    T = TypeVar("T")

_LOGGER = logging.getLogger(__name__)


@overload
def cache_to_disk(func: Callable[P, T]) -> Callable[P, T]: ...
@overload
def cache_to_disk(
    *,
    dump_function: Callable[[Any, SupportsWrite[bytes]], None] = pickle.dump,
    load_function: Callable[[BufferedReader], Any] = pickle.load,  # noqa: S301
    dependencies: list[str] | None = None,
    function_name: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...
def cache_to_disk(
    func: Callable[P, T] | None = None,
    *,
    dump_function: Callable[[Any, SupportsWrite[bytes]], None] = pickle.dump,
    load_function: Callable[[BufferedReader], Any] = pickle.load,  # noqa: S301
    dependencies: list[str] | None = None,
    function_name: str | None = None,
):
    """Decorator for caching the result of a function to disk.

    This function works similarly to `functools.cache`, but it stores the result of the
    function to disk as a pickle file.

    .. tip::

        - Caching can be disabled by setting the environment variable :code:`NO_CACHE`.
          This can be useful to test if caches are correctly invalidated.

        - Set :code:`COMPWA_CACHE_DIR` to change the cache directory. Alternatively,
          have a look at the implementation of :func:`get_system_cache_directory` to see
          how the cache directory is determined from system environment variables.
    """
    if func is None:
        return _cache_to_disk_implementation(
            dump_function=dump_function,
            load_function=load_function,
            dependencies=dependencies,
            function_name=function_name,
        )
    return _cache_to_disk_implementation()(func)


def _cache_to_disk_implementation(
    *,
    dump_function: Callable[[Any, SupportsWrite[bytes]], None] = pickle.dump,
    load_function: Callable[[BufferedReader], Any] = pickle.load,  # noqa: S301
    dependencies: list[str] | None = None,
    function_name: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if "NO_CACHE" in os.environ:
            _warn_once("AmpForm cache disabled by NO_CACHE environment variable.")
            return func
        function_identifier = f"{func.__module__}.{func.__name__}"
        dependency_identifiers = _get_dependency_identifiers(func, dependencies or [])
        nonlocal function_name
        if function_name is None:
            function_name = func.__name__

        @wraps(func)
        def wrapped_function(*args: P.args, **kwargs: P.kwargs) -> T:
            hashable_object = make_hashable(
                function_identifier, *dependency_identifiers, args, kwargs
            )
            h = get_readable_hash(hashable_object)
            cache_file = _get_cache_dir() / h[:2] / h[2:]
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return load_function(f)
            result = func(*args, **kwargs)
            msg = f"No cache file {cache_file}, performing {function_name}()..."
            _LOGGER.warning(msg)
            cache_file.parent.mkdir(exist_ok=True, parents=True)
            with open(cache_file, "wb") as f:
                dump_function(result, f)
            return result

        return wrapped_function

    return decorator


def _get_dependency_identifiers(func: Callable, dependencies: list[str]) -> list[str]:
    dependency_identifiers = []
    if (function_package := _get_package(func)) is not None:
        dependency_identifiers.append(function_package)
    dependency_identifiers.extend(dependencies)
    return sorted(_package_with_version(p) for p in sorted(dependency_identifiers))


def _get_package(func: Callable) -> str | None:
    if "." not in func.__module__:
        return None
    return func.__module__.split(".")[0]


@cache
def _package_with_version(distribution_name: str) -> str:
    try:
        v = _remove_dev(version(distribution_name))
    except PackageNotFoundError:
        return distribution_name
    else:
        return f"{distribution_name}-{v}"


def _remove_dev(version: str) -> str:
    """Remove the ".dev" suffix from a version string.

    >>> _remove_dev("0.15.7.dev15+g3c1b3cec.d20250301")
    '0.15.7'
    >>> _remove_dev("0.15.7")
    '0.15.7'
    >>> _remove_dev("0.15.7.post1")
    '0.15.7'
    """
    return re.sub(r"(\.(dev|post).*)?$", "", version)


@cache
def _get_cache_dir() -> Path:
    if compwa_cache_dir := os.getenv("COMPWA_CACHE_DIR"):
        system_cache_dir = compwa_cache_dir
    else:
        system_cache_dir = get_system_cache_directory()
    return Path(system_cache_dir) / "ampform"


@cache
def _warn_once(msg):
    _LOGGER.warning(msg)


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


@cache
def get_readable_hash(obj: Hashable) -> str:
    """Get a human-readable hash of any hashable Python object.

    Args:
        obj: Any hashable object, mutable or immutable, to be hashed.
    """
    b = to_bytes(obj)
    h = hashlib.md5(b, usedforsecurity=False)
    return h.hexdigest()


def to_bytes(obj) -> bytes:
    """Convert any Python object to `bytes` using :func:`pickle.dumps`."""
    if isinstance(obj, (bytes, bytearray)):
        return obj
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def make_hashable(*args) -> Hashable:
    return tuple(_make_hashable_impl(x) for x in args)


def _make_hashable_impl(obj) -> Hashable:
    if isinstance(obj, abc.Mapping):
        return frozendict(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, abc.Iterable):
        hashable_items = (make_hashable(x) for x in obj)
        if isinstance(obj, abc.Sequence):
            return tuple(hashable_items)
        if isinstance(obj, set):
            return frozenset(hashable_items)
    return obj
