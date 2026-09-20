"""Global and scoped configuration.

Right now this holds one setting, and it is the one that matters most: what `bool(decision)` does
when the decision is UNSURE. There is no safe default answer, so `gut` refuses to pick one silently
and raises instead. See D9 in DECISIONS.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, get_args

from gut._errors import ConfigurationError

if TYPE_CHECKING:
    from gut._backends.base import Backend
    from gut._cache import Cache
    from gut._decision import BaseDecision

UnsureLiteral: TypeAlias = Literal["raise", "true", "false"]
"""The named policies for coercing an UNSURE decision to a bool."""

UnsurePolicy: TypeAlias = "UnsureLiteral | Callable[[BaseDecision], bool]"
"""A named policy, or a callback that decides per decision."""

DEFAULT_ON_UNSURE: Final[UnsureLiteral] = "raise"
"""What an unconfigured `gut` does with `bool(decision)` on an UNSURE decision."""

_configured: UnsurePolicy = DEFAULT_ON_UNSURE
_override: ContextVar[UnsurePolicy | None] = ContextVar("gut_on_unsure_override", default=None)
_backend: Backend | None = None
_cache: Cache | None = None


def _validate(policy: UnsurePolicy) -> UnsurePolicy:
    if callable(policy):
        return policy
    if policy not in get_args(UnsureLiteral):
        allowed = ", ".join(repr(name) for name in get_args(UnsureLiteral))
        raise ConfigurationError(
            f"on_unsure must be one of {allowed}, or a callable taking the decision; "
            f"got {policy!r}."
        )
    return policy


def configure(
    *,
    on_unsure: UnsurePolicy | None = None,
    backend: Backend | None = None,
    cache: Cache | None = None,
) -> None:
    """Set process-wide defaults.

    Args:
        on_unsure: What `bool(decision)` should do when the decision is UNSURE. `"raise"` (the
            default) raises `UnsureDecision`, `"true"` and `"false"` coerce, and a callable is
            handed the decision and returns a bool.
        backend: What answers questions. Left unset, `gut` has no backend and says so; offline work
            uses `configure(backend=FakeBackend(...))`.
        cache: Where answers are kept between calls. Defaults to a bounded in-memory LRU; pass
            `SQLiteCache(...)` to persist across restarts, or `NullCache()` to turn caching off.

    Raises:
        ConfigurationError: `on_unsure` is not a recognised policy.
    """
    global _configured, _backend, _cache
    if on_unsure is not None:
        _configured = _validate(on_unsure)
    if backend is not None:
        _backend = backend
    if cache is not None:
        _cache = cache


def current_backend() -> Backend:
    """The configured backend.

    Raises:
        ConfigurationError: No backend has been configured.
    """
    if _backend is None:
        raise ConfigurationError(
            "No backend is configured. For offline work use "
            "gut.configure(backend=gut.FakeBackend(answers={...})); for real answers install "
            "gut[jev], set TYPESAFE_API_KEY, and configure gut.JevBackend()."
        )
    return _backend


def current_on_unsure() -> UnsurePolicy:
    """The policy in force here: a scoped override if active, otherwise the configured one."""
    override = _override.get()
    return _configured if override is None else override


@contextmanager
def on_unsure(policy: UnsurePolicy) -> Iterator[None]:
    """Override the UNSURE-to-bool policy for the duration of the block.

    The override lives in a `ContextVar`, so it follows `async` tasks and does not leak across
    threads.

    Example:
        ```python
        with gut.on_unsure("false"):
            if decision:  # an UNSURE decision reads as False here, and only here
                ...
        ```

    Raises:
        ConfigurationError: `policy` is not a recognised policy.
    """
    token = _override.set(_validate(policy))
    try:
        yield
    finally:
        _override.reset(token)


def current_cache() -> Cache:
    """The configured cache, creating the default in-memory one on first use."""
    global _cache
    if _cache is None:
        from gut._cache import MemoryCache

        _cache = MemoryCache()
    return _cache


def reset_configuration() -> None:
    """Restore the unconfigured defaults. Intended for tests."""
    global _configured, _backend, _cache
    _configured = DEFAULT_ON_UNSURE
    _backend = None
    _cache = None
