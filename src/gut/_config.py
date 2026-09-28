"""Global and scoped configuration.

Four settings, and the one that matters most is what `bool(decision)` does when the decision is
UNSURE. There is no safe default answer, so `gut` refuses to pick one silently and raises instead.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, get_args

from gut._errors import ConfigurationError

if TYPE_CHECKING:
    from gut._backends.base import Backend
    from gut._cache import Cache
    from gut._decision import BaseDecision

logger = logging.getLogger("gut")

UnsureLiteral: TypeAlias = Literal["raise", "true", "false"]
"""The named policies for coercing an UNSURE decision to a bool."""

UnsurePolicy: TypeAlias = "UnsureLiteral | Callable[[BaseDecision], bool]"
"""A named policy, or a callback that decides per decision."""

DecisionHook: TypeAlias = "Callable[[BaseDecision], object]"
"""Called with every decision as it is made."""

DEFAULT_ON_UNSURE: Final[UnsureLiteral] = "raise"
"""What an unconfigured `gut` does with `bool(decision)` on an UNSURE decision."""

NO_BACKEND: Final = (
    "No backend is configured. Pick the model that answers, and the rest of your code stays the "
    "same:\n"
    "  gut.configure(backend=gut.JevBackend())                            # TypeSafe's Jev\n"
    "  gut.configure(backend=gut.ZeroShotBackend())                       # NLI model, local\n"
    '  gut.configure(backend=gut.TransformersBackend("Qwen/Qwen3-0.6B"))  # small LLM, local\n'
    '  gut.configure(backend=gut.OpenAICompatibleBackend("gpt-4.1-nano")) # or Ollama, vLLM\n'
    "  gut.configure(backend=gut.FakeBackend(answers={...}))              # tests, no model\n"
    "Setting TYPESAFE_API_KEY also selects Jev. See docs/backends.md."
)

_configured: UnsurePolicy = DEFAULT_ON_UNSURE
_override: ContextVar[UnsurePolicy | None] = ContextVar("gut_on_unsure_override", default=None)
_backend: Backend | None = None
_cache: Cache | None = None
_on_decision: DecisionHook | None = None


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
    backend: Backend | None = None,
    on_unsure: UnsurePolicy | None = None,
    cache: Cache | None = None,
    on_decision: DecisionHook | None = None,
) -> None:
    """Set process-wide defaults.

    Args:
        backend: What answers questions -- any object with `model_id` and `ask()`. Left unset,
            `gut` uses Jev when `TYPESAFE_API_KEY` is set and otherwise says what to configure.
        on_unsure: What `bool(decision)` should do when the decision is UNSURE. `"raise"` (the
            default) raises `UnsureDecision`, `"true"` and `"false"` coerce, and a callable is
            handed the decision and returns a bool.
        cache: Where answers are kept between calls. Defaults to a bounded in-memory LRU; pass
            `SQLiteCache(...)` to persist across restarts, or `NullCache()` to turn caching off.
        on_decision: Called with every decision as it is made -- the place to log, count or trace
            them. `decision.to_dict()` is a ready-made record. An exception it raises is logged
            and swallowed: observing a decision must never break it.

    Raises:
        ConfigurationError: `on_unsure` is not a recognised policy.
    """
    global _configured, _backend, _cache, _on_decision
    if on_unsure is not None:
        _configured = _validate(on_unsure)
    if backend is not None:
        _backend = backend
    if cache is not None:
        _cache = cache
    if on_decision is not None:
        _on_decision = on_decision


def current_backend() -> Backend:
    """The configured backend, falling back to Jev when the environment names a key.

    Building a `JevBackend` implicitly is a real side effect -- it starts billable calls -- so it
    only happens when `TYPESAFE_API_KEY` is set, a variable nothing but Jev reads. `OPENAI_API_KEY`
    is deliberately not treated the same way: plenty of machines have one set for other reasons,
    and finding it is not a statement that `gut` should spend it.

    Raises:
        ConfigurationError: No backend is configured and none can be inferred.
    """
    global _backend
    if _backend is None:
        _backend = _infer_backend()
    return _backend


def _infer_backend() -> Backend:
    """Build a `JevBackend` if the environment names a key, otherwise say what to configure."""
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise ConfigurationError(NO_BACKEND)
    # The module imports without the SDK; a missing extra surfaces from the constructor, with
    # an install hint attached.
    from gut._backends.jev import JevBackend

    return JevBackend()


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


def notify(decision: BaseDecision) -> None:
    """Hand a finished decision to the `on_decision` hook, if there is one."""
    if _on_decision is None:
        return
    try:
        _on_decision(decision)
    except Exception:
        logger.warning("gut: the on_decision hook raised; the decision stands", exc_info=True)


def reset_configuration() -> None:
    """Restore the unconfigured defaults. Intended for tests."""
    global _configured, _backend, _cache, _on_decision
    _configured = DEFAULT_ON_UNSURE
    _backend = None
    _cache = None
    _on_decision = None
