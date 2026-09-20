"""The prefetch scope: where batched answers wait for the code that asks for them.

`@semantic` and `judge()` both work the same way. They answer a set of questions up front in one
call, put the answers here, and then run the caller's code unchanged. When that code reaches a
`likely(...)`, the answer is already sitting in the scope and no request happens.

The scope lives in a `ContextVar`, so it follows `async` tasks and never leaks between threads. It
is keyed by the state *and* the question, not by the question alone: the same predicate asked about
two different subjects inside one scope must not collide.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from gut._cache import CacheEntry
from gut._questions import QuestionSpec, State, state_fingerprint

ScopeKey = tuple[str, str]
"""`(state fingerprint, question fingerprint)`."""


def scope_key(state: State, spec: QuestionSpec) -> ScopeKey:
    """The identity a prefetched answer is stored under."""
    return (state_fingerprint(state), spec.fingerprint)


@dataclass
class Prefetch:
    """Answers fetched ahead of the code that will ask for them."""

    entries: dict[ScopeKey, CacheEntry] = field(default_factory=dict)
    hits: int = 0
    """How many questions were served from here rather than from the backend."""
    misses: int = 0
    """How many questions were asked that nobody prefetched -- each one a separate call."""

    def take(self, state: State, spec: QuestionSpec) -> CacheEntry | None:
        """Return the prefetched answer for this question, or `None`.

        Answers are not consumed: the same question asked twice inside one scope is served twice.
        """
        entry = self.entries.get(scope_key(state, spec))
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def add(self, state: State, spec: QuestionSpec, entry: CacheEntry) -> None:
        """Store a prefetched answer."""
        self.entries[scope_key(state, spec)] = entry


_active: ContextVar[Prefetch | None] = ContextVar("gut_prefetch", default=None)


def current_prefetch() -> Prefetch | None:
    """The prefetch in force here, if any."""
    return _active.get()


@contextmanager
def prefetch_scope(prefetch: Prefetch) -> Iterator[Prefetch]:
    """Make `prefetch` the active scope for the duration of the block.

    Scopes nest: an inner scope shadows an outer one completely rather than merging with it, so a
    decorated function calling another decorated function behaves exactly as it would alone.
    """
    token = _active.set(prefetch)
    try:
        yield prefetch
    finally:
        _active.reset(token)
