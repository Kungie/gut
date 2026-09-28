"""Asking a backend about many subjects at once.

A backend is two members, `model_id` and `ask()`, and `ask()` takes one subject. Some backends can
do much better with many: a local model runs them through one batched forward pass, and a cascade
sends only the unsettled ones on to its next stage. Those implement an optional third member,

    def ask_many(self, items: Sequence[tuple[State, Mapping[str, QuestionSpec]]])
        -> list[BackendResponse]

returning one response per item, in order. Every other backend -- Jev, `FakeBackend`, one you
wrote -- gets its `ask()` called once per subject from a small thread pool, which is the right shape
for anything behind a network: Jev takes one subject per request, and its SDK already backs off on
a 429.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Final, TypeAlias

from gut._backends.base import Backend, BackendResponse
from gut._errors import BackendError
from gut._questions import QuestionSpec, State

Item: TypeAlias = "tuple[State, Mapping[str, QuestionSpec]]"
"""One subject and the named questions to ask about it."""

DEFAULT_CONCURRENCY: Final = 8
"""Subjects in flight at once for a backend without `ask_many`. Well under Jev's 1,200 requests a
minute, and under what a local Ollama or vLLM serves comfortably."""


def ask_all(
    backend: Backend, items: Sequence[Item], *, concurrency: int = DEFAULT_CONCURRENCY
) -> list[BackendResponse]:
    """One response per item, in order, by the fastest route the backend offers.

    Raises:
        BackendError: `ask_many` returned the wrong number of responses, or `concurrency` is not
            positive. Anything a backend raises propagates unchanged.
    """
    if concurrency < 1:
        raise BackendError(f"concurrency must be at least 1, got {concurrency}.")
    if not items:
        return []

    many = getattr(backend, "ask_many", None)
    if callable(many):
        responses = list(many(list(items)))
        if len(responses) != len(items):
            raise BackendError(
                f"{type(backend).__name__}.ask_many returned {len(responses)} responses for "
                f"{len(items)} subjects."
            )
        return responses

    workers = min(concurrency, len(items))
    if workers == 1:
        return [backend.ask(state, questions) for state, questions in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda item: backend.ask(*item), items))
