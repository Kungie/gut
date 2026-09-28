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

The async side mirrors it. A backend that speaks `async` natively implements `aask()` and, if it can
batch, `aask_many()`; the Jev and OpenAI-compatible backends do. Anything else is run in a worker
thread, so an event loop is never blocked by a backend that was written synchronously -- which is
also the right answer for a local model, whose work is on the CPU or GPU either way.
"""

from __future__ import annotations

import asyncio
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
    _check_concurrency(concurrency)
    if not items:
        return []

    many = getattr(backend, "ask_many", None)
    if callable(many):
        return _checked(backend, "ask_many", list(many(list(items))), len(items))

    workers = min(concurrency, len(items))
    if workers == 1:
        return [backend.ask(state, questions) for state, questions in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda item: backend.ask(*item), items))


async def aask_one(
    backend: Backend, state: State, questions: Mapping[str, QuestionSpec]
) -> BackendResponse:
    """One subject, asked natively if the backend speaks `async`, from a worker thread if not."""
    native = getattr(backend, "aask", None)
    if callable(native):
        response: BackendResponse = await native(state, questions)
        return response
    return await asyncio.to_thread(backend.ask, state, questions)


async def aask_all(
    backend: Backend, items: Sequence[Item], *, concurrency: int = DEFAULT_CONCURRENCY
) -> list[BackendResponse]:
    """`ask_all`, without blocking the event loop.

    A native `aask_many` gets everything at once; a native `aask` gets the subjects concurrently,
    `concurrency` at a time; anything else runs `ask_all` in a worker thread, so a local model still
    batches.
    """
    _check_concurrency(concurrency)
    if not items:
        return []

    many = getattr(backend, "aask_many", None)
    if callable(many):
        return _checked(backend, "aask_many", list(await many(list(items))), len(items))

    native = getattr(backend, "aask", None)
    if callable(native):
        gate = asyncio.Semaphore(concurrency)

        async def one(item: Item) -> BackendResponse:
            async with gate:
                response: BackendResponse = await native(*item)
                return response

        return list(await asyncio.gather(*(one(item) for item in items)))

    return await asyncio.to_thread(ask_all, backend, items, concurrency=concurrency)


def _check_concurrency(concurrency: int) -> None:
    if concurrency < 1:
        raise BackendError(f"concurrency must be at least 1, got {concurrency}.")


def _checked(
    backend: Backend, method: str, responses: list[BackendResponse], expected: int
) -> list[BackendResponse]:
    if len(responses) != expected:
        raise BackendError(
            f"{type(backend).__name__}.{method} returned {len(responses)} responses for "
            f"{expected} subjects."
        )
    return responses
