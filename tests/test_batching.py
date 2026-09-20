"""Tests for batch planning: token estimation, splitting, and cache interaction."""

from __future__ import annotations

import pytest

import gut
from gut._backends.base import BackendResponse
from gut._backends.fake import FakeBackend
from gut._batching import (
    CONTEXT_LIMIT_TOKENS,
    estimate_tokens,
    fetch,
    plan_batches,
)
from gut._cache import MemoryCache, cache_key
from gut._errors import BackendError
from gut._questions import NoulSpec, QuestionSpec, State


def specs(count: int, *, padding: int = 0) -> list[NoulSpec]:
    filler = "x" * padding
    return [NoulSpec(f"question number {index} {filler}") for index in range(count)]


class NarrowBackend(FakeBackend):
    """A backend that declares a very small context window."""

    context_limit_tokens = 200


# --------------------------------------------------------------------------- estimation


def test_estimation_scales_with_size() -> None:
    assert estimate_tokens("x" * 400) > estimate_tokens("x" * 40)
    assert estimate_tokens("") >= 1  # never zero, so a question always costs something


def test_the_default_limit_is_the_documented_one() -> None:
    assert CONTEXT_LIMIT_TOKENS == 64_000


# --------------------------------------------------------------------------- planning


def test_nothing_to_ask_is_no_batches() -> None:
    assert plan_batches("a ticket", {}, FakeBackend()) == []


def test_a_small_set_stays_in_one_batch() -> None:
    batches = plan_batches("a ticket", {f"q{i}": s for i, s in enumerate(specs(5))}, FakeBackend())
    assert len(batches) == 1
    assert len(batches[0]) == 5


def test_a_batch_too_large_for_the_window_is_split() -> None:
    questions = {f"q{i}": s for i, s in enumerate(specs(8, padding=400))}
    batches = plan_batches("a ticket", questions, NarrowBackend())

    assert len(batches) > 1
    # Nothing is lost or duplicated in the split.
    packed = [name for batch in batches for name in batch]
    assert sorted(packed) == sorted(questions)
    assert len(packed) == len(set(packed))


def test_a_single_oversized_question_is_still_asked() -> None:
    """Dropping a question the caller asked for would be worse than an error they can read."""
    huge = {"q0": NoulSpec("y" * 20_000)}
    batches = plan_batches("a ticket", huge, NarrowBackend())
    assert batches == [huge]


def test_a_backend_without_declared_limits_uses_the_documented_ceiling() -> None:
    questions = {f"q{i}": s for i, s in enumerate(specs(8, padding=400))}
    assert len(plan_batches("a ticket", questions, FakeBackend())) == 1


# --------------------------------------------------------------------------- fetching


def test_fetch_answers_everything_in_one_call() -> None:
    backend = FakeBackend(default=0.5)
    gut.configure(cache=gut.NullCache())
    resolved = fetch("a ticket", specs(4), backend)

    assert len(resolved) == 4
    assert backend.call_count == 1


def test_fetch_asks_a_repeated_question_once() -> None:
    backend = FakeBackend(default=0.5)
    gut.configure(cache=gut.NullCache())
    question = NoulSpec("is a bug report")
    resolved = fetch("a ticket", [question, question, question], backend)

    assert len(resolved) == 1
    assert len(backend.calls[0].names) == 1


def test_fetch_uses_the_cache_and_only_asks_for_the_rest() -> None:
    cache = MemoryCache()
    gut.configure(cache=cache)
    backend = FakeBackend(default=0.5)
    wanted = specs(3)

    fetch("a ticket", wanted[:2], backend)
    backend.reset()

    resolved = fetch("a ticket", wanted, backend)
    assert len(resolved) == 3
    assert backend.call_count == 1
    assert len(backend.calls[0].names) == 1  # only the one that was not already cached


def test_fetch_writes_what_it_learns_into_the_cache() -> None:
    cache = MemoryCache()
    gut.configure(cache=cache)
    backend = FakeBackend(default=0.5)
    question = NoulSpec("is a bug report")

    fetch("a ticket", [question], backend)
    assert cache.get(cache_key("a ticket", question, backend.model_id)) is not None


def test_fetch_reports_a_backend_that_skips_a_question() -> None:
    class ForgetfulBackend:
        model_id = "forgetful-1.0"

        def ask(self, state: State, questions: dict[str, QuestionSpec]) -> BackendResponse:
            first = next(iter(questions))
            inner = FakeBackend(default=0.5).ask(state, {first: questions[first]})
            return BackendResponse(answers=inner.answers, model="forgetful-1.0")

    gut.configure(cache=gut.NullCache())
    with pytest.raises(BackendError, match="answered without"):
        fetch("a ticket", specs(2), ForgetfulBackend())  # type: ignore[arg-type]
