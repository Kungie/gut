"""Tests for answer caching: keys, the two stores, and the effect on real calls."""

from __future__ import annotations

from pathlib import Path

import pytest

import gut
from gut._backends.base import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._cache import (
    DEFAULT_MAXSIZE,
    Cache,
    CacheEntry,
    MemoryCache,
    NullCache,
    SQLiteCache,
    cache_key,
)
from gut._questions import ChoiceSpec, NoulSpec

BUG = NoulSpec("is a bug report")
OTHER = NoulSpec("asks for a refund")


def entry(p: float = 0.5, model: str = "jev-1.13.0") -> CacheEntry:
    return CacheEntry(answer=NoulAnswer(p=p), model=model)


# --------------------------------------------------------------------------- keys


def test_the_key_covers_state_question_and_model() -> None:
    base = cache_key("a ticket", BUG, "jev-1.13.0")
    assert base == cache_key("a ticket", BUG, "jev-1.13.0")
    assert base != cache_key("another ticket", BUG, "jev-1.13.0")
    assert base != cache_key("a ticket", OTHER, "jev-1.13.0")
    assert base != cache_key("a ticket", BUG, "jev-1.12.0")


def test_the_key_is_insensitive_to_dict_ordering() -> None:
    left = cache_key({"subject": "x", "body": "y"}, BUG, "m")
    right = cache_key({"body": "y", "subject": "x"}, BUG, "m")
    assert left == right


def test_question_criteria_change_the_key() -> None:
    plain = ChoiceSpec(None, {"a": None, "b": None})
    described = ChoiceSpec(None, {"a": "the first", "b": None})
    assert cache_key("s", plain, "m") != cache_key("s", described, "m")


# --------------------------------------------------------------------------- stores


@pytest.mark.parametrize("store", [MemoryCache(), NullCache(), SQLiteCache(":memory:")])
def test_every_store_satisfies_the_protocol(store: object) -> None:
    assert isinstance(store, Cache)


def test_memory_cache_round_trips() -> None:
    cache = MemoryCache()
    assert cache.get("k") is None
    cache.set("k", entry(0.9))
    stored = cache.get("k")
    assert stored is not None
    assert stored.answer == NoulAnswer(p=0.9)
    assert stored.model == "jev-1.13.0"
    assert len(cache) == 1

    cache.clear()
    assert cache.get("k") is None
    assert len(cache) == 0


def test_memory_cache_evicts_least_recently_used() -> None:
    cache = MemoryCache(maxsize=2)
    cache.set("a", entry(0.1))
    cache.set("b", entry(0.2))
    cache.get("a")  # "a" is now the most recently used, so "b" goes first
    cache.set("c", entry(0.3))

    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert len(cache) == 2


def test_memory_cache_reports_its_bound() -> None:
    assert MemoryCache(maxsize=7).maxsize == 7
    assert MemoryCache().maxsize == DEFAULT_MAXSIZE


def test_memory_cache_with_no_room_stores_nothing() -> None:
    cache = MemoryCache(maxsize=0)
    cache.set("k", entry())
    assert cache.get("k") is None
    assert len(cache) == 0


def test_memory_cache_rejects_a_negative_bound() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MemoryCache(maxsize=-1)


def test_null_cache_never_remembers() -> None:
    cache = NullCache()
    cache.set("k", entry())
    cache.clear()
    assert cache.get("k") is None


def test_sqlite_cache_survives_a_new_connection(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "answers.db"
    first = SQLiteCache(path)
    first.set("k", entry(0.77, model="jev-1.13.0"))
    first.close()

    second = SQLiteCache(path)
    stored = second.get("k")
    assert stored is not None
    assert stored.answer == NoulAnswer(p=0.77)
    assert stored.model == "jev-1.13.0"
    assert len(second) == 1
    second.close()


def test_sqlite_cache_round_trips_every_answer_kind() -> None:
    cache = SQLiteCache(":memory:")
    assert cache.get("never stored") is None
    answers: list[Answer] = [
        NoulAnswer(p=0.1),
        ChoiceAnswer(choice="a", confidence=0.6, probabilities={"a": 0.6, "b": 0.4}),
        ScoreAnswer(
            score=1.5,
            confidence=0.5,
            probabilities={1: 0.5, 2: 0.5},
            legend={0: "x", 1: "y", 2: "z"},
        ),
    ]
    for index, answer in enumerate(answers):
        cache.set(str(index), CacheEntry(answer=answer, model="m"))
    for index, answer in enumerate(answers):
        stored = cache.get(str(index))
        assert stored is not None
        assert stored.answer == answer

    cache.set("0", CacheEntry(answer=NoulAnswer(p=0.9), model="m"))  # replaces, does not duplicate
    assert len(cache) == 3
    cache.clear()
    assert len(cache) == 0
    cache.close()


# --------------------------------------------------------------------------- effect on calls


def test_a_repeated_question_does_not_reach_the_backend() -> None:
    backend = gut.FakeBackend(answers={"is a bug report": 0.91})
    gut.configure(backend=backend)

    first = gut.likely("a ticket", "is a bug report")
    second = gut.likely("a ticket", "is a bug report")

    assert backend.call_count == 1
    assert first.p == second.p == 0.91
    assert first.cached is False
    assert second.cached is True
    assert first.latency_ms is not None
    assert second.latency_ms is None
    # The model reported is still the one that answered, not the alias that was asked for.
    assert second.model == "fake-1.0"


def test_a_different_subject_is_a_different_question() -> None:
    backend = gut.FakeBackend(default=0.5)
    gut.configure(backend=backend)

    gut.likely("ticket one", "is a bug report")
    gut.likely("ticket two", "is a bug report")
    assert backend.call_count == 2


def test_a_different_model_is_not_a_cache_hit() -> None:
    gut.configure(backend=gut.FakeBackend(default=0.5, model="fake-1.0"))
    gut.likely("a ticket", "is a bug report")

    newer = gut.FakeBackend(default=0.9, model="fake-2.0")
    decision = gut.likely("a ticket", "is a bug report", backend=newer)
    assert newer.call_count == 1
    assert decision.p == 0.9
    assert decision.model == "fake-2.0"


def test_caching_can_be_turned_off() -> None:
    backend = gut.FakeBackend(answers={"is a bug report": 0.91})
    gut.configure(backend=backend, cache=NullCache())

    for _ in range(3):
        assert gut.likely("a ticket", "is a bug report").cached is False
    assert backend.call_count == 3


def test_the_default_cache_is_bounded_and_in_memory() -> None:
    gut.configure(backend=gut.FakeBackend(default=0.5))
    gut.likely("a ticket", "is a bug report")

    from gut._config import current_cache

    cache = current_cache()
    assert isinstance(cache, MemoryCache)
    assert len(cache) == 1
