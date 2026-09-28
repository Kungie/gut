"""Tests for `each()`: one question, many subjects, as few model calls as the backend allows."""

from __future__ import annotations

import enum
import threading
from collections.abc import Mapping, Sequence

import pytest

import gut
from gut import BackendError, FakeBackend
from gut._backends._many import ask_all
from gut._decision import BaseDecision

SPAM = {"WIN A PRIZE": 0.97, "lunch at 1?": 0.02, "maybe a deal": 0.5}


def by_text(state: gut.State, name: str, spec: gut.QuestionSpec) -> float | str | int | None:
    match spec:
        case gut.NoulSpec():
            return SPAM.get(str(state), 0.1)
        case gut.ChoiceSpec():
            return "BILLING" if "invoice" in str(state) else "OTHER"
        case _:
            return 2 if "down" in str(state) else 0


class Team(enum.Enum):
    BILLING = "invoices"
    PLATFORM = "outages"
    OTHER = "anything else"


@pytest.fixture
def backend() -> FakeBackend:
    fake = FakeBackend(rule=by_text)
    gut.configure(backend=fake, cache=gut.NullCache())
    return fake


def test_each_answers_exactly_what_likely_would(backend: FakeBackend) -> None:
    texts = list(SPAM)
    batched = gut.each(texts).likely("is spam", ask_human=True)
    single = [gut.likely(text, "is spam", ask_human=True) for text in texts]
    assert (
        [d.outcome for d in batched]
        == [d.outcome for d in single]
        == [
            gut.YES,
            gut.NO,
            gut.UNSURE,
        ]
    )
    assert [d.p for d in batched] == [d.p for d in single]
    assert all(d.source == "batch" for d in batched)
    assert batched[0].question == "is spam"


def test_a_repeated_subject_is_asked_once(backend: FakeBackend) -> None:
    decisions = gut.each(["WIN A PRIZE", "lunch at 1?", "WIN A PRIZE"]).likely("is spam")
    assert [d.outcome for d in decisions] == [gut.YES, gut.NO, gut.YES]
    assert backend.call_count == 2


def test_cached_subjects_are_not_asked_again() -> None:
    fake = FakeBackend(rule=by_text)
    gut.configure(backend=fake)
    gut.likely("WIN A PRIZE", "is spam")
    decisions = gut.each(["WIN A PRIZE", "lunch at 1?"]).likely("is spam")
    assert [d.source for d in decisions] == ["cache", "batch"]
    assert fake.call_count == 2
    assert gut.likely("lunch at 1?", "is spam").source == "cache"
    again = gut.each(["lunch at 1?", "WIN A PRIZE"]).likely("is spam")
    assert [d.source for d in again] == ["cache", "cache"]
    assert fake.call_count == 2


def test_classify_and_rate_too(backend: FakeBackend) -> None:
    teams = gut.each(["invoice is wrong", "hello"]).classify(Team)
    assert [d.value for d in teams] == [Team.BILLING, Team.OTHER]
    urgency = gut.each(["site is down", "whenever"]).rate(["later", "soon", "now"])
    assert [d.nearest_level for d in urgency] == [2, 0]


def test_posture_words_and_their_errors_work_the_same(backend: FakeBackend) -> None:
    unsure = gut.each(["maybe a deal"]).classify(Team, stakes="high", ask_human=True)
    assert unsure[0].min_confidence == 0.8
    with pytest.raises(gut.PolicyError, match="not both"):
        gut.each(["x"]).likely("q", ask_human=True, cost_false_yes=1, cost_false_no=1)


def test_nothing_to_ask_asks_nothing(backend: FakeBackend) -> None:
    assert gut.each([]).likely("is spam") == []
    assert len(gut.each(iter(["a", "b"]))) == 2
    assert backend.call_count == 0


def test_every_decision_reaches_the_hook() -> None:
    seen: list[BaseDecision] = []
    gut.configure(
        backend=FakeBackend(rule=by_text),
        cache=gut.NullCache(),
        on_decision=seen.append,
    )
    gut.each(list(SPAM)).likely("is spam")
    assert [d.to_dict()["source"] for d in seen] == ["batch"] * 3


def test_a_backend_that_takes_one_subject_gets_them_concurrently() -> None:
    gate = threading.Barrier(4, timeout=5)

    class Slow(FakeBackend):
        def ask(self, state, questions):  # type: ignore[no-untyped-def]
            gate.wait()  # deadlocks unless four subjects are in flight at once
            return super().ask(state, questions)

    decisions = gut.each(["a", "b", "c", "d"], backend=Slow(default=0.9), concurrency=4).likely("q")
    assert len(decisions) == 4


def test_a_backend_with_ask_many_gets_everything_in_one_call() -> None:
    class Batched(FakeBackend):
        batches: list[int]

        def ask_many(self, items):  # type: ignore[no-untyped-def]
            self.batches.append(len(items))
            return [self.ask(state, questions) for state, questions in items]

    batched = Batched(default=0.9)
    batched.batches = []
    gut.configure(cache=gut.NullCache())
    gut.each(["a", "b", "c"], backend=batched).likely("q")
    assert batched.batches == [3]


def test_a_cascade_sends_only_the_unsettled_subjects_on() -> None:
    cheap = FakeBackend(rule=by_text, model="cheap")
    costly = FakeBackend(default=0.95, model="costly")
    cascade = gut.Cascade(cheap, costly)
    gut.configure(backend=cascade, cache=gut.NullCache())

    decisions = gut.each(list(SPAM)).likely("is spam")

    assert [d.model for d in decisions] == ["cheap", "cheap", "costly"]
    assert [call.state for call in costly.calls] == ["maybe a deal"]
    assert cascade.answered_by == {"cheap": 2, "costly": 1}


def test_a_failing_cascade_stage_hands_the_whole_batch_on() -> None:
    cascade = gut.Cascade(FakeBackend(model="broken"), FakeBackend(default=0.9, model="spare"))
    decisions = gut.each(["a", "b"], backend=cascade).likely("q")
    assert [d.model for d in decisions] == ["spare", "spare"]


def test_a_cascade_refuses_an_empty_question_set_in_a_batch() -> None:
    cascade = gut.Cascade(FakeBackend(default=0.9), FakeBackend(default=0.9))
    with pytest.raises(BackendError, match="at least one question"):
        cascade.ask_many([("a", {"q": gut.NoulSpec("q")}), ("b", {})])


def test_a_last_stage_that_skips_a_subject_is_named() -> None:
    class Forgetful(FakeBackend):
        def ask_many(self, items):  # type: ignore[no-untyped-def]
            return [gut.BackendResponse(answers={}, model="f") for _ in items]

    cascade = gut.Cascade(FakeBackend(default=0.5), Forgetful(default=0.9))
    with pytest.raises(BackendError, match="No stage of the cascade answered 'q'"):
        gut.each(["a"], backend=cascade).likely("q")


# --------------------------------------------------------------------------- the plumbing


def test_ask_all_checks_what_ask_many_returns() -> None:
    class Short(FakeBackend):
        def ask_many(self, items):  # type: ignore[no-untyped-def]
            return []

    with pytest.raises(BackendError, match="returned 0 responses for 1 subjects"):
        ask_all(Short(default=0.9), [("a", {"q": gut.NoulSpec("q")})])


def test_ask_all_refuses_a_concurrency_below_one() -> None:
    with pytest.raises(BackendError, match="concurrency must be at least 1"):
        ask_all(FakeBackend(default=0.9), [], concurrency=0)


def test_ask_all_with_one_worker_asks_in_order() -> None:
    fake = FakeBackend(default=0.9)
    items: Sequence[tuple[gut.State, Mapping[str, gut.QuestionSpec]]] = [
        ("a", {"q": gut.NoulSpec("q")}),
        ("b", {"q": gut.NoulSpec("q")}),
    ]
    assert len(ask_all(fake, items, concurrency=1)) == 2
    assert [call.state for call in fake.calls] == ["a", "b"]


def test_a_response_missing_the_answer_is_an_error() -> None:
    class Empty(FakeBackend):
        def ask(self, state, questions):  # type: ignore[no-untyped-def]
            return gut.BackendResponse(answers={}, model="empty")

    with pytest.raises(BackendError, match="no answer for a subject"):
        gut.each(["a"], backend=Empty()).likely("q")
