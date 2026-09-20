"""Tests for judge(): the lazy-resolution rule, and what it costs."""

from __future__ import annotations

import enum

import pytest

import gut
from gut import FakeBackend, Outcome, judge
from gut._backends.fake import FakeValue
from gut._errors import JudgeClosedError
from gut._judge import Lazy
from gut._questions import ChoiceSpec, QuestionSpec, ScoreSpec, State

URGENCY = ["can wait", "this week", "today"]


class Team(enum.Enum):
    BILLING = "invoices and charges"
    PLATFORM = "outages and errors"
    OTHER = "anything else"


def _by_question_type(state: State, name: str, spec: QuestionSpec) -> FakeValue | None:
    match spec:
        case ChoiceSpec():
            return next(iter(spec.criteria))
        case ScoreSpec():
            return 1
        case _:
            return None


@pytest.fixture
def backend() -> FakeBackend:
    fake = FakeBackend(default=0.9, rule=_by_question_type)
    gut.configure(backend=fake, cache=gut.NullCache())
    return fake


# --------------------------------------------------------------------------- the rule


def test_everything_registered_before_the_first_read_goes_out_together(
    backend: FakeBackend,
) -> None:
    with judge("a ticket") as j:
        bug = j.likely("is a bug report")
        team = j.classify(Team)
        urgency = j.rate(URGENCY)

        assert backend.call_count == 0  # nothing has been read yet
        assert bool(bug) is True

    assert backend.call_count == 1
    assert len(backend.calls[0].names) == 3
    # The other two were answered by that same request.
    assert team.value is Team.BILLING
    assert urgency.score == 1.0
    assert backend.call_count == 1


def test_interleaving_reads_costs_a_request_each_time(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        first = j.likely("is a bug report")
        assert bool(first) is True
        second = j.likely("asks for a refund")
        assert bool(second) is True

    assert backend.call_count == 2


def test_a_question_nobody_reads_is_never_asked(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        j.likely("never read")

    assert backend.call_count == 0


def test_resolve_asks_without_reading(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        j.likely("is a bug report")
        j.likely("asks for a refund")
        j.resolve()
        assert backend.call_count == 1
        j.resolve()  # nothing left pending
        assert backend.call_count == 1


def test_handles_still_resolve_after_the_block(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        bug = j.likely("is a bug report")

    assert bool(bug) is True
    assert backend.call_count == 1


def test_registering_after_the_block_is_refused(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        pass
    with pytest.raises(JudgeClosedError, match="has ended"):
        j.likely("too late")


def test_the_request_count_is_reported(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        assert j.requests == 0
        bool(j.likely("is a bug report"))
        assert j.requests == 1
        bool(j.likely("asks for a refund"))
        assert j.requests == 2


def test_an_exception_inside_the_block_propagates(backend: FakeBackend) -> None:
    with pytest.raises(RuntimeError, match="boom"), judge("a ticket") as j:
        j.likely("is a bug report")
        raise RuntimeError("boom")


# --------------------------------------------------------------------------- the handle


def test_repr_never_resolves(backend: FakeBackend) -> None:
    """A debugger printing a handle must not spend money."""
    with judge("a ticket") as j:
        handle = j.likely("is a bug report")
        assert "pending" in repr(handle)
        assert backend.call_count == 0

        bool(handle)
        assert "pending" not in repr(handle)
        assert "Decision" in repr(handle)


def test_pending_is_readable_without_resolving(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        handle = j.likely("is a bug report")
        assert isinstance(handle, Lazy)
        assert handle.pending is True
        assert backend.call_count == 0
        assert handle.pending is not None  # still nothing asked
        bool(handle)
        assert handle.pending is False


def test_a_handle_behaves_like_the_decision(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        handle = j.likely("is a bug report")

    assert handle.p == 0.9
    assert handle.outcome is Outcome.YES
    assert handle.source == "prefetch"
    assert handle.model == "fake-1.0"
    assert handle.id
    assert handle.policy.implied_threshold == 0.5


def test_match_works_on_a_handle(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        handle = j.likely("is a bug report")

    result = "unmatched"
    match handle:
        case gut.YES:
            result = "yes"
        case gut.NO:
            result = "no"
        case gut.UNSURE:
            result = "unsure"
    assert result == "yes"


def test_handles_hash_like_their_decision(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        handle = j.likely("is a bug report")
    assert hash(handle) == hash(Outcome.YES)
    assert handle == Outcome.YES


def test_the_underlying_decision_is_reachable(backend: FakeBackend) -> None:
    """`.decision` and `.pending` live on the handle, not on the type it is declared as (D15)."""
    with judge("a ticket") as j:
        handle = j.likely("is a bug report")
    assert isinstance(handle, Lazy)
    assert isinstance(handle.decision, gut.Decision)


def test_an_unknown_attribute_still_raises(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        handle = j.likely("is a bug report")
    assert isinstance(handle, Lazy)
    with pytest.raises(AttributeError):
        handle.no_such_field  # noqa: B018


# --------------------------------------------------------------------------- arguments


def test_costs_are_applied_per_question(backend: FakeBackend) -> None:
    fake = FakeBackend(answers={"the customer threatens to cancel": 0.05})
    gut.configure(backend=fake, cache=gut.NullCache())

    with judge("an email") as j:
        cheap = j.likely("the customer threatens to cancel")
        expensive = j.likely("the customer threatens to cancel", cost_false_yes=2, cost_false_no=50)

    # Same probability, opposite decisions: the costs differ, not the answer.
    assert cheap.outcome is Outcome.NO
    assert expensive.outcome is Outcome.YES
    assert fake.call_count == 1  # and the duplicate question was asked once


def test_min_confidence_applies() -> None:
    # A score landing between two levels splits its probability, so confidence is 0.5.
    fake = FakeBackend(
        default=1.5,
        rule=lambda state, name, spec: "BILLING" if isinstance(spec, ChoiceSpec) else None,
    )
    gut.configure(backend=fake, cache=gut.NullCache())

    with judge("a ticket") as j:
        strict = j.classify(Team, min_confidence=0.99)
        rated = j.rate(URGENCY, min_confidence=0.99)

    assert strict.outcome is Outcome.UNSURE
    assert strict.value is Outcome.UNSURE
    assert rated.outcome is Outcome.UNSURE
    assert rated.confidence == 0.5
    assert rated.score == 1.5


def test_an_optional_question_is_passed_through(backend: FakeBackend) -> None:
    with judge("a ticket") as j:
        j.classify(Team, question="Which team owns this?")
        j.resolve()
    assert backend.calls[0].specs["q0"].instructions == "Which team owns this?"


def test_the_backend_can_be_overridden_for_the_block(backend: FakeBackend) -> None:
    other = FakeBackend(default=0.1, model="other-1.0")
    with judge("a ticket", backend=other) as j:
        handle = j.likely("is a bug report")

    assert handle.model == "other-1.0"
    assert other.call_count == 1
    assert backend.call_count == 0


def test_the_cache_serves_a_judge_too() -> None:
    fake = FakeBackend(default=0.9)
    gut.configure(backend=fake, cache=gut.MemoryCache())

    gut.likely("a ticket", "is a bug report")
    assert fake.call_count == 1

    with judge("a ticket") as j:
        handle = j.likely("is a bug report")
    assert handle.p == 0.9
    assert fake.call_count == 1  # answered from cache, no second request


def test_a_warning_is_still_raised_for_an_enum_without_a_catch_all(
    backend: FakeBackend,
) -> None:
    class Sentiment(enum.Enum):
        POSITIVE = "pleased"
        NEGATIVE = "annoyed"

    with pytest.warns(UserWarning, match="no catch-all member"), judge("a ticket") as j:
        j.classify(Sentiment)
