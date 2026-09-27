"""Tests for `on_decision` and `Decision.to_dict()`: seeing what `gut` decided, and why."""

from __future__ import annotations

import enum
import json
import logging

import pytest

import gut
from gut import FakeBackend, Outcome
from gut._backends.fake import FakeValue
from gut._decision import BaseDecision
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
            return 1.5
        case _:
            return None


@pytest.fixture
def seen() -> list[BaseDecision]:
    decisions: list[BaseDecision] = []
    gut.configure(
        backend=FakeBackend(rule=_by_question_type, default=0.83),
        cache=gut.NullCache(),
        on_decision=decisions.append,
    )
    return decisions


def test_nothing_is_called_unless_asked() -> None:
    gut.configure(backend=FakeBackend(default=0.83))
    assert gut.likely("a ticket", "is a bug report")


def test_every_decision_reaches_the_hook(seen: list[BaseDecision]) -> None:
    a = gut.likely("a ticket", "is a bug report")
    b = gut.classify("a ticket", Team)
    c = gut.rate("a ticket", URGENCY)
    assert seen == [a, b, c]
    assert [type(d) for d in seen] == [gut.Decision, gut.ChoiceDecision, gut.ScoreDecision]


def test_a_yes_no_record_has_the_probability_and_what_it_meant(seen: list[BaseDecision]) -> None:
    decision = gut.likely(
        "a ticket", "the customer threatens to cancel", cost_false_yes=2, cost_false_no=50
    )
    record = decision.to_dict()
    assert record["kind"] == "likely"
    assert record["id"] == decision.id
    assert record["outcome"] == "yes"
    assert record["question"] == "the customer threatens to cancel"
    assert record["p"] == 0.83
    assert record["policy"] == "yes above 0.0385, no at or below it"
    assert record["model"] == "fake-1.0"
    assert record["source"] == "backend"
    assert record["latency_ms"] >= 0


def test_a_classification_record_names_the_member(seen: list[BaseDecision]) -> None:
    record = gut.classify("a ticket", Team).to_dict()
    assert record["kind"] == "classify"
    assert record["value"] == "BILLING"
    assert set(record["probabilities"]) == {"BILLING", "PLATFORM", "OTHER"}
    assert record["confidence"] == 0.9
    assert "question" not in record


def test_an_unsure_classification_has_no_value(seen: list[BaseDecision]) -> None:
    record = gut.classify("a ticket", Team, min_confidence=0.99).to_dict()
    assert record["outcome"] == "unsure"
    assert "value" not in record
    assert record["min_confidence"] == 0.99


def test_a_rating_record_has_the_score_and_the_nearest_level(seen: list[BaseDecision]) -> None:
    record = gut.rate("a ticket", URGENCY, question="how urgent is this").to_dict()
    assert record["kind"] == "rate"
    assert record["score"] == 1.5
    assert record["level"] == "today"
    assert record["probabilities"] == {"1": 0.5, "2": 0.5}
    assert record["question"] == "how urgent is this"


def test_records_are_json_and_leave_out_what_does_not_apply(seen: list[BaseDecision]) -> None:
    record = gut.likely("a ticket", "is a bug report").to_dict()
    json.dumps(record)
    for absent in ("score", "value", "confidence", "min_confidence"):
        assert absent not in record


def test_a_batched_decision_says_how_it_was_obtained(seen: list[BaseDecision]) -> None:
    with gut.judge("a ticket") as j:
        first = j.likely("is a bug report")
        second = j.likely("asks for a refund")
        assert first and second
    assert [d.source for d in seen] == ["prefetch", "prefetch"]
    assert "latency_ms" not in seen[0].to_dict()


def test_a_cache_hit_is_reported_as_one() -> None:
    decisions: list[BaseDecision] = []
    gut.configure(backend=FakeBackend(default=0.83), on_decision=decisions.append)
    gut.likely("same ticket", "is a bug report")
    gut.likely("same ticket", "is a bug report")
    assert [d.source for d in decisions] == ["backend", "cache"]
    assert decisions[1].latency_ms is None


def test_a_hook_that_raises_never_breaks_the_decision(caplog: pytest.LogCaptureFixture) -> None:
    def broken(decision: BaseDecision) -> None:
        raise RuntimeError("the log server is down")

    gut.configure(backend=FakeBackend(default=0.83), on_decision=broken)
    with caplog.at_level(logging.WARNING, logger="gut"):
        decision = gut.likely("a ticket", "is a bug report")
    assert decision == Outcome.YES
    assert "on_decision hook raised" in caplog.text


def test_the_base_decision_describes_itself_too() -> None:
    bare = BaseDecision(outcome=Outcome.NO, id="x", model="m")
    assert bare.to_dict() == {"kind": "decision", "id": "x", "outcome": "no", "model": "m",
                              "source": "backend"}  # fmt: skip
