"""Tests for the decision log: what is recorded, where it goes, and what happens when it breaks."""

from __future__ import annotations

import enum
import json
import logging
from pathlib import Path

import pytest

import gut
from gut import FakeBackend, JSONLSink, MemorySink, NullSink, Outcome
from gut._backends.fake import FakeValue
from gut._config import current_sink, recording
from gut._log import DecisionRecord, ResolutionRecord, Sink
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
def sink() -> MemorySink:
    recorder = MemorySink()
    gut.configure(
        backend=FakeBackend(default=0.83, rule=_by_question_type),
        cache=gut.NullCache(),
        sink=recorder,
    )
    return recorder


# --------------------------------------------------------------------------- the default


def test_nothing_is_recorded_unless_asked() -> None:
    gut.configure(backend=FakeBackend(default=0.5), cache=gut.NullCache())
    assert isinstance(current_sink(), NullSink)
    assert recording() is False

    gut.likely("a ticket", "is a bug report")  # must not raise, must not write


def test_every_sink_satisfies_the_protocol() -> None:
    assert isinstance(NullSink(), Sink)
    assert isinstance(MemorySink(), Sink)


def test_the_null_sink_discards_both_kinds() -> None:
    null = NullSink()
    null.decision(
        DecisionRecord(
            id="x",
            kind="noul",
            timestamp="t",
            outcome="yes",
            model="m",
            source="backend",
            question={},
            site={},
        )
    )
    null.resolution(ResolutionRecord(id="x", timestamp="t", actual=True, decided="yes"))


# --------------------------------------------------------------------------- what is recorded


def test_a_yes_no_decision_records_its_probability_and_costs(sink: MemorySink) -> None:
    decision = gut.likely(
        "an email",
        "the customer threatens to cancel",
        cost_false_yes=2,
        cost_false_no=50,
        cost_human=5,
    )

    (record,) = sink.decisions
    assert record.id == decision.id
    assert record.kind == "noul"
    assert record.outcome == "yes"
    assert record.p == 0.83
    assert record.costs == {"cost_false_yes": 2.0, "cost_false_no": 50.0, "cost_human": 5.0}
    assert record.model == "fake-1.0"
    assert record.source == "backend"
    assert record.latency_ms is not None
    assert record.question == {"type": "noul", "instructions": "the customer threatens to cancel"}


def test_a_classification_records_the_chosen_member_and_distribution(sink: MemorySink) -> None:
    gut.classify("a ticket", Team)

    (record,) = sink.decisions
    assert record.kind == "choice"
    assert record.value == "BILLING"
    assert record.probabilities is not None
    assert set(record.probabilities) == {"BILLING", "PLATFORM", "OTHER"}
    assert record.confidence == 0.9


def test_an_unsure_classification_records_no_value(sink: MemorySink) -> None:
    gut.classify("a ticket", Team, min_confidence=0.99)

    (record,) = sink.decisions
    assert record.outcome == "unsure"
    assert record.value is None
    assert record.min_confidence == 0.99


def test_a_rating_records_its_score_and_levels(sink: MemorySink) -> None:
    gut.rate("a ticket", URGENCY)

    (record,) = sink.decisions
    assert record.kind == "score"
    assert record.score == 1.5
    assert record.probabilities == {"1": 0.5, "2": 0.5}
    assert record.question["criteria"] == URGENCY


def test_the_site_is_recorded_for_debugging(sink: MemorySink) -> None:
    gut.likely("a ticket", "is a bug report")

    (record,) = sink.decisions
    assert record.site["module"] == __name__
    assert record.site["function"] == "test_the_site_is_recorded_for_debugging"
    assert record.site["file"].endswith("test_log.py")
    assert record.site["line"] > 0


def test_a_batched_decision_records_how_it_was_obtained(sink: MemorySink) -> None:
    with gut.judge("a ticket") as j:
        bug = j.likely("is a bug report")
        j.likely("asks for a refund")
        bool(bug)

    assert [record.source for record in sink.decisions] == ["prefetch", "prefetch"]


def test_a_cache_hit_is_recorded_as_one() -> None:
    recorder = MemorySink()
    gut.configure(backend=FakeBackend(default=0.5), cache=gut.MemoryCache(), sink=recorder)
    gut.likely("a ticket", "is a bug report")
    gut.likely("a ticket", "is a bug report")

    assert [record.source for record in recorder.decisions] == ["backend", "cache"]
    assert recorder.decisions[1].latency_ms is None


def test_records_leave_out_what_does_not_apply(sink: MemorySink) -> None:
    gut.likely("a ticket", "is a bug report")
    payload = sink.decisions[0].to_json()

    assert payload["type"] == "decision"
    assert "score" not in payload
    assert "value" not in payload
    assert "confidence" not in payload


# --------------------------------------------------------------------------- resolutions


def test_resolve_records_under_the_same_id(sink: MemorySink) -> None:
    decision = gut.likely("an email", "the customer threatens to cancel")
    record = decision.resolve(actual=True, note="customer did churn")

    assert record.id == decision.id
    assert record.actual is True
    assert record.decided == decision.outcome.value
    assert record.note == "customer did churn"
    assert sink.resolutions == [record]


def test_resolve_accepts_an_enum_member(sink: MemorySink) -> None:
    decision = gut.classify("a ticket", Team)
    record = decision.resolve(actual=Team.PLATFORM)
    assert record.actual == "PLATFORM"


def test_resolve_accepts_a_level_number(sink: MemorySink) -> None:
    record = gut.rate("a ticket", URGENCY).resolve(actual=2)
    assert record.actual == 2


def test_resolve_renders_anything_else_as_text(sink: MemorySink) -> None:
    record = gut.likely("a ticket", "is a bug report").resolve(actual=object())
    assert isinstance(record.actual, str)


def test_a_resolution_always_carries_its_actual_value(sink: MemorySink) -> None:
    payload = gut.likely("a ticket", "is a bug report").resolve(actual=None).to_json()
    assert payload["type"] == "resolution"
    assert payload["actual"] is None  # present even though it is None
    assert "note" not in payload


def test_resolving_works_without_a_sink() -> None:
    gut.configure(backend=FakeBackend(default=0.5), cache=gut.NullCache())
    record = gut.likely("a ticket", "is a bug report").resolve(actual=False)
    assert record.actual is False


# --------------------------------------------------------------------------- the jsonl sink


def test_jsonl_writes_one_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "decisions.jsonl"
    with JSONLSink(path) as recorder:
        gut.configure(backend=FakeBackend(default=0.83), cache=gut.NullCache(), sink=recorder)
        decision = gut.likely("an email", "the customer threatens to cancel")
        decision.resolve(actual=True)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first, second = (json.loads(line) for line in lines)
    assert first["type"] == "decision"
    assert first["p"] == 0.83
    assert second["type"] == "resolution"
    assert second["id"] == first["id"] == decision.id


def test_jsonl_appends_across_runs(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    for _ in range(2):
        recorder = JSONLSink(path)
        gut.configure(backend=FakeBackend(default=0.5), cache=gut.NullCache(), sink=recorder)
        gut.likely("a ticket", "is a bug report")
        recorder.close()

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_a_structured_subject_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    with JSONLSink(path) as recorder:
        gut.configure(backend=FakeBackend(default=0.5), cache=gut.NullCache(), sink=recorder)
        gut.likely({"subject": "down", "body": "nothing loads"}, "is a bug report")

    assert json.loads(path.read_text(encoding="utf-8").strip())["type"] == "decision"


# --------------------------------------------------------------------------- failure


class BrokenSink:
    """A sink that raises on everything, as a misconfigured one would."""

    def decision(self, record: DecisionRecord) -> None:
        raise RuntimeError("disk on fire")

    def resolution(self, record: ResolutionRecord) -> None:
        raise RuntimeError("disk still on fire")


def test_a_broken_sink_does_not_break_the_decision(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gut.configure(backend=FakeBackend(default=0.83), cache=gut.NullCache(), sink=BrokenSink())

    with caplog.at_level(logging.WARNING, logger="gut"):
        decision = gut.likely("an email", "the customer threatens to cancel")
        record = decision.resolve(actual=True)

    assert decision.outcome is Outcome.YES
    assert decision.p == 0.83
    assert record.actual is True
    assert sum("sink" in message for message in caplog.messages) == 2


def test_memory_sink_can_be_emptied() -> None:
    recorder = MemorySink()
    gut.configure(backend=FakeBackend(default=0.5), cache=gut.NullCache(), sink=recorder)
    gut.likely("a ticket", "is a bug report").resolve(actual=True)

    assert recorder.decisions and recorder.resolutions
    recorder.clear()
    assert not recorder.decisions
    assert not recorder.resolutions
