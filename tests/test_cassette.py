"""Tests for record/replay cassettes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gut._backends.base import NoulAnswer
from gut._backends.fake import FakeBackend
from gut._cassette import Cassette, CassetteBackend, record_requested
from gut._errors import CassetteMissError
from gut._questions import ChoiceSpec, NoulSpec

BUG = NoulSpec("is a bug report")
REFUND = NoulSpec("asks for a refund")
TEAM = ChoiceSpec("which team", {"BILLING": "invoices", "PLATFORM": "outages"})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("", False),
        ("no", False),
        ("  ", False),
    ],
)
def test_the_environment_decides_the_mode(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    monkeypatch.setenv("GUT_RECORD", value)
    assert record_requested() is expected


def test_unset_means_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GUT_RECORD", raising=False)
    assert record_requested() is False


# --------------------------------------------------------------------------- round trip


def test_recording_then_replaying_gives_the_same_answers(tmp_path: Path) -> None:
    path = tmp_path / "cassettes" / "tape.json"
    live = FakeBackend(answers={"is a bug report": 0.91, "asks for a refund": 0.2})

    with CassetteBackend(path, live, record=True) as recorder:
        recorded = recorder.ask("a ticket", {"a": BUG, "b": REFUND})
    assert live.call_count == 1
    assert path.exists()

    replay = CassetteBackend(path, record=False, model=live.model_id)
    replayed = replay.ask("a ticket", {"a": BUG, "b": REFUND})

    assert replayed.answers == recorded.answers
    assert live.call_count == 1  # nothing reached the real backend the second time


def test_entries_are_keyed_per_question_not_per_batch(tmp_path: Path) -> None:
    """A cassette recorded before batching changed must still replay afterwards."""
    path = tmp_path / "tape.json"
    live = FakeBackend(default=0.5)

    with CassetteBackend(path, live, record=True) as recorder:
        recorder.ask("a ticket", {"a": BUG})
        recorder.ask("a ticket", {"b": REFUND})
    assert live.call_count == 2

    # Now ask both together, and under different names.
    replay = CassetteBackend(path, record=False, model=live.model_id)
    response = replay.ask("a ticket", {"first": BUG, "second": REFUND})
    assert set(response.answers) == {"first", "second"}


def test_a_recording_run_only_asks_for_what_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "tape.json"
    live = FakeBackend(default=0.5)

    with CassetteBackend(path, live, record=True) as recorder:
        recorder.ask("a ticket", {"a": BUG})
    live.reset()

    with CassetteBackend(path, live, record=True) as recorder:
        recorder.ask("a ticket", {"a": BUG, "b": REFUND})

    assert live.call_count == 1
    assert len(live.calls[0].names) == 1  # only the new question


def test_every_question_kind_survives_the_trip(tmp_path: Path) -> None:
    path = tmp_path / "tape.json"
    live = FakeBackend(default=0.5, rule=lambda s, n, spec: "BILLING" if spec is TEAM else None)

    with CassetteBackend(path, live, record=True) as recorder:
        recorded = recorder.ask("a ticket", {"a": BUG, "t": TEAM})

    replay = CassetteBackend(path, record=False, model=live.model_id)
    assert replay.ask("a ticket", {"a": BUG, "t": TEAM}).answers == recorded.answers


# --------------------------------------------------------------------------- misses


def test_replaying_something_unrecorded_fails_loudly(tmp_path: Path) -> None:
    """Quietly reaching for the network would make CI and a laptop disagree, and bill both."""
    path = tmp_path / "tape.json"
    with CassetteBackend(path, FakeBackend(default=0.5), record=True) as recorder:
        recorder.ask("a ticket", {"a": BUG})

    replay = CassetteBackend(path, record=False, model="fake-1.0")
    with pytest.raises(CassetteMissError, match="asks for a refund"):
        replay.ask("a ticket", {"b": REFUND})


def test_the_miss_message_says_how_to_fix_it(tmp_path: Path) -> None:
    replay = CassetteBackend(tmp_path / "missing.json", record=False)
    with pytest.raises(CassetteMissError, match="GUT_RECORD=1"):
        replay.ask("a ticket", {"a": BUG})


def test_a_changed_subject_is_a_miss(tmp_path: Path) -> None:
    path = tmp_path / "tape.json"
    with CassetteBackend(path, FakeBackend(default=0.5), record=True) as recorder:
        recorder.ask("a ticket", {"a": BUG})

    replay = CassetteBackend(path, record=False, model="fake-1.0")
    with pytest.raises(CassetteMissError):
        replay.ask("a different ticket", {"a": BUG})


def test_a_different_model_is_a_miss(tmp_path: Path) -> None:
    path = tmp_path / "tape.json"
    with CassetteBackend(path, FakeBackend(default=0.5), record=True) as recorder:
        recorder.ask("a ticket", {"a": BUG})

    with pytest.raises(CassetteMissError):
        CassetteBackend(path, record=False, model="jev-2.0.0").ask("a ticket", {"a": BUG})


def test_recording_without_a_real_backend_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CassetteMissError, match="needs a real backend"):
        CassetteBackend(tmp_path / "tape.json", record=True)


# --------------------------------------------------------------------------- the file


def test_the_file_is_readable_and_diffable(tmp_path: Path) -> None:
    path = tmp_path / "tape.json"
    with CassetteBackend(path, FakeBackend(answers={"is a bug report": 0.91}), record=True) as r:
        r.ask("a ticket", {"a": BUG})

    document = json.loads(path.read_text(encoding="utf-8"))
    (entry,) = document["entries"]
    assert entry["state"] == "a ticket"
    assert entry["answer"] == {"type": "noul", "p": 0.91}
    assert entry["model"] == "fake-1.0"
    # The question is written once into a table and referred to, so a large one is not repeated
    # per entry. Still readable: the table is right there in the same file.
    assert document["questions"][entry["question"]] == {
        "type": "noul",
        "instructions": "is a bug report",
    }


def test_a_repeated_question_is_stored_once(tmp_path: Path) -> None:
    """A 151-option Choice is 11 KB; repeating it per entry made one recording 8 MB."""
    path = tmp_path / "tape.json"
    with CassetteBackend(path, FakeBackend(default=0.5), record=True) as r:
        for index in range(20):
            r.ask(f"ticket {index}", {"a": BUG})

    document = json.loads(path.read_text(encoding="utf-8"))
    assert len(document["entries"]) == 20
    assert len(document["questions"]) == 1


def test_a_version_one_cassette_still_loads(tmp_path: Path) -> None:
    """Questions used to live inside each entry. Those files must keep working."""
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "state": "a ticket",
                        "question": BUG.canonical(),
                        "model": "fake-1.0",
                        "answer": {"type": "noul", "p": 0.42},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cassette = Cassette(path)
    assert cassette.get("a ticket", BUG, "fake-1.0") == NoulAnswer(p=0.42)

    # And re-saving migrates it.
    cassette.put("another", BUG, "fake-1.0", NoulAnswer(p=0.1))
    cassette.save()
    assert "questions" in json.loads(path.read_text(encoding="utf-8"))


def test_entries_are_written_in_a_stable_order(tmp_path: Path) -> None:
    path = tmp_path / "tape.json"
    live = FakeBackend(default=0.5)
    with CassetteBackend(path, live, record=True) as r:
        r.ask("b ticket", {"x": REFUND})
        r.ask("a ticket", {"y": BUG})
    first = path.read_text(encoding="utf-8")

    path.unlink()
    with CassetteBackend(path, live, record=True) as r:
        r.ask("a ticket", {"y": BUG})
        r.ask("b ticket", {"x": REFUND})
    assert path.read_text(encoding="utf-8") == first


def test_nothing_is_written_when_nothing_was_recorded(tmp_path: Path) -> None:
    path = tmp_path / "tape.json"
    with CassetteBackend(path, FakeBackend(default=0.5), record=True) as r:
        r.ask("a ticket", {"a": BUG})
    written = path.stat().st_mtime_ns

    replay = CassetteBackend(path, record=False, model="fake-1.0")
    replay.ask("a ticket", {"a": BUG})
    replay.save()
    assert path.stat().st_mtime_ns == written


def test_a_cassette_can_be_used_directly(tmp_path: Path) -> None:
    cassette = Cassette(tmp_path / "tape.json")
    assert len(cassette) == 0
    assert cassette.get("a ticket", BUG, "m") is None

    cassette.put("a ticket", BUG, "m", NoulAnswer(p=0.42))
    assert len(cassette) == 1
    # Read both before asserting: mypy narrows an attribute on `is True` and never widens it again.
    dirty_after_put = cassette.dirty
    cassette.save()
    assert (dirty_after_put, cassette.dirty) == (True, False)

    reloaded = Cassette(tmp_path / "tape.json")
    assert reloaded.get("a ticket", BUG, "m") == NoulAnswer(p=0.42)


def test_a_cassette_object_can_be_handed_over(tmp_path: Path) -> None:
    cassette = Cassette(tmp_path / "tape.json")
    cassette.put("a ticket", BUG, "fake-1.0", NoulAnswer(p=0.7))
    backend = CassetteBackend(cassette, record=False, model="fake-1.0")
    assert backend.model_id == "fake-1.0"
    assert backend.ask("a ticket", {"a": BUG}).answers["a"] == NoulAnswer(p=0.7)


def test_the_model_defaults_to_the_inner_backend(tmp_path: Path) -> None:
    backend = CassetteBackend(tmp_path / "tape.json", FakeBackend(model="fake-9.0"), record=True)
    assert backend.model_id == "fake-9.0"


def test_replaying_without_a_model_says_so(tmp_path: Path) -> None:
    assert CassetteBackend(tmp_path / "tape.json", record=False).model_id == "cassette"
