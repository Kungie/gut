"""Tests for the `gut eval` command line."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gut._backends.base import ChoiceAnswer, NoulAnswer
from gut._cassette import Cassette
from gut._cli import build_backend, discover, main
from gut._errors import GutError
from gut._questions import ChoiceSpec, NoulSpec

MODEL = "test-1.0"
QUESTION = "the customer threatens to cancel"

CANCEL = """
question: "the customer threatens to cancel"
min_accuracy: 0.5
examples:
  - text: "I'm cancelling."
    expected: yes
  - text: "How do I cancel?"
    expected: no
"""

TEAM = """
question: "which team owns this"
min_accuracy: 0.5
options:
  BILLING: "invoices and charges"
  PLATFORM: "outages and errors"
examples:
  - text: "charged twice"
    expected: BILLING
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A directory with one predicate file and a cassette that answers it."""
    predicates = tmp_path / "predicates"
    predicates.mkdir()
    (predicates / "cancel.yaml").write_text(CANCEL, encoding="utf-8")

    cassette = Cassette(tmp_path / "tape.json")
    spec = NoulSpec(QUESTION)
    cassette.put("I'm cancelling.", spec, MODEL, NoulAnswer(p=0.92))
    cassette.put("How do I cancel?", spec, MODEL, NoulAnswer(p=0.11))
    cassette.save()
    return tmp_path


def run(workspace: Path, *extra: str) -> int:
    return main(
        [
            "eval",
            str(workspace / "predicates"),
            "--cassette",
            str(workspace / "tape.json"),
            "--model",
            MODEL,
            *extra,
        ]
    )


# --------------------------------------------------------------------------- discovery


def test_discovery_finds_predicate_files(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(CANCEL, encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.yml").write_text(TEAM, encoding="utf-8")
    assert [p.name for p in discover([str(tmp_path)])] == ["a.yaml", "b.yml"]


def test_discovery_ignores_other_yaml(tmp_path: Path) -> None:
    """Pointing this at a repository must not complain about its CI config."""
    (tmp_path / "docker-compose.yml").write_text("services:\n  web:\n", encoding="utf-8")
    (tmp_path / "broken.yaml").write_text("question: [unclosed\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    assert discover([str(tmp_path)]) == []


def test_a_file_can_be_named_directly(tmp_path: Path) -> None:
    path = tmp_path / "a.yaml"
    path.write_text(CANCEL, encoding="utf-8")
    assert discover([str(path)]) == [path]


def test_naming_a_file_that_is_not_yaml_finds_nothing(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text(CANCEL, encoding="utf-8")
    assert discover([str(path)]) == []


# --------------------------------------------------------------------------- backends


def test_without_a_key_or_a_cassette_there_is_nothing_to_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(GutError, match="Nothing to ask"):
        build_backend(None, None)


def test_recording_without_a_key_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("GUT_RECORD", "1")
    with pytest.raises(GutError, match="needs a real model"):
        build_backend(str(tmp_path / "tape.json"), None)


def test_a_key_alone_reaches_the_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-used")
    from gut._backends.jev import JevBackend

    backend = build_backend(None, "jev-1.13.0")
    assert isinstance(backend, JevBackend)
    assert backend.model_id == "jev-1.13.0"
    backend.close()


def test_a_cassette_alone_replays(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("GUT_RECORD", raising=False)
    backend = build_backend(str(tmp_path / "tape.json"), MODEL)
    assert backend.model_id == MODEL


# --------------------------------------------------------------------------- running


def test_a_passing_predicate_exits_zero(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert run(workspace) == 0

    out = capsys.readouterr().out
    assert "PASS  cancel" in out
    assert "accuracy   100%" in out
    assert "Brier" in out
    assert "cal. error" in out
    assert "calibrating the probability" in out
    assert "1/1 predicates passed" in out


def test_too_few_examples_is_called_out(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    run(workspace)
    assert "these numbers are noise" in capsys.readouterr().out


def test_a_failing_predicate_exits_one(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    path = workspace / "predicates" / "cancel.yaml"
    path.write_text(CANCEL.replace("min_accuracy: 0.5", "min_accuracy: 1.0\nthreshold: 0.95"))

    assert run(workspace) == 1
    out = capsys.readouterr().out
    assert "FAIL  cancel" in out
    assert "wrong: expected True" in out
    assert "failed: cancel" in out


def test_max_ece_can_fail_a_file(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opt-in: a file written before anyone measured calibration must not start failing."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    path = workspace / "predicates" / "cancel.yaml"

    path.write_text(CANCEL + "max_ece: 0.5\n")
    assert run(workspace) == 0
    assert "PASS  cancel" in capsys.readouterr().out

    path.write_text(CANCEL + "max_ece: 0.01\n")
    assert run(workspace) == 1
    out = capsys.readouterr().out
    assert "FAIL  cancel" in out
    # A file that fails on calibration has to say so; accuracy was perfect.
    assert "is above this file's max_ece" in out
    assert "accuracy   100%" in out


def test_nothing_to_run_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", str(tmp_path)]) == 1
    assert "No predicate files found" in capsys.readouterr().err


def test_a_gut_error_is_reported_not_raised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    (tmp_path / "a.yaml").write_text(CANCEL, encoding="utf-8")
    assert main(["eval", str(tmp_path)]) == 2
    assert "gut: Nothing to ask" in capsys.readouterr().err


# --------------------------------------------------------------------------- output formats


def test_json_output_is_machine_readable(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    run(workspace, "--json")

    payload = json.loads(capsys.readouterr().out)
    assert payload["min_examples_for_calibration"] == 30
    (predicate,) = payload["predicates"]
    assert predicate["name"] == "cancel"
    assert predicate["kind"] == "noul"
    assert predicate["passed"] is True
    assert predicate["accuracy"] == 1.0
    assert predicate["calibrating"] == "probability"
    assert predicate["calibration"]["reliable"] is False
    assert predicate["calibration"]["bins"]


def test_a_choice_file_calibrates_its_confidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    predicates = tmp_path / "predicates"
    predicates.mkdir()
    (predicates / "team.yaml").write_text(TEAM, encoding="utf-8")

    spec = ChoiceSpec(
        "which team owns this",
        {"BILLING": "invoices and charges", "PLATFORM": "outages and errors"},
    )
    cassette = Cassette(tmp_path / "tape.json")
    cassette.put(
        "charged twice",
        spec,
        MODEL,
        ChoiceAnswer(choice="BILLING", confidence=0.8, probabilities={"BILLING": 0.8}),
    )
    cassette.save()

    assert run(tmp_path, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["predicates"][0]["calibrating"] == "confidence"


def test_bins_can_be_changed(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    run(workspace, "--json", "--bins", "10")
    payload = json.loads(capsys.readouterr().out)
    labels = [b["low"] for b in payload["predicates"][0]["calibration"]["bins"]]
    assert labels == [0.1, 0.9]


def test_plot_writes_a_diagram(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    destination = workspace / "reliability.png"
    assert run(workspace, "--plot", str(destination)) == 0
    assert destination.exists()
    assert destination.stat().st_size > 0
    assert "reliability diagram written" in capsys.readouterr().out


def test_enough_examples_drops_the_caveat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    predicates = tmp_path / "predicates"
    predicates.mkdir()
    bodies = [f"ticket number {index}" for index in range(40)]
    lines = "\n".join(f'  - text: "{body}"\n    expected: yes' for body in bodies)
    (predicates / "many.yaml").write_text(
        f'question: "{QUESTION}"\nmin_accuracy: 0.5\nexamples:\n{lines}\n', encoding="utf-8"
    )

    cassette = Cassette(tmp_path / "tape.json")
    for body in bodies:
        cassette.put(body, NoulSpec(QUESTION), MODEL, NoulAnswer(p=0.9))
    cassette.save()

    assert run(tmp_path) == 0
    out = capsys.readouterr().out
    assert "these numbers are noise" not in out
    assert "cal. error 0.100" in out  # said 0.9, happened 1.0


def test_without_a_cassette_nothing_is_saved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-cassette path: a live backend has nothing to write back."""
    import gut
    from gut import _cli

    predicates = tmp_path / "predicates"
    predicates.mkdir()
    (predicates / "cancel.yaml").write_text(CANCEL, encoding="utf-8")
    monkeypatch.setattr(_cli, "build_backend", lambda cassette, model: gut.FakeBackend(default=0.9))

    assert main(["eval", str(predicates)]) == 0
    assert "PASS  cancel" in capsys.readouterr().out


def test_the_command_needs_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([])
