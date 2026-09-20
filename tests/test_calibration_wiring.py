"""Tests for corrections in the decision path, and for `gut calibrate`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gut
from gut._backends.base import ChoiceAnswer, NoulAnswer
from gut._calibrators import (
    CalibrationSet,
    Entry,
    Identity,
    Isotonic,
    Platt,
    fit,
)
from gut._cassette import Cassette
from gut._cli import main
from gut._errors import CalibrationError
from gut._questions import NoulSpec

QUESTION = "the customer threatens to cancel"
SPEC = NoulSpec(QUESTION)
MODEL = "test-1.0"


def doubling() -> CalibrationSet:
    """A correction that maps 0.4 to 0.8, for seeing the effect clearly."""
    found = CalibrationSet()
    found.add(
        Entry(
            fingerprint=SPEC.fingerprint,
            calibrator=Isotonic(points=((0.0, 0.0), (0.4, 0.8), (1.0, 1.0))),
            question=SPEC.canonical(),
            model="fake-1.0",
        )
    )
    return found


# --------------------------------------------------------------------------- the decision path


def test_without_a_calibration_nothing_changes() -> None:
    gut.configure(backend=gut.FakeBackend(default=0.4), cache=gut.NullCache())
    decision = gut.likely("a ticket", QUESTION)
    assert decision.p == 0.4
    assert decision.raw_p is None
    assert decision.outcome is gut.NO


def test_a_correction_moves_the_decision_and_keeps_the_original() -> None:
    gut.configure(
        backend=gut.FakeBackend(default=0.4), cache=gut.NullCache(), calibration=doubling()
    )
    decision = gut.likely("a ticket", QUESTION)

    assert decision.p == pytest.approx(0.8)
    assert decision.raw_p == 0.4
    # 0.4 would have been NO under p > 0.5; the corrected 0.8 is YES.
    assert decision.outcome is gut.YES


def test_a_question_with_no_correction_is_untouched() -> None:
    """Keyed per question: a correction for one predicate must not leak onto another."""
    gut.configure(
        backend=gut.FakeBackend(default=0.4), cache=gut.NullCache(), calibration=doubling()
    )
    other = gut.likely("a ticket", "asks for a refund")
    assert other.p == 0.4
    assert other.raw_p is None


def test_the_cache_stores_what_the_model_said_not_the_correction() -> None:
    """So a cached answer survives refitting the calibrator."""
    backend = gut.FakeBackend(default=0.4)
    gut.configure(backend=backend, cache=gut.MemoryCache())

    first = gut.likely("a ticket", QUESTION)
    assert first.p == 0.4

    gut.configure(calibration=doubling())
    second = gut.likely("a ticket", QUESTION)

    assert second.source == "cache"
    assert second.raw_p == 0.4
    assert second.p == pytest.approx(0.8)
    assert backend.call_count == 1


def test_a_batched_answer_is_corrected_too() -> None:
    gut.configure(
        backend=gut.FakeBackend(default=0.4), cache=gut.NullCache(), calibration=doubling()
    )
    with gut.judge("a ticket") as j:
        handle = j.likely(QUESTION)
    assert handle.p == pytest.approx(0.8)
    assert handle.source == "prefetch"


def test_confidence_is_corrected_for_a_rating() -> None:
    import enum

    class Team(enum.Enum):
        A = "the first thing"
        B = "the second thing"
        OTHER = "anything else"

    from gut._questions import ChoiceSpec

    spec = ChoiceSpec(None, {m.name: m.value for m in Team})
    calibration = CalibrationSet()
    calibration.add(
        Entry(fingerprint=spec.fingerprint, calibrator=Platt(a=1.0, b=-3.0), model="fake-1.0")
    )

    gut.configure(
        backend=gut.FakeBackend(default="A"), cache=gut.NullCache(), calibration=calibration
    )
    decision = gut.classify("a ticket", Team)
    assert decision.raw_confidence == 0.9
    assert decision.confidence < 0.9


def test_the_correction_is_recorded() -> None:
    sink = gut.MemorySink()
    gut.configure(
        backend=gut.FakeBackend(default=0.4),
        cache=gut.NullCache(),
        calibration=doubling(),
        sink=sink,
    )
    gut.likely("a ticket", QUESTION)

    (record,) = sink.decisions
    assert record.p == pytest.approx(0.8)
    assert record.raw_p == 0.4


def test_a_correction_fitted_against_another_model_warns_once() -> None:
    gut.configure(
        backend=gut.FakeBackend(default=0.4, model="other-9.0"),
        cache=gut.NullCache(),
        calibration=doubling(),
    )
    with pytest.warns(UserWarning, match="fitted against 'fake-1.0'"):
        gut.likely("a ticket", QUESTION)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gut.likely("a ticket", QUESTION)  # the same pair is not warned about twice


# --------------------------------------------------------------------------- the set


def test_a_set_round_trips_through_a_file(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    doubling().save(path)

    loaded = CalibrationSet.load(path)
    assert len(loaded) == 1
    assert loaded.for_question(SPEC.fingerprint).apply(0.4) == pytest.approx(0.8)
    assert isinstance(loaded.for_question("unknown"), Identity)

    document = json.loads(path.read_text(encoding="utf-8"))
    (entry,) = document["entries"]
    # Readable without the code that produced it.
    assert entry["question"]["instructions"] == QUESTION
    assert entry["model"] == "fake-1.0"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("{}", "no 'entries'"),
        ("[1, 2]", "no 'entries'"),
        ("not json", "cannot be read"),
        ('{"entries": [{"calibrator": {"method": "identity"}}]}', "needs a fingerprint"),
    ],
)
def test_a_corrupt_set_is_rejected(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(CalibrationError, match=message):
        CalibrationSet.load(path)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="cannot be read"):
        CalibrationSet.load(tmp_path / "nowhere.json")


# --------------------------------------------------------------------------- gut calibrate


def stretched_workspace(tmp_path: Path, count: int = 60) -> Path:
    """A predicate whose model is badly under-confident, with enough examples to fit on."""
    predicates = tmp_path / "predicates"
    predicates.mkdir()
    bodies = [f"ticket {index}" for index in range(count)]
    lines = "\n".join(
        f'  - text: "{body}"\n    expected: {"yes" if index % 2 else "no"}'
        for index, body in enumerate(bodies)
    )
    (predicates / "cancel.yaml").write_text(
        f'question: "{QUESTION}"\nmin_accuracy: 0.5\nexamples:\n{lines}\n', encoding="utf-8"
    )

    cassette = Cassette(tmp_path / "tape.json")
    for index, body in enumerate(bodies):
        # Ranks perfectly, but every probability is squashed towards the middle.
        cassette.put(body, SPEC, MODEL, NoulAnswer(p=0.55 if index % 2 else 0.45))
    cassette.save()
    return tmp_path


def calibrate(workspace: Path, *extra: str) -> int:
    return main(
        [
            "calibrate",
            str(workspace / "predicates"),
            "--cassette",
            str(workspace / "tape.json"),
            "--model",
            MODEL,
            "--out",
            str(workspace / "calibration.json"),
            *extra,
        ]
    )


def test_calibrate_fits_and_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    workspace = stretched_workspace(tmp_path)

    assert calibrate(workspace) == 0
    out = capsys.readouterr().out
    assert "out-of-fold" in out
    assert "1 correction(s) written" in out

    loaded = CalibrationSet.load(workspace / "calibration.json")
    assert len(loaded) == 1
    # The squashed 0.55 should be pushed towards 1.
    assert loaded.for_question(SPEC.fingerprint).apply(0.55) > 0.9


def test_calibrate_refuses_to_ship_a_correction_that_loses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isotonic chases noise on a small sample, and the fold split is what exposes it."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    workspace = stretched_workspace(tmp_path)
    cassette = Cassette(workspace / "tape.json")
    for index in range(60):
        # Two probability levels whose outcomes do not separate cleanly: there is a little to
        # gain in sample and nothing to gain out of it.
        cassette.put(f"ticket {index}", SPEC, MODEL, NoulAnswer(p=0.2 if index % 2 else 0.8))
    cassette.save()

    lines = "\n".join(
        f'  - text: "ticket {index}"\n    expected: {"yes" if index % 5 else "no"}'
        for index in range(60)
    )
    (workspace / "predicates" / "cancel.yaml").write_text(
        f'question: "{QUESTION}"\nmin_accuracy: 0.1\nexamples:\n{lines}\n', encoding="utf-8"
    )

    assert calibrate(workspace) == 1
    out = capsys.readouterr().out
    assert "DROPPED" in out
    assert "Nothing worth shipping" in out


def test_keep_all_ships_it_anyway(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    workspace = stretched_workspace(tmp_path)
    assert calibrate(workspace, "--keep-all") == 0


def test_calibrate_skips_a_thin_predicate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    workspace = stretched_workspace(tmp_path, count=10)

    assert calibrate(workspace) == 1
    out = capsys.readouterr().out
    assert "skipped: 10 examples" in out
    assert "Nothing worth shipping" in out


def test_platt_can_be_asked_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    workspace = stretched_workspace(tmp_path)

    calibrate(workspace, "--method", "platt", "--keep-all")
    assert "platt a=" in capsys.readouterr().out


def test_too_few_folds_falls_back_to_in_sample(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    workspace = stretched_workspace(tmp_path)

    assert calibrate(workspace, "--folds", "1") == 0
    assert "in-sample only" in capsys.readouterr().out


def test_a_mixed_run_reports_what_it_left_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One predicate ships, one is too thin, one loses out of fold."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    workspace = stretched_workspace(tmp_path)
    predicates = workspace / "predicates"
    cassette = Cassette(workspace / "tape.json")

    thin_question = "is a bug report"
    thin_spec = NoulSpec(thin_question)
    lines = "\n".join(f'  - text: "thin {index}"\n    expected: yes' for index in range(5))
    (predicates / "thin.yaml").write_text(
        f'question: "{thin_question}"\nmin_accuracy: 0.1\nexamples:\n{lines}\n', encoding="utf-8"
    )
    for index in range(5):
        cassette.put(f"thin {index}", thin_spec, MODEL, NoulAnswer(p=0.9))

    losing_question = "asks for a refund"
    losing_spec = NoulSpec(losing_question)
    lines = "\n".join(
        f'  - text: "losing {index}"\n    expected: {"yes" if index % 5 else "no"}'
        for index in range(60)
    )
    (predicates / "losing.yaml").write_text(
        f'question: "{losing_question}"\nmin_accuracy: 0.1\nexamples:\n{lines}\n',
        encoding="utf-8",
    )
    for index in range(60):
        cassette.put(f"losing {index}", losing_spec, MODEL, NoulAnswer(p=0.2 if index % 2 else 0.8))
    cassette.save()

    assert calibrate(workspace) == 0
    out = capsys.readouterr().out
    assert "1 correction(s) written" in out
    assert "skipped for want of data: thin" in out
    assert "dropped for making things worse: losing" in out


def test_calibrate_works_without_a_cassette(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from gut import _cli

    workspace = stretched_workspace(tmp_path)
    monkeypatch.setattr(
        _cli, "build_backend", lambda cassette, model: gut.FakeBackend(default=0.55)
    )
    assert (
        main(
            [
                "calibrate",
                str(workspace / "predicates"),
                "--out",
                str(workspace / "calibration.json"),
            ]
        )
        == 0
    )
    assert "correction(s) written" in capsys.readouterr().out


def test_calibrate_needs_predicate_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["calibrate", str(tmp_path)]) == 1
    assert "No predicate files found" in capsys.readouterr().err


def test_eval_can_measure_the_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    workspace = stretched_workspace(tmp_path)
    calibrate(workspace)
    capsys.readouterr()

    assert (
        main(
            [
                "eval",
                str(workspace / "predicates"),
                "--cassette",
                str(workspace / "tape.json"),
                "--model",
                MODEL,
                "--calibration",
                str(workspace / "calibration.json"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "1 correction(s)" in out
    assert "in-sample" in out


# --------------------------------------------------------------------------- fitting shapes


def test_fitting_cannot_fix_a_ranking_problem() -> None:
    """Calibration rescales; it does not reorder. Accuracy is untouched by construction."""
    pairs = [(0.9, index % 3 == 0) for index in range(60)]
    curve = fit(pairs)
    ordered_before = sorted(pairs, key=lambda pair: pair[0])
    ordered_after = sorted(
        ((curve.apply(p), event) for p, event in pairs), key=lambda pair: pair[0]
    )
    assert [event for _, event in ordered_before] == [event for _, event in ordered_after]


def test_an_answer_type_survives_correction() -> None:
    from gut._calibrators import correct

    answer = ChoiceAnswer(choice="A", confidence=0.9, probabilities={"A": 0.9, "B": 0.1})
    calibration = CalibrationSet()
    calibration.add(Entry(fingerprint="abc", calibrator=Platt(a=1.0, b=-2.0)))

    corrected, raw = correct(answer, "abc", MODEL, calibration)
    assert isinstance(corrected, ChoiceAnswer)
    assert corrected.probabilities == answer.probabilities  # only the confidence moves
    assert raw is answer
