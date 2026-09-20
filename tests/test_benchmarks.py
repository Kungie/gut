"""Tests for the benchmark machinery.

The benchmarks talk to a real model over the network, so what is tested here is everything that
does not: the sampling, the scoring, and the guards. A wrong split or a wrong error rate would be
invisible in the output and would quietly invalidate every number in `docs/benchmarks.md`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from benchmarks._core import (
    Asked,
    Example,
    RunStats,
    Source,
    ask_all,
    check_budget,
    estimate_tokens,
    fetch,
    stratified_split,
)
from benchmarks._datasets import benchmarks
from benchmarks._metrics import (
    forced_choice,
    keyword_rule,
    out_of_scope,
    pairs_binary,
    pairs_choice,
    sweep_binary,
    sweep_choice,
    threshold_binary,
)

from gut._backends.base import ChoiceAnswer, NoulAnswer
from gut._calibration import calibrate
from gut._questions import NoulSpec

SPEC = NoulSpec("is a bug report")


def pool(counts: dict[str, int]) -> list[Example]:
    return [
        Example(text=f"{label}-{index}", label=label)
        for label, count in counts.items()
        for index in range(count)
    ]


def noul(probability: float, label: str) -> Asked:
    return Asked(
        example=Example(text="x", label=label),
        answer=NoulAnswer(p=probability),
        latency_ms=1.0,
    )


def choice(chosen: str, confidence: float, label: str) -> Asked:
    return Asked(
        example=Example(text="x", label=label),
        answer=ChoiceAnswer(
            choice=chosen, confidence=confidence, probabilities={chosen: confidence}
        ),
        latency_ms=1.0,
    )


if TYPE_CHECKING:
    from benchmarks._run import Result


# --------------------------------------------------------------------------- sampling


def test_the_split_is_deterministic() -> None:
    examples = pool({"a": 300, "b": 300})
    first = stratified_split(examples, dev_size=40, test_size=100)
    second = stratified_split(examples, dev_size=40, test_size=100)
    assert [e.text for e in first.dev] == [e.text for e in second.dev]
    assert [e.text for e in first.test] == [e.text for e in second.test]


def test_a_different_seed_draws_a_different_sample() -> None:
    examples = pool({"a": 300, "b": 300})
    first = stratified_split(examples, dev_size=40, test_size=100, seed=1)
    second = stratified_split(examples, dev_size=40, test_size=100, seed=2)
    assert [e.text for e in first.test] != [e.text for e in second.test]


def test_dev_and_test_never_overlap() -> None:
    """The whole method rests on this: wording is written on dev, test is seen once."""
    split = stratified_split(pool({"a": 300, "b": 300}), dev_size=40, test_size=100)
    assert not {e.text for e in split.dev} & {e.text for e in split.test}


def test_proportions_are_kept() -> None:
    split = stratified_split(pool({"common": 900, "rare": 100}), dev_size=100, test_size=200)
    counts = split.counts()
    assert counts["test"]["common"] == pytest.approx(180, abs=2)
    assert counts["test"]["rare"] == pytest.approx(20, abs=2)


def test_a_quota_overrides_the_natural_share() -> None:
    """CLINC needs this: out-of-scope is 5% of the corpus and ~18% of what is worth measuring."""
    split = stratified_split(
        pool({"in": 950, "oos": 200}), dev_size=100, test_size=200, quota={"oos": (20, 50)}
    )
    counts = split.counts()
    assert counts["test"]["oos"] == 50
    assert counts["dev"]["oos"] == 20


def test_every_label_survives_even_when_rare() -> None:
    split = stratified_split(pool({"a": 5000, "b": 3}), dev_size=50, test_size=100)
    assert split.counts()["test"]["b"] >= 1


# --------------------------------------------------------------------------- fetching


def test_a_changed_source_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A benchmark whose inputs can change underneath it is not a benchmark."""
    from benchmarks import _core

    monkeypatch.setattr(_core, "DATA_DIR", tmp_path)
    payload = b"the file as it is today"

    class Response:
        content = payload

        def raise_for_status(self) -> Response:
            return self

    monkeypatch.setattr("httpx.get", lambda *a, **k: Response())

    good = Source(url="https://example.test/x.json", sha256=hashlib.sha256(payload).hexdigest())
    assert fetch(good) == payload

    stale = Source(url="https://example.test/y.json", sha256="0" * 64)
    with pytest.raises(ValueError, match="re-pin it deliberately"):
        fetch(stale)
    # The bad download is not left in the cache to be trusted next time.
    assert not (tmp_path / stale.filename).exists()


# --------------------------------------------------------------------------- budget


def test_the_budget_guard_refuses_an_expensive_run() -> None:
    huge = [Example(text="x" * 100_000, label="a") for _ in range(1000)]
    with pytest.raises(RuntimeError, match="exceeds the"):
        check_budget(huge, SPEC, budget=0.001)


def test_a_cheap_run_is_allowed() -> None:
    assert check_budget(pool({"a": 10}), SPEC, budget=1.0) < 1.0


def test_the_estimate_counts_the_question_too() -> None:
    short = estimate_tokens(pool({"a": 10}), NoulSpec("q"))
    long = estimate_tokens(pool({"a": 10}), NoulSpec("q" * 400))
    assert long > short


# --------------------------------------------------------------------------- scoring


def test_a_posture_trades_coverage_for_errors() -> None:
    """The claim the whole benchmark exists to test, on numbers chosen to make it visible."""
    asked = [noul(0.95, "bug")] * 10 + [noul(0.05, "other")] * 10 + [noul(0.5, "bug")] * 10
    rows = {row.label.strip(): row for row in sweep_binary(asked, "bug")}

    wide_open = rows["no arguments"]
    assert wide_open.coverage == 1.0

    careful = rows["stakes=high   lean=none"]
    assert careful.coverage < wide_open.coverage
    assert careful.error_rate < wide_open.error_rate
    # The ten it declined are exactly the ambiguous ones.
    assert careful.human == 10


def test_the_hard_threshold_decides_everything() -> None:
    asked = [noul(0.95, "bug"), noul(0.5, "bug"), noul(0.05, "other")]
    row = threshold_binary(asked, "bug")
    assert row.coverage == 1.0
    assert row.false_no == 1  # 0.5 is below 0.7, and it was a bug
    assert row.false_yes == 0


def test_false_positives_and_negatives_are_kept_apart() -> None:
    asked = [noul(0.99, "other"), noul(0.01, "bug")]
    row = threshold_binary(asked, "bug")
    assert (row.false_yes, row.false_no) == (1, 1)


def test_a_choice_posture_declines_on_low_confidence() -> None:
    asked = [choice("A", 0.99, "A"), choice("B", 0.3, "A")]
    rows = {row.label.strip(): row for row in sweep_choice(asked)}
    assert rows["no arguments"].automatic == 2
    assert rows["no arguments"].wrong == 1
    assert rows["stakes=medium lean=none"].automatic == 1
    assert rows["stakes=medium lean=none"].wrong == 0


def test_choosing_the_catch_all_counts_as_declining() -> None:
    asked = [choice("OTHER", 0.99, "A"), choice("A", 0.99, "A")]
    rows = {row.label.strip(): row for row in sweep_choice(asked, abstain_label="OTHER")}
    assert rows["no arguments"].automatic == 1
    assert rows["no arguments"].wrong == 0


def test_forced_choice_is_the_classifier_baseline() -> None:
    asked = [choice("A", 0.2, "A"), choice("B", 0.9, "A")]
    row = forced_choice(asked)
    assert row.coverage == 1.0
    assert row.wrong == 1


def test_a_keyword_rule_is_scored_the_same_way() -> None:
    asked = [noul(0.0, "bug"), noul(0.0, "other")]
    asked = [
        Asked(
            example=Example(text="it crashes", label="bug"), answer=asked[0].answer, latency_ms=1
        ),
        Asked(
            example=Example(text="how do I", label="other"), answer=asked[1].answer, latency_ms=1
        ),
    ]
    row = keyword_rule(asked, lambda text: "crash" in text, "bug", label="regex")
    assert row.wrong == 0
    assert row.coverage == 1.0


# --------------------------------------------------------------------------- out of scope


def test_out_of_scope_is_split_by_how_it_was_declined() -> None:
    asked = [
        choice("OTHER", 0.99, "oos"),  # declined by naming the catch-all
        choice("BANK", 0.20, "oos"),  # declined by low confidence
        choice("BANK", 0.99, "oos"),  # confidently wrong: the failure that matters
        choice("BANK", 0.99, "BANK"),  # in scope, answered
        choice("BANK", 0.20, "BANK"),  # in scope, wrongly declined
    ]
    analysis = out_of_scope(asked, oos_label="oos", abstain_label="OTHER", stakes="medium")

    assert analysis.out_of_scope == 3
    assert (analysis.caught_by_other, analysis.caught_by_unsure) == (1, 1)
    assert analysis.confidently_wrong == 1
    assert analysis.recall == pytest.approx(2 / 3)
    assert analysis.in_scope_lost == 1
    assert analysis.in_scope_cost == 0.5


def test_without_a_floor_only_the_catch_all_declines() -> None:
    asked = [choice("BANK", 0.20, "oos"), choice("OTHER", 0.99, "oos")]
    analysis = out_of_scope(asked, oos_label="oos", abstain_label="OTHER", stakes=None)
    assert analysis.caught_by_unsure == 0
    assert analysis.caught_by_other == 1
    assert analysis.confidently_wrong == 1


# --------------------------------------------------------------------------- calibration pairs


def test_pairs_line_up_with_what_is_being_calibrated() -> None:
    binary = pairs_binary([noul(0.9, "bug"), noul(0.1, "other")], "bug")
    assert binary == [(0.9, True), (0.1, False)]

    choices = pairs_choice([choice("A", 0.8, "A"), choice("A", 0.8, "B")])
    assert choices == [(0.8, True), (0.8, False)]


# --------------------------------------------------------------------------- asking


def test_answers_come_back_in_order() -> None:
    import gut

    examples = pool({"a": 30})
    backend = gut.FakeBackend(rule=lambda state, name, spec: len(str(state)) / 100)
    asked, stats = ask_all(examples, SPEC, backend, workers=4)

    assert [item.example.text for item in asked] == [e.text for e in examples]
    assert stats.requests == 30
    assert len(stats.latencies_ms) == 30


def test_percentiles_survive_an_empty_run() -> None:
    assert RunStats().percentile(0.5) == 0.0


# --------------------------------------------------------------------------- writing results


def _result(name: str, *, median_ms: float) -> Result:
    """A `Result` carrying nothing but the name and the one latency the writer reasons about."""
    from benchmarks._run import MODEL, Result

    return Result(
        benchmark=benchmarks()[name],
        model=MODEL,
        split={},
        dev_size=0,
        test_size=0,
        stats=RunStats(requests=1, latencies_ms=[median_ms]),
        postures_raw=[],
        postures_calibrated=[],
        baselines=[],
        calibration_raw=calibrate([]),
        calibration_fitted=calibrate([]),
    )


def _entry(name: str, median: float, **cost: object) -> dict[str, object]:
    return {"name": name, "cost": {"median_ms": median, "p95_ms": median * 2, **cost}}


def test_a_partial_run_leaves_the_other_benchmarks_alone(tmp_path: Path) -> None:
    """Running one benchmark used to write a file containing that one alone, deleting every entry
    it had not rerun -- and with them latency figures that cost money to measure."""
    import json

    from benchmarks._run import save

    path = tmp_path / "results.json"
    path.write_text(
        json.dumps([_entry("clinc", 306.1, timing_from="the recording run"), _entry("sms", 322.0)])
    )

    save([_result("nlbse-bug", median_ms=299.3)], path)

    written = {entry["name"]: entry for entry in json.loads(path.read_text())}
    assert set(written) == {"clinc", "sms", "nlbse-bug"}
    assert written["clinc"]["cost"]["median_ms"] == 306.1
    assert written["sms"]["cost"]["median_ms"] == 322.0


def test_an_offline_rerun_keeps_timings_only_a_live_run_could_measure(tmp_path: Path) -> None:
    """Replaying a cassette takes microseconds, so the rerun measures ~0 ms. Those zeros must not
    overwrite the recorded figures, and the file must say where the kept numbers came from."""
    import json

    from benchmarks._run import save

    path = tmp_path / "results.json"
    path.write_text(json.dumps([_entry("sms", 322.0, timing_from="a live sample of 40")]))

    save([_result("sms", median_ms=0.0)], path)

    (written,) = json.loads(path.read_text())
    assert written["cost"]["median_ms"] == 322.0
    assert written["cost"]["timing_from"] == "a live sample of 40"


def test_a_live_rerun_replaces_the_old_timings(tmp_path: Path) -> None:
    """The guard is for zeros only: a real measurement is always the better number."""
    import json

    from benchmarks._run import save

    path = tmp_path / "results.json"
    path.write_text(json.dumps([_entry("sms", 322.0, timing_from="the recording run")]))

    save([_result("sms", median_ms=280.5)], path)

    (written,) = json.loads(path.read_text())
    assert written["cost"]["median_ms"] == 280.5
    assert "timing_from" not in written["cost"]
