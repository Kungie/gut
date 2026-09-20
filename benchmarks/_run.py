"""Running one benchmark end to end, and rendering what it found.

The shape of every run is the same, and the order matters:

1. Split once, with a fixed seed. Dev and test never overlap.
2. Ask the **dev** set. Fit a calibrator on it.
3. Ask the **test** set, once. Score it raw and calibrated.

Nothing about the questions is allowed to change between steps 2 and 3, which is the only reason
the test numbers are worth anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gut
from benchmarks._core import (
    CASSETTE_DIR,
    Asked,
    RunStats,
    Split,
    ask_all,
    check_budget,
    stratified_split,
)
from benchmarks._datasets import Benchmark
from benchmarks._metrics import (
    HEADER,
    OutOfScope,
    Row,
    forced_choice,
    keyword_rule,
    measure,
    out_of_scope,
    pairs_binary,
    pairs_choice,
    sweep_binary,
    sweep_choice,
    threshold_binary,
)
from gut._calibration import Calibration
from gut._calibrators import Calibrator, fit
from gut._cassette import Cassette, CassetteBackend
from gut._posture import STAKES_CONFIDENCE, Stakes

MODEL = "jev-1.13.0"
"""Pinned. Every number in docs/benchmarks.md was produced by this version."""

REPORT_STAKES: Stakes = "medium"
"""The posture the headline comparisons use. The middle of three, chosen before any results."""


@dataclass(frozen=True, slots=True)
class Result:
    """One benchmark, measured."""

    benchmark: Benchmark
    model: str
    split: dict[str, dict[str, int]]
    dev_size: int
    test_size: int
    stats: RunStats
    postures_raw: list[Row]
    postures_calibrated: list[Row]
    baselines: list[Row]
    calibration_raw: Calibration
    calibration_fitted: Calibration
    out_of_scope_raw: OutOfScope | None = None
    out_of_scope_calibrated: OutOfScope | None = None

    def posture(self, label_starts: str, calibrated: bool) -> Row:
        """One row of the sweep, by the start of its label."""
        rows = self.postures_calibrated if calibrated else self.postures_raw
        return next(row for row in rows if row.label.startswith(label_starts))


def backend_for(name: str, *, record: bool) -> CassetteBackend:
    """Replay this benchmark's cassette, or record it against the live model."""
    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    live = gut.JevBackend(model=MODEL) if record else None
    # Explicit rather than reading GUT_RECORD: the flag is the benchmark's, not the environment's.
    return CassetteBackend(
        Cassette(CASSETTE_DIR / f"{name}.json"), live, record=record, model=MODEL
    )


def run(bench: Benchmark, *, record: bool = False) -> Result:
    """Split, ask, fit, score."""
    print(f"\n{bench.name}: {bench.what}")
    examples = bench.load()
    split: Split = stratified_split(examples, quota=bench.quota)
    print(f"  pool {len(examples):,}   dev {len(split.dev)}   test {len(split.test)}")

    backend = backend_for(bench.name, record=record)
    if record:
        check_budget([*split.dev, *split.test], bench.spec)

    dev, dev_stats = ask_all(split.dev, bench.spec, backend)
    test, test_stats = ask_all(split.test, bench.spec, backend)
    backend.save()

    stats = RunStats(
        requests=dev_stats.requests + test_stats.requests,
        input_tokens_estimate=dev_stats.input_tokens_estimate + test_stats.input_tokens_estimate,
        latencies_ms=[*dev_stats.latencies_ms, *test_stats.latencies_ms],
        wall_seconds=dev_stats.wall_seconds + test_stats.wall_seconds,
    )

    # Fitted on dev, never on test.
    dev_pairs = _pairs(bench, dev)
    calibrator = fit(dev_pairs, minimum=0)

    return _score(bench, test, calibrator, stats, split)


def _pairs(bench: Benchmark, asked: list[Asked]) -> list[tuple[float, bool]]:
    if bench.kind == "binary":
        assert bench.positive is not None
        return pairs_binary(asked, bench.positive)
    return pairs_choice(asked)


def _score(
    bench: Benchmark,
    test: list[Asked],
    calibrator: Calibrator,
    stats: RunStats,
    split: Split,
) -> Result:
    test_pairs = _pairs(bench, test)
    baselines: list[Row] = []

    if bench.kind == "binary":
        assert bench.positive is not None
        raw = sweep_binary(test, bench.positive)
        calibrated = sweep_binary(test, bench.positive, calibrator=calibrator)
        baselines.append(threshold_binary(test, bench.positive))
        if bench.keyword is not None:
            baselines.append(
                keyword_rule(test, bench.keyword, bench.positive, label=bench.keyword_name)
            )
        oos_raw = oos_cal = None
    else:
        raw = sweep_choice(test, abstain_label=bench.abstain_label)
        calibrated = sweep_choice(test, calibrator=calibrator, abstain_label=bench.abstain_label)
        baselines.append(forced_choice(test))
        oos_raw = oos_cal = None
        if bench.oos_label and bench.abstain_label:
            oos_raw = out_of_scope(
                test,
                oos_label=bench.oos_label,
                abstain_label=bench.abstain_label,
                stakes=REPORT_STAKES,
            )
            oos_cal = out_of_scope(
                test,
                oos_label=bench.oos_label,
                abstain_label=bench.abstain_label,
                stakes=REPORT_STAKES,
                calibrator=calibrator,
            )

    return Result(
        benchmark=bench,
        model=MODEL,
        split=split.counts(),
        dev_size=len(split.dev),
        test_size=len(split.test),
        stats=stats,
        postures_raw=raw,
        postures_calibrated=calibrated,
        baselines=baselines,
        calibration_raw=measure(test_pairs),
        calibration_fitted=measure(test_pairs, calibrator),
        out_of_scope_raw=oos_raw,
        out_of_scope_calibrated=oos_cal,
    )


def probe(bench: Benchmark, *, record: bool = False) -> str:
    """Ask the **dev** sample only, and report enough to judge the question wording.

    Deliberately separate from `run`. Wording is written against dev and frozen before the test set
    is scored; a function that showed both at once would make that discipline impossible to keep.
    """
    examples = bench.load()
    split = stratified_split(examples, quota=bench.quota)
    backend = backend_for(bench.name, record=record)
    if record:
        check_budget(split.dev, bench.spec)

    dev, stats = ask_all(split.dev, bench.spec, backend)
    backend.save()

    pairs = _pairs(bench, dev)
    calibration = measure(pairs)
    accuracy = sum(1 for _, event in pairs if event) / len(pairs)
    wrong = [item for item, (_, event) in zip(dev, pairs, strict=True) if not event]

    lines = [
        f"\n{bench.name} — dev only ({len(dev)} examples)",
        f"  agreement with the label   {accuracy:.1%}",
        f"  Brier {calibration.brier:.3f}   ECE {calibration.ece:.3f}",
        f"  question: {bench.spec.instructions}",
        "",
        "  a few it got wrong:",
    ]
    for item in wrong[:5]:
        text = item.example.text.replace("\n", " ")[:88]
        lines.append(f"    labelled {item.example.label:<10} {text}")
    lines.append(f"\n  {stats.requests} requests, ~${stats.cost_usd:.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- rendering


def render(result: Result) -> str:
    """The whole result as text."""
    bench = result.benchmark
    lines = [
        "",
        f"=== {bench.name} — {bench.what}",
        f"    model {result.model}   dev {result.dev_size}   test {result.test_size}",
        f"    {bench.citation}",
        "",
        "  the model's own numbers (this is Jev, not gut)",
        f"    Brier {result.calibration_raw.brier:.3f} -> {result.calibration_fitted.brier:.3f}"
        f"   ECE {result.calibration_raw.ece:.3f} -> {result.calibration_fitted.ece:.3f}"
        f"   (after a calibrator fitted on dev)",
        "",
        "  what gut does with them, raw",
        HEADER,
    ]
    lines.extend(row.line() for row in result.postures_raw)
    lines.append("")
    lines.append("  the same, calibrated")
    lines.append(HEADER)
    lines.extend(row.line() for row in result.postures_calibrated)
    lines.append("")
    lines.append("  what you would otherwise have written")
    lines.append(HEADER)
    lines.extend(row.line() for row in result.baselines)
    if bench.published:
        lines.append(f"    for context, trained models: {bench.published}")

    if result.out_of_scope_raw is not None:
        lines.append("")
        lines.append(f"  out of scope, at stakes={REPORT_STAKES}")
        for label, analysis in (
            ("raw", result.out_of_scope_raw),
            ("calibrated", result.out_of_scope_calibrated),
        ):
            if analysis is None:  # pragma: no cover - both are set together
                continue
            lines.append(
                f"    {label:<11} declined {analysis.caught}/{analysis.out_of_scope}"
                f" ({analysis.recall:.0%})  by OTHER {analysis.caught_by_other}"
                f"  by UNSURE {analysis.caught_by_unsure}"
                f"  confidently wrong {analysis.confidently_wrong}"
            )
            lines.append(
                f"    {'':<11} in-scope wrongly declined "
                f"{analysis.in_scope_lost}/{analysis.in_scope} ({analysis.in_scope_cost:.0%})"
            )

    stats = result.stats
    lines.extend(
        [
            "",
            f"  {stats.requests} requests   median {stats.percentile(0.5):.0f} ms"
            f"   p95 {stats.percentile(0.95):.0f} ms   ~${stats.cost_usd:.4f}",
        ]
    )
    return "\n".join(lines)


def to_json(result: Result) -> dict[str, Any]:
    """The same, for `docs/benchmarks.md` to be generated from."""
    return {
        "name": result.benchmark.name,
        "what": result.benchmark.what,
        "model": result.model,
        "licence": result.benchmark.licence,
        "citation": result.benchmark.citation,
        "published": result.benchmark.published,
        "dev_size": result.dev_size,
        "test_size": result.test_size,
        "split": result.split,
        "calibration": {
            "raw": result.calibration_raw.to_json(),
            "fitted": result.calibration_fitted.to_json(),
        },
        "postures": {
            "raw": [_row_json(row) for row in result.postures_raw],
            "calibrated": [_row_json(row) for row in result.postures_calibrated],
        },
        "baselines": [_row_json(row) for row in result.baselines],
        "out_of_scope": (
            None
            if result.out_of_scope_raw is None
            else {
                "stakes": REPORT_STAKES,
                "floor": STAKES_CONFIDENCE[REPORT_STAKES],
                "raw": _oos_json(result.out_of_scope_raw),
                "calibrated": _oos_json(result.out_of_scope_calibrated),
            }
        ),
        "cost": {
            "requests": result.stats.requests,
            "median_ms": round(result.stats.percentile(0.5), 1),
            "p95_ms": round(result.stats.percentile(0.95), 1),
            "usd": round(result.stats.cost_usd, 5),
        },
    }


def _row_json(row: Row) -> dict[str, Any]:
    return {
        "posture": row.label.strip(),
        "coverage": round(row.coverage, 4),
        "error_rate": round(row.error_rate, 4),
        "wrong": row.wrong,
        "false_yes": row.false_yes,
        "false_no": row.false_no,
        "automatic": row.automatic,
        "total": row.total,
    }


def _oos_json(analysis: OutOfScope | None) -> dict[str, Any] | None:
    if analysis is None:  # pragma: no cover - always paired with a raw analysis
        return None
    return {
        "out_of_scope": analysis.out_of_scope,
        "by_other": analysis.caught_by_other,
        "by_unsure": analysis.caught_by_unsure,
        "confidently_wrong": analysis.confidently_wrong,
        "recall": round(analysis.recall, 4),
        "in_scope": analysis.in_scope,
        "in_scope_lost": analysis.in_scope_lost,
        "in_scope_cost": round(analysis.in_scope_cost, 4),
    }


def save(results: list[Result], path: Path) -> None:
    """Write every result, so the documentation can be generated rather than transcribed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([to_json(result) for result in results], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
