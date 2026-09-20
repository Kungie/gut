"""The `gut` command line.

```bash
gut eval predicates/                          # against the configured model
gut eval predicates/ --cassette tape.json     # offline, from a recording
GUT_RECORD=1 gut eval predicates/ --cassette tape.json --model jev-1.13.0

gut calibrate predicates/ --cassette tape.json --out calibration.json
```

`gut eval` runs predicate files and reports not just whether the answers were right but whether the
numbers attached to them were worth anything. Accuracy tells you the model picked the right side.
Brier score and expected calibration error tell you whether `0.9` means what the cost rule assumes
it means, and the reliability table tells you where it doesn't.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

import yaml

from gut._backends.base import Backend
from gut._calibration import MIN_EXAMPLES, Calibration, calibrate
from gut._calibrators import MIN_FIT_EXAMPLES, CalibrationSet, Entry, Method, fit
from gut._errors import GutError
from gut._evals import EvalResult, load_suite, looks_like_suite, run_suite

SUFFIXES = (".yaml", ".yml")

COMMANDS: Final[tuple[str, ...]] = ("eval", "calibrate")
"""Every subcommand `gut` offers, so documentation can be checked against it."""

OVERFIT_GAP = 0.02
"""How much better in-sample has to look than out-of-fold before the gap is called out."""


def discover(paths: Sequence[str]) -> list[Path]:
    """Every predicate file under the given paths, in a stable order.

    A directory is walked; a file is taken as given. YAML that is not a predicate file is skipped
    silently, because pointing this at a repository should not complain about its CI config.
    """
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        candidates: Iterable[Path] = (
            sorted(p for suffix in SUFFIXES for p in path.rglob(f"*{suffix}"))
            if path.is_dir()
            else [path]
        )
        for candidate in candidates:
            if candidate.suffix not in SUFFIXES:
                continue
            try:
                document = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if looks_like_suite(document):
                found.append(candidate)
    return found


def build_backend(cassette: str | None, model: str | None) -> Backend:
    """The backend to evaluate against.

    With a cassette this replays, or records when `GUT_RECORD=1` -- which is what makes `gut eval`
    usable in CI without a key and without pretending the answers are real.
    """
    from gut._cassette import CassetteBackend, record_requested

    live: Backend | None = None
    if os.environ.get("TYPESAFE_API_KEY", "").strip():
        from gut._backends.jev import JevBackend

        live = JevBackend(model=model)

    if cassette is not None:
        if record_requested() and live is None:
            raise GutError(
                "GUT_RECORD=1 asks for a recording, which needs a real model: set "
                "TYPESAFE_API_KEY, or drop GUT_RECORD to replay what is already there."
            )
        return CassetteBackend(cassette, live, model=model or (live.model_id if live else None))

    if live is None:
        raise GutError(
            "Nothing to ask. Set TYPESAFE_API_KEY for the real model, or pass --cassette to "
            "replay a recording."
        )
    return live


def _render(result: EvalResult, *, bins: int) -> str:
    """One predicate file's results as text."""
    suite = result.suite
    calibration = result.calibration_with(bins)
    verdict = "PASS" if result.passed else "FAIL"
    lines = [
        f"{verdict}  {suite.name}  ({suite.kind})",
        f"        question   {suite.spec.instructions or suite.spec.canonical()}",
        f"        accuracy   {result.accuracy:.0%}  ({result.correct}/{len(result.results)})"
        f"   min {suite.min_accuracy:.0%}",
        f"        Brier      {calibration.brier:.3f}   0 is perfect, 0.25 is a coin flip",
        f"        cal. error {calibration.ece:.3f}"
        + (f"   max {suite.max_ece:.3f}" if suite.max_ece is not None else ""),
        f"        calibrating the {result.calibration_kind} the model attached to each answer",
    ]
    if result.ece_exceeded:
        assert suite.max_ece is not None
        lines.append(
            f"        ! calibration error {calibration.ece:.3f} is above this file's "
            f"max_ece of {suite.max_ece:.3f}"
        )
    if calibration.caveat:
        lines.append(f"        ! {calibration.caveat}")
    lines.append(calibration.table(indent="        "))
    for failure in result.failures:
        lines.append(f"        wrong: expected {failure.example.expected!r}, got {failure.actual}")
        lines.append(f"               {failure.example.label}")
    return "\n".join(lines)


def _plot(results: list[EvalResult], destination: str, bins: int) -> None:
    """Write a reliability diagram."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - only without the extra installed
        raise GutError('A reliability diagram needs matplotlib: pip install "gut[plot]"') from error

    figure, axes = plt.subplots(figsize=(5, 5))
    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey", label="perfect")
    for result in results:
        calibration = result.calibration_with(bins)
        if not calibration.bins:  # pragma: no cover - a suite always has at least one example
            continue
        axes.plot(
            [b.predicted for b in calibration.bins],
            [b.observed for b in calibration.bins],
            marker="o",
            label=f"{result.suite.name} (n={calibration.count})",
        )
    axes.set_xlabel("claimed")
    axes.set_ylabel("observed")
    axes.set_title("Reliability")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _to_json(results: list[EvalResult], bins: int) -> dict[str, Any]:
    return {
        "min_examples_for_calibration": MIN_EXAMPLES,
        "predicates": [
            {
                "name": result.suite.name,
                "path": str(result.suite.path),
                "kind": result.suite.kind,
                "passed": result.passed,
                "examples": len(result.results),
                "correct": result.correct,
                "accuracy": round(result.accuracy, 6),
                "min_accuracy": result.suite.min_accuracy,
                "max_ece": result.suite.max_ece,
                "calibrating": result.calibration_kind,
                "calibration": result.calibration_with(bins).to_json(),
            }
            for result in results
        ],
    }


def run_eval(arguments: argparse.Namespace) -> int:
    """Run `gut eval`, returning the process exit code."""
    import gut

    files = discover(arguments.paths)
    if not files:
        print(f"No predicate files found under {', '.join(arguments.paths)}.", file=sys.stderr)
        return 1

    backend = build_backend(arguments.cassette, arguments.model)
    calibration = CalibrationSet.load(arguments.calibration) if arguments.calibration else None
    gut.configure(backend=backend, calibration=calibration)
    results = [run_suite(load_suite(path), backend) for path in files]

    if arguments.cassette is not None:
        from gut._cassette import CassetteBackend

        assert isinstance(backend, CassetteBackend)
        backend.save()

    if arguments.plot:
        _plot(results, arguments.plot, arguments.bins)

    if arguments.json:
        print(json.dumps(_to_json(results, arguments.bins), indent=2))
    else:
        if calibration is not None:
            print(
                f"measuring the pipeline with {len(calibration)} correction(s) from "
                f"{arguments.calibration} applied.\n"
                "If these were fitted on these same examples, the numbers below are in-sample.\n"
            )
        for result in results:
            print(_render(result, bins=arguments.bins))
            print()
        failed = [r.suite.name for r in results if not r.passed]
        print(f"{len(results) - len(failed)}/{len(results)} predicates passed")
        if failed:
            print(f"failed: {', '.join(failed)}")
        if arguments.plot:
            print(f"reliability diagram written to {arguments.plot}")

    return 1 if any(not result.passed for result in results) else 0


def _out_of_fold(pairs: list[tuple[float, bool]], method: Method, folds: int) -> Calibration | None:
    """Corrected probabilities from calibrators that never saw the point they correct.

    Fitting a calibrator and then scoring it on the same examples flatters it -- isotonic in
    particular can fit noise exactly. So the improvement is reported from k-fold: each point is
    corrected by a calibrator fitted on the other folds. The calibrator that actually ships is
    then fitted on everything, which is the usual arrangement and the reason both numbers are
    printed.

    `None` when there is not enough data to split.
    """
    if folds < 2 or len(pairs) < folds * 2:
        return None
    corrected: list[tuple[float, bool]] = []
    for index in range(folds):
        # Stride rather than shuffle: the split is reproducible without seeding anything.
        holdout = pairs[index::folds]
        training = [pair for position, pair in enumerate(pairs) if position % folds != index]
        if not training or not holdout:  # pragma: no cover - guarded by the length check
            return None
        curve = fit(training, method=method, minimum=0)
        corrected.extend((curve.apply(probability), event) for probability, event in holdout)
    return calibrate(corrected)


def _fit_one(
    result: EvalResult, method: Method, folds: int, model: str | None
) -> tuple[Entry, Calibration, Calibration, Calibration | None]:
    """Fit one predicate, and measure what it bought."""
    pairs = [(item.probability, item.event) for item in result.results]
    before = calibrate(pairs)
    curve = fit(pairs, method=method)
    in_sample = calibrate([(curve.apply(p), event) for p, event in pairs])
    honest = _out_of_fold(pairs, method, folds)

    entry = Entry(
        fingerprint=result.suite.spec.fingerprint,
        calibrator=curve,
        question=result.suite.spec.canonical(),
        model=model,
        examples=len(pairs),
        before={"brier": round(before.brier, 6), "ece": round(before.ece, 6)},
        after=(
            None
            if honest is None
            else {"brier": round(honest.brier, 6), "ece": round(honest.ece, 6)}
        ),
    )
    return entry, before, in_sample, honest


def run_calibrate(arguments: argparse.Namespace) -> int:
    """Run `gut calibrate`, returning the process exit code."""
    import gut

    files = discover(arguments.paths)
    if not files:
        print(f"No predicate files found under {', '.join(arguments.paths)}.", file=sys.stderr)
        return 1

    backend = build_backend(arguments.cassette, arguments.model)
    gut.configure(backend=backend, calibration=CalibrationSet())
    results = [run_suite(load_suite(path), backend) for path in files]
    if arguments.cassette is not None:
        from gut._cassette import CassetteBackend

        assert isinstance(backend, CassetteBackend)
        backend.save()

    produced = CalibrationSet()
    skipped: list[str] = []
    worsened: list[str] = []
    print(
        f"\nfitting {arguments.method} corrections   "
        f"(out-of-fold over {arguments.folds} folds, so the improvement is not self-graded)\n"
    )
    for result in results:
        name = result.suite.name
        if len(result.results) < MIN_FIT_EXAMPLES:
            skipped.append(name)
            print(f"  {name:<18} skipped: {len(result.results)} examples, needs {MIN_FIT_EXAMPLES}")
            continue
        entry, before, in_sample, honest = _fit_one(
            result, arguments.method, arguments.folds, arguments.model
        )
        measured = honest or in_sample
        label = "out-of-fold" if honest else "in-sample only"
        print(
            f"  {name:<18} brier {before.brier:.3f} -> {measured.brier:.3f}   "
            f"ece {before.ece:.3f} -> {measured.ece:.3f}   ({label})"
        )
        print(f"  {'':<18} {entry.calibrator}")
        if honest is not None and in_sample.ece < honest.ece - OVERFIT_GAP:
            print(
                f"  {'':<18} in-sample ece would have read {in_sample.ece:.3f}; "
                f"that gap is the overfit"
            )

        # A correction that loses out of fold is worse than none. Fitting noise onto an
        # already-calibrated question is the usual cause, and the in-sample number hides it.
        if honest is not None and honest.ece > before.ece and not arguments.keep_all:
            worsened.append(name)
            print(
                f"  {'':<18} DROPPED: this makes calibration worse out of fold "
                f"({before.ece:.3f} -> {honest.ece:.3f}). Pass --keep-all to ship it anyway."
            )
            continue
        produced.add(entry)

    if not len(produced):
        print("\nNothing worth shipping was fitted.")
        return 1

    produced.save(arguments.out)
    print(f"\n{len(produced)} correction(s) written to {arguments.out}")
    if skipped:
        print(f"skipped for want of data: {', '.join(skipped)}")
    if worsened:
        print(f"dropped for making things worse: {', '.join(worsened)}")
    print("\nLoad it with:  gut.configure(calibration=CalibrationSet.load(path))")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The `gut` command line."""
    parser = argparse.ArgumentParser(prog="gut", description="Tools for gut decisions.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    evaluate = subcommands.add_parser(
        COMMANDS[0], help="run predicate files and measure accuracy and calibration"
    )
    evaluate.add_argument("paths", nargs="*", default=["."], help="files or directories")
    evaluate.add_argument("--cassette", help="record/replay file, so this can run offline")
    evaluate.add_argument("--model", help="model to ask, or the one a cassette was recorded from")
    evaluate.add_argument(
        "--bins", type=int, default=5, help="buckets in the reliability table (default 5)"
    )
    evaluate.add_argument("--json", action="store_true", help="machine-readable output")
    evaluate.add_argument("--plot", metavar="OUT.png", help="reliability diagram, needs gut[plot]")
    evaluate.add_argument(
        "--calibration",
        help="apply corrections from this file, measuring the pipeline rather than the model",
    )
    evaluate.set_defaults(handler=run_eval)

    calibrate_command = subcommands.add_parser(
        COMMANDS[1], help="fit corrections to a model's probabilities from predicate files"
    )
    calibrate_command.add_argument("paths", nargs="*", default=["."], help="files or directories")
    calibrate_command.add_argument("--cassette", help="record/replay file, so this can run offline")
    calibrate_command.add_argument("--model", help="the model these corrections are fitted for")
    calibrate_command.add_argument(
        "--method",
        choices=("isotonic", "platt"),
        default="isotonic",
        help="isotonic assumes only monotonicity; platt needs less data (default: isotonic)",
    )
    calibrate_command.add_argument(
        "--folds", type=int, default=5, help="folds for the honest estimate (default 5)"
    )
    calibrate_command.add_argument(
        "--out", default="calibration.json", help="where to write the corrections"
    )
    calibrate_command.add_argument(
        "--keep-all",
        action="store_true",
        help="ship corrections even where they make calibration worse out of fold",
    )
    calibrate_command.set_defaults(handler=run_calibrate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns an exit code rather than raising at the user."""
    arguments = build_parser().parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except GutError as error:
        print(f"gut: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
