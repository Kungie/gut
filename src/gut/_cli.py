"""The `gut` command line.

```bash
gut eval predicates/                          # against the configured model
gut eval predicates/ --cassette tape.json     # offline, from a recording
GUT_RECORD=1 gut eval predicates/ --cassette tape.json --model jev-1.13.0
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
from typing import Any

import yaml

from gut._backends.base import Backend
from gut._calibration import MIN_EXAMPLES
from gut._errors import GutError
from gut._evals import EvalResult, load_suite, looks_like_suite, run_suite

SUFFIXES = (".yaml", ".yml")


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
    gut.configure(backend=backend)
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


def build_parser() -> argparse.ArgumentParser:
    """The `gut` command line."""
    parser = argparse.ArgumentParser(prog="gut", description="Tools for gut decisions.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    evaluate = subcommands.add_parser(
        "eval", help="run predicate files and measure accuracy and calibration"
    )
    evaluate.add_argument("paths", nargs="*", default=["."], help="files or directories")
    evaluate.add_argument("--cassette", help="record/replay file, so this can run offline")
    evaluate.add_argument("--model", help="model to ask, or the one a cassette was recorded from")
    evaluate.add_argument(
        "--bins", type=int, default=5, help="buckets in the reliability table (default 5)"
    )
    evaluate.add_argument("--json", action="store_true", help="machine-readable output")
    evaluate.add_argument("--plot", metavar="OUT.png", help="reliability diagram, needs gut[plot]")
    evaluate.set_defaults(handler=run_eval)
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
