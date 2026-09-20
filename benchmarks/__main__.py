"""`python -m benchmarks` — run one dataset or all of them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks._datasets import benchmarks
from benchmarks._run import (
    measure_batching,
    measure_latency,
    probe,
    render,
    render_consistency,
    run,
    save,
)

RESULTS = Path(__file__).parent / "results.json"


def _recorded_median(name: str, results: Path) -> float:
    """The live median latency for one benchmark, from a previous timed run.

    Replaying a cassette measures nothing, so the batching comparison quotes a figure that was
    measured against the real model rather than one it just made up.
    """
    if not results.exists():
        return 0.0
    for entry in json.loads(results.read_text()):
        if entry["name"] == name:
            return float(entry["cost"].get("median_ms", 0.0))
    return 0.0


def main(argv: list[str] | None = None) -> int:
    """Run the benchmarks and print what they found."""
    available = benchmarks()
    parser = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__)
    parser.add_argument(
        "names",
        nargs="*",
        default=[],
        metavar="NAME",
        help=f"which to run (default: all). One of: {', '.join(available)}",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="ask the live model and re-record the cassettes; needs TYPESAFE_API_KEY",
    )
    parser.add_argument("--json", type=Path, default=RESULTS, help="where to write the results")
    parser.add_argument(
        "--latency",
        action="store_true",
        help="time a small live sample per benchmark and merge it into the results",
    )
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="ask the dev sample and stop, for judging question wording before the test set",
    )
    arguments = parser.parse_args(argv)

    unknown = [name for name in arguments.names if name not in available]
    if unknown:
        parser.error(
            f"unknown benchmark(s): {', '.join(unknown)}. Choose from {', '.join(available)}."
        )
    chosen = arguments.names or list(available)

    if arguments.latency:
        recorded = json.loads(arguments.json.read_text()) if arguments.json.exists() else []
        by_name = {entry["name"]: entry for entry in recorded}
        for name in chosen:
            stats = measure_latency(available[name])
            median, p95 = stats.percentile(0.5), stats.percentile(0.95)
            print(
                f"  {name:<12} median {median:>6.0f} ms   p95 {p95:>6.0f} ms"
                f"   ({stats.requests} live requests)"
            )
            if name not in by_name:
                print(f"    (no scored run for {name} yet, so there is nothing to merge into)")
                continue
            by_name[name]["cost"] |= {
                "median_ms": round(median, 1),
                "p95_ms": round(p95, 1),
                "timing_from": f"a live sample of {stats.requests}",
            }
        arguments.json.write_text(
            json.dumps(list(by_name.values()), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"merged into {arguments.json}")
        return 0

    if arguments.dev_only:
        for name in chosen:
            print(probe(available[name], record=arguments.record))
        return 0

    results = []
    for name in chosen:
        try:
            results.append(run(available[name], record=arguments.record))
        except Exception as error:
            print(f"\n{name}: failed — {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        print(render(results[-1]))

    if len(results) > 1:
        print(render_consistency(results))

    if {"nlbse-bug", "nlbse-kind"} <= set(chosen):
        print(measure_batching(_recorded_median("nlbse-bug", arguments.json)).render())

    save(results, arguments.json)
    total = sum(result.stats.cost_usd for result in results)
    requests = sum(result.stats.requests for result in results)
    print(f"\n{requests} requests across {len(results)} benchmarks, ~${total:.4f}")
    print(f"results written to {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
