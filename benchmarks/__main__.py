"""`python -m benchmarks` — run one dataset or all of them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks._datasets import benchmarks
from benchmarks._run import probe, render, run, save

RESULTS = Path(__file__).parent / "results.json"


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

    save(results, arguments.json)
    total = sum(result.stats.cost_usd for result in results)
    requests = sum(result.stats.requests for result in results)
    print(f"\n{requests} requests across {len(results)} benchmarks, ~${total:.4f}")
    print(f"results written to {arguments.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
