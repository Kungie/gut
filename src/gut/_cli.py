"""gut on the command line: judgments over lines of text, like grep over meaning.

    git log --format=%s | gut filter "adds a new feature"
    gut map tickets.txt --classify team=billing,platform,other --rate urgency="can wait,today,now"

`filter` prints the lines a claim is true of. `map` prints one JSON object per line with every
answer. Both read lines from the files named, or from standard input, and with `--read-files` each
line is a path whose file is judged instead. A summary -- how many, what it cost, how long it
took -- goes to standard error, so it never mixes into a pipe.

The model comes from the environment, as for the MCP server: `TYPESAFE_API_KEY` for Jev, or
`GUT_BACKEND` for another one (see `gut._env`). `--max-cost` stops sending requests once a budget is
spent, which needs a backend that reports cost -- Jev through OpenRouter, or a local model.

Exit status: 0 when something was printed, 1 when `filter` matched nothing, 2 on an error, and 3
when the budget stopped the run, with the line to resume from.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, TextIO

from gut._backends import Backend
from gut._backends._many import DEFAULT_CONCURRENCY
from gut._decision import BaseDecision, ChoiceDecision, Decision, ScoreDecision
from gut._each import each
from gut._env import NO_MODEL, backend_from_env
from gut._errors import BudgetExceeded, GutError
from gut._judge import judge
from gut._options import options_enum
from gut._outcomes import Outcome
from gut._usage import Usage, usage

CHUNK: Final = 64
"""Lines asked about together. Results are printed chunk by chunk, so a long input streams, and a
budget is checked between chunks."""

MAX_FILE_CHARS: Final = 200_000
"""A file longer than this is skipped rather than sent: it would not fit in one request."""

OUTCOMES: Final = ("yes", "no", "unsure")

FILES_HELP: Final = "read these instead of standard input"


@dataclass(frozen=True)
class Record:
    """One thing to judge: a line of input, or the file a line names."""

    number: int
    """Its position in the input, counting from 1."""
    shown: str
    """What is printed for it: the line itself, or the path."""
    text: str
    """What the model reads."""


@dataclass
class Question:
    """One question from the `map` command line."""

    name: str
    kind: str
    claim: str | None = None
    options: type[Enum] | None = None
    levels: list[str] = field(default_factory=list)


class UsageError(Exception):
    """A mistake on the command line, reported without a traceback."""


# --------------------------------------------------------------------------- reading the input


def lines(paths: Sequence[str], stdin: TextIO) -> Iterator[str]:
    """Every line of the named files, in order, or of standard input when none are named."""
    if not paths:
        for line in stdin:
            yield line.rstrip("\r\n")
        return
    for path in paths:
        if path == "-":
            yield from (line.rstrip("\r\n") for line in stdin)
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:  # noqa: PTH123
                yield from (line.rstrip("\r\n") for line in handle)
        except OSError as error:
            raise UsageError(f"cannot read {path}: {error.strerror or error}") from error


def records(
    paths: Sequence[str], stdin: TextIO, *, read_files: bool, skipped: list[str]
) -> Iterator[Record]:
    """The records to judge, skipping blank lines and files that cannot be read as text."""
    for number, line in enumerate(lines(paths, stdin), start=1):
        if not line.strip():
            continue
        if not read_files:
            yield Record(number, line, line)
            continue
        path = Path(line.strip())
        try:
            raw = path.read_bytes()
        except OSError as error:
            skipped.append(f"{path}: {error.strerror or error}")
            continue
        if b"\0" in raw[:8192]:
            skipped.append(f"{path}: binary")
            continue
        text = raw.decode("utf-8", errors="replace")
        if len(text) > MAX_FILE_CHARS:
            skipped.append(f"{path}: longer than {MAX_FILE_CHARS:,} characters")
            continue
        yield Record(number, str(path), f"File: {path}\n\n{text}")


def chunks(items: Iterator[Record], first: int, size: int) -> Iterator[list[Record]]:
    """`items` in lists: the first of `first`, the rest of `size`."""
    batch: list[Record] = []
    limit = first
    for item in items:
        batch.append(item)
        if len(batch) == limit:
            yield batch
            batch, limit = [], size
    if batch:
        yield batch


# --------------------------------------------------------------------------- the questions


def parse_question(kind: str, value: str) -> Question:
    """`NAME=...` from `--likely`, `--classify` or `--rate`."""
    name, sep, body = value.partition("=")
    name, body = name.strip(), body.strip()
    if not sep or not name or not body:
        raise UsageError(f"--{kind} takes NAME=..., got {value!r}")
    if kind == "likely":
        return Question(name, kind, claim=body)
    if kind == "classify":
        if body.startswith("{"):
            try:
                described = json.loads(body)
            except ValueError as error:
                raise UsageError(f"--classify {name}: not valid JSON: {error}") from error
            # Starting with "{", valid JSON can only be an object.
            return Question(name, kind, options=options_enum(described))
        return Question(name, kind, options=options_enum(_split(body)))
    return Question(name, kind, levels=_split(body))


def _split(body: str) -> list[str]:
    return [part.strip() for part in body.split(",") if part.strip()]


def compact(decision: BaseDecision) -> dict[str, Any]:
    """The part of a decision a line of output needs."""
    if isinstance(decision, Decision):
        return {"outcome": decision.outcome.value, "p": round(decision.p, 4)}
    if isinstance(decision, ChoiceDecision):
        value = decision.value
        return {
            "outcome": decision.outcome.value,
            "value": None if value is Outcome.UNSURE else value.name,
            "confidence": round(decision.confidence, 4),
        }
    assert isinstance(decision, ScoreDecision)
    record = decision.to_dict()
    return {
        "outcome": decision.outcome.value,
        "score": round(decision.score, 4),
        "level": record.get("level"),
        "confidence": round(decision.confidence, 4),
    }


# --------------------------------------------------------------------------- running


@dataclass
class Tally:
    """What the summary line reports."""

    records: int = 0
    outcomes: dict[str, int] = field(default_factory=lambda: dict.fromkeys(OUTCOMES, 0))
    printed: int = 0
    models: set[str] = field(default_factory=set)


async def run_filter(
    args: argparse.Namespace, backend: Backend | None, batch: list[Record], tally: Tally
) -> None:
    decisions = await each(
        [r.text for r in batch], backend=backend, concurrency=args.concurrency
    ).alikely(args.claim, ask_human=args.ask_human, stakes=args.stakes, lean=args.lean)
    show = set(args.show)
    for record, decision in zip(batch, decisions, strict=True):
        outcome = decision.outcome.value
        tally.outcomes[outcome] += 1
        tally.models.add(decision.model)
        if outcome in show:
            print(record.shown, flush=False)
            tally.printed += 1
    sys.stdout.flush()


async def run_map(
    args: argparse.Namespace,
    backend: Backend | None,
    batch: list[Record],
    tally: Tally,
    questions: list[Question],
) -> None:
    gate = asyncio.Semaphore(args.concurrency)

    async def one(record: Record) -> dict[str, BaseDecision]:
        async with gate:
            with judge(record.text, backend=backend) as j:
                handles: dict[str, Any] = {}
                for q in questions:
                    if q.kind == "likely":
                        assert q.claim is not None
                        handles[q.name] = j.likely(
                            q.claim, ask_human=args.ask_human, stakes=args.stakes, lean=args.lean
                        )
                    elif q.kind == "classify":
                        assert q.options is not None
                        handles[q.name] = j.classify(
                            q.options, ask_human=args.ask_human, stakes=args.stakes
                        )
                    else:
                        handles[q.name] = j.rate(
                            q.levels, ask_human=args.ask_human, stakes=args.stakes
                        )
            await j.aresolve()
            return {name: handle.decision for name, handle in handles.items()}

    answered = await asyncio.gather(*(one(record) for record in batch))
    key = "path" if args.read_files else "text"
    for record, decisions in zip(batch, answered, strict=True):
        line: dict[str, Any] = {"line": record.number, key: record.shown}
        for name, decision in decisions.items():
            line[name] = compact(decision)
            tally.models.add(decision.model)
        print(json.dumps(line, ensure_ascii=False), flush=False)
        tally.printed += 1
    sys.stdout.flush()


async def run(args: argparse.Namespace, stdin: TextIO) -> int:
    questions = _questions(args)
    backend = _backend(args)
    skipped: list[str] = []
    tally = Tally()
    started = time.perf_counter()
    # With a budget, ask one record first: a backend that does not report cost is found out
    # before it has been sent a whole chunk.
    first = 1 if args.max_cost is not None else CHUNK
    inputs = records(args.files, stdin, read_files=args.read_files, skipped=skipped)
    status = 0
    with usage(max_cost=args.max_cost) as spent:
        for batch in chunks(inputs, first, CHUNK):
            try:
                if args.command == "filter":
                    await run_filter(args, backend, batch, tally)
                else:
                    await run_map(args, backend, batch, tally, questions)
            except BudgetExceeded as error:
                _stopped(error, batch[0], args)
                status = 3
                break
            tally.records += len(batch)
    for note in skipped:
        print(f"gut: skipped {note}", file=sys.stderr)
    if not args.quiet:
        print(
            _summary(args, tally, spent, len(skipped), time.perf_counter() - started),
            file=sys.stderr,
        )
    if status:
        return status
    return 1 if args.command == "filter" and not tally.printed else 0


def _questions(args: argparse.Namespace) -> list[Question]:
    if args.command == "filter":
        return []
    questions = [parse_question(kind, value) for kind, value in args.questions or ()]
    if not questions:
        raise UsageError("map needs at least one of --likely, --classify or --rate")
    names = [q.name for q in questions]
    if len(set(names)) < len(names):
        raise UsageError("every question needs its own NAME")
    if args.lean and not any(q.kind == "likely" for q in questions):
        raise UsageError("--lean only applies to --likely questions")
    return questions


def _backend(args: argparse.Namespace) -> Backend | None:
    env = dict(os.environ)
    if args.backend:
        env["GUT_BACKEND"] = args.backend
    if args.model:
        env["GUT_MODEL"] = args.model
    return backend_from_env(env)


def _stopped(error: BudgetExceeded, record: Record, args: argparse.Namespace) -> None:
    print(f"gut: {error}", file=sys.stderr)
    if isinstance(error.usage, Usage) and error.usage.cost_known:
        where = f"line {record.number}"
        hint = (
            f"tail -n +{record.number} FILE | gut ..."
            if len(args.files) <= 1
            else "the same command on what is left"
        )
        print(f"gut: resume from {where}: {hint}", file=sys.stderr)


def _summary(
    args: argparse.Namespace, tally: Tally, spent: Usage, skipped: int, seconds: float
) -> str:
    parts = [f"{tally.records:,} record{'' if tally.records == 1 else 's'}"]
    if args.command == "filter":
        parts.append(
            ", ".join(
                f"{tally.outcomes[o]:,} {o}" for o in OUTCOMES if o != "unsure" or args.ask_human
            )
        )
    if skipped:
        parts.append(f"{skipped:,} skipped")
    parts.append(str(spent))
    parts.append(f"{seconds:.1f}s")
    if tally.models:
        parts.append(", ".join(sorted(tally.models)))
    return "gut: " + " · ".join(parts)


# --------------------------------------------------------------------------- the command line


def parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    from gut import __version__

    top = argparse.ArgumentParser(
        prog="gut",
        description="Judgment calls over lines of text, answered by a small, fast, cheap model.",
        epilog="The model: TYPESAFE_API_KEY for Jev, or GUT_BACKEND (jev, openrouter, ollaya, "
        "openai, openai-decisions, ollama, zeroshot, transformers) with GUT_MODEL. "
        "Docs: https://gutpy.dev/docs/cli.html",
    )
    top.add_argument("--version", action="version", version=f"gut {__version__}")
    commands = top.add_subparsers(dest="command", required=True, metavar="COMMAND")

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--read-files", action="store_true", help="each line is a path: judge the file it names"
    )
    shared.add_argument("--ask-human", action="store_true", help="allow the answer 'unsure'")
    shared.add_argument(
        "--stakes",
        choices=("low", "medium", "high"),
        help="how sure an answer must be (needs --ask-human)",
    )
    shared.add_argument("--lean", choices=("yes", "no"), help="which way to err on a yes/no claim")
    shared.add_argument(
        "--max-cost",
        type=_dollars,
        metavar="USD",
        help="stop sending requests once this much is spent",
    )
    shared.add_argument(
        "--concurrency",
        type=_positive,
        default=DEFAULT_CONCURRENCY,
        metavar="N",
        help=f"requests in flight at once (default {DEFAULT_CONCURRENCY})",
    )
    shared.add_argument("--backend", help="overrides GUT_BACKEND")
    shared.add_argument("--model", help="overrides GUT_MODEL")
    shared.add_argument("-q", "--quiet", action="store_true", help="no summary on standard error")

    flt = commands.add_parser(
        "filter",
        parents=[shared],
        help="print the lines a claim is true of",
        description="Print the lines a claim is true of, like grep for meaning.",
    )
    flt.add_argument("claim", help='a short claim about each line, e.g. "is spam"')
    flt.add_argument("files", nargs="*", metavar="FILE", help=FILES_HELP)
    flt.add_argument(
        "--show",
        type=_outcomes,
        default=["yes"],
        metavar="OUTCOMES",
        help="which answers to print, comma-separated: yes, no, unsure (default yes)",
    )

    mapped = commands.add_parser(
        "map",
        parents=[shared],
        help="print every answer about every line, as JSON",
        description="Print one JSON object per line, with an answer for every question.",
    )
    mapped.add_argument("files", nargs="*", metavar="FILE", help=FILES_HELP)
    mapped.add_argument(
        "--likely",
        dest="questions",
        action="append",
        type=lambda v: ("likely", v),
        metavar="NAME=CLAIM",
        help="a yes/no claim",
    )
    mapped.add_argument(
        "--classify",
        dest="questions",
        action="append",
        type=lambda v: ("classify", v),
        metavar="NAME=A,B,C",
        help='options, or JSON {"label": "description"}',
    )
    mapped.add_argument(
        "--rate",
        dest="questions",
        action="append",
        type=lambda v: ("rate", v),
        metavar="NAME=L0,L1,...",
        help="an ordered scale, 2 to 10 levels",
    )
    return top, {"filter": flt, "map": mapped}


def _dollars(value: str) -> float:
    try:
        amount = float(value.lstrip("$"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an amount: {value!r}") from None
    if amount < 0:
        raise argparse.ArgumentTypeError("must be zero or more")
    return amount


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _outcomes(value: str) -> list[str]:
    chosen = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [part for part in chosen if part not in OUTCOMES]
    if unknown or not chosen:
        raise argparse.ArgumentTypeError("choose from yes, no, unsure")
    return chosen


def parse(argv: list[str]) -> argparse.Namespace:
    """The command line, with options allowed anywhere: `gut filter "is spam" app.log --ask-human`.

    argparse cannot mix options and positionals after a subcommand, so the subcommand's own parser
    reads its part intermixed.
    """
    top, commands = parser()
    if not argv or argv[0] not in commands:
        return top.parse_args(argv)  # --help, --version, or a mistake argparse explains
    args = commands[argv[0]].parse_intermixed_args(argv[1:])
    args.command = argv[0]
    return args


def main(argv: Sequence[str] | None = None, stdin: TextIO | None = None) -> int:
    """The `gut` command. Returns the exit status."""
    args = parse(list(sys.argv[1:] if argv is None else argv))
    if args.stakes and not args.ask_human:
        print("gut: --stakes only matters with --ask-human, which allows 'unsure'", file=sys.stderr)
        return 2
    if args.command == "filter" and "unsure" in args.show and not args.ask_human:
        print("gut: --show unsure needs --ask-human; otherwise nothing is unsure", file=sys.stderr)
        return 2
    try:
        return asyncio.run(run(args, sys.stdin if stdin is None else stdin))
    except UsageError as error:
        print(f"gut: {error}", file=sys.stderr)
        return 2
    except GutError as error:
        text = NO_MODEL if "No backend is configured" in str(error) else str(error)
        print(f"gut: {text}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130
    except BrokenPipeError:  # pragma: no cover - `gut ... | head`
        sys.stderr.close()
        return 0


def entry() -> None:  # pragma: no cover - the console script
    """The console script behind `gut` and `gutfeel`."""
    sys.exit(main())


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    entry()
