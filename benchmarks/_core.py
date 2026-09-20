"""Shared machinery for the benchmarks: fetching, sampling, and asking.

Two decisions shape everything here.

**Each example is asked exactly once.** A risk posture changes how an answer is *acted on*, never
what was asked -- that is a property of the design, asserted by the test suite -- so the entire
posture sweep is computed afterwards from the probabilities that one call returned. Running ten
postures costs what one costs, and more importantly every posture is scored on *identical* model
answers, which is the only way the comparison between them means anything.

**Nothing is downloaded without a checksum.** Sources are pinned to a commit where the host has
commits, and verified by SHA-256 either way. A benchmark whose inputs can change underneath it is
not a benchmark.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx

from gut._backends.base import Answer, Backend
from gut._questions import QuestionSpec, canonical_json

DATA_DIR: Final = Path(__file__).parent / ".data"
CASSETTE_DIR: Final = Path(__file__).parent / "cassettes"
SEED: Final = 20260920
"""Fixed so that every sample in this repository is reproducible. See D32."""

DEV_SIZE: Final = 200
TEST_SIZE: Final = 500

WORKERS: Final = 6
"""Concurrent requests. Jev allows 1,200/min; six at ~400 ms each is roughly 15/s."""

PRICE_PER_MTOK: Final = 0.042
"""TypeSafe's published input price. Output is free."""

BUDGET_USD: Final = 5.0
"""Refuse to start a run that would plausibly cost more than this."""

CHARS_PER_TOKEN: Final = 4


@dataclass(frozen=True, slots=True)
class Source:
    """A file to download, pinned and checksummed."""

    url: str
    sha256: str
    revision: str | None = None
    """The commit the URL was pinned to, where the host has commits."""

    @property
    def filename(self) -> str:
        """Where it is cached locally."""
        stem = self.url.rsplit("/", 1)[-1]
        return hashlib.sha256(self.url.encode()).hexdigest()[:12] + "-" + stem


def fetch(source: Source) -> bytes:
    """Download `source` once, verify it, and cache it.

    Raises:
        ValueError: The bytes do not match the recorded checksum.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cached = DATA_DIR / source.filename
    if cached.exists():
        payload = cached.read_bytes()
    else:
        response = httpx.get(source.url, follow_redirects=True, timeout=120.0)
        payload = response.raise_for_status().content
        cached.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    if digest != source.sha256:
        cached.unlink(missing_ok=True)
        raise ValueError(
            f"{source.url} hashes to {digest}, not the recorded {source.sha256}. The upstream file "
            f"changed; re-pin it deliberately rather than silently benchmarking something else."
        )
    return payload


@dataclass(frozen=True, slots=True)
class Example:
    """One input and its ground-truth label."""

    text: str
    label: str
    meta: Mapping[str, str] = field(default_factory=dict)

    def truncated(self, limit: int = 4000) -> str:
        """The text, shortened. Long issue bodies cost tokens and hurt accuracy."""
        return self.text if len(self.text) <= limit else self.text[:limit]


@dataclass(frozen=True, slots=True)
class Split:
    """A dataset divided once, and never redrawn."""

    dev: tuple[Example, ...]
    test: tuple[Example, ...]

    def counts(self) -> dict[str, dict[str, int]]:
        """Label counts either side, for the record."""
        return {
            name: dict(sorted(_label_counts(part).items()))
            for name, part in (("dev", self.dev), ("test", self.test))
        }


def _label_counts(examples: Sequence[Example]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[example.label] += 1
    return dict(counts)


def stratified_split(
    examples: Sequence[Example],
    *,
    dev_size: int = DEV_SIZE,
    test_size: int = TEST_SIZE,
    seed: int = SEED,
    quota: Mapping[str, tuple[int, int]] | None = None,
) -> Split:
    """Draw a dev and a held-out test sample, keeping label proportions.

    Deterministic given the seed. Dev is drawn first and removed, so the two never overlap -- the
    whole point being that question wording is written against dev and the test set is seen once.

    `quota` fixes `(dev, test)` counts for named labels and strays deliberately from the natural
    proportions. CLINC needs it: out-of-scope is 5% of the corpus but ~18% of the canonical test
    split, and it is the thing most worth measuring here. The departure is reported, not hidden.
    """
    quota = quota or {}
    by_label: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    rng = random.Random(seed)
    for pool in by_label.values():
        rng.shuffle(pool)

    free_labels = [label for label in sorted(by_label) if label not in quota]
    free_total = sum(len(by_label[label]) for label in free_labels)
    free_dev = max(0, dev_size - sum(dev for dev, _ in quota.values()))
    free_test = max(0, test_size - sum(test for _, test in quota.values()))

    dev: list[Example] = []
    test: list[Example] = []
    for label in sorted(by_label):
        pool = by_label[label]
        if label in quota:
            want_dev, want_test = quota[label]
        else:
            share = len(pool) / free_total if free_total else 0.0
            want_dev = max(1, round(free_dev * share))
            want_test = max(1, round(free_test * share))
        dev.extend(pool[:want_dev])
        test.extend(pool[want_dev : want_dev + want_test])

    rng.shuffle(dev)
    rng.shuffle(test)
    return Split(dev=tuple(dev), test=tuple(test))


# --------------------------------------------------------------------------- asking


@dataclass(frozen=True, slots=True)
class Asked:
    """One example, and what the model said about it."""

    example: Example
    answer: Answer
    latency_ms: float


@dataclass
class RunStats:
    """What a run cost."""

    requests: int = 0
    input_tokens_estimate: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def cost_usd(self) -> float:
        """Estimated, from the published input price. Output tokens are free."""
        return self.input_tokens_estimate / 1_000_000 * PRICE_PER_MTOK

    def percentile(self, fraction: float) -> float:
        """Latency at a percentile, in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, int(fraction * len(ordered)))
        return ordered[index]


def estimate_tokens(examples: Sequence[Example], spec: QuestionSpec) -> int:
    """Roughly how many input tokens asking every example would cost.

    A character count over four, the same crude estimate `gut` uses for splitting batches. It only
    has to be good enough to refuse a run that would cost real money.
    """
    question = len(canonical_json(spec.canonical()))
    return sum((len(example.truncated()) + question) // CHARS_PER_TOKEN for example in examples)


def check_budget(
    examples: Sequence[Example], spec: QuestionSpec, *, budget: float = BUDGET_USD
) -> float:
    """Print what a run will cost, and refuse if it is more than expected.

    Raises:
        RuntimeError: The estimate exceeds the budget.
    """
    tokens = estimate_tokens(examples, spec)
    cost = tokens / 1_000_000 * PRICE_PER_MTOK
    print(f"  {len(examples)} examples, ~{tokens:,} input tokens, ~${cost:.4f}")
    if cost > budget:
        raise RuntimeError(
            f"Estimated ${cost:.2f} exceeds the ${budget:.2f} budget. Sample fewer examples, or "
            f"raise the budget deliberately."
        )
    return cost


def ask_all(
    examples: Sequence[Example],
    spec: QuestionSpec,
    backend: Backend,
    *,
    workers: int = WORKERS,
) -> tuple[list[Asked], RunStats]:
    """Ask one question about every example, concurrently, preserving order.

    One call per example: they are different subjects, so there is nothing to batch. Concurrency is
    what makes a 500-example run take a minute instead of four.
    """
    stats = RunStats()
    results: list[Asked | None] = [None] * len(examples)

    def ask_one(index: int) -> None:
        example = examples[index]
        started = time.perf_counter()
        response = backend.ask(example.truncated(), {"q": spec})
        latency = (time.perf_counter() - started) * 1000.0
        results[index] = Asked(example=example, answer=response.answers["q"], latency_ms=latency)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(ask_one, range(len(examples))))
    stats.wall_seconds = time.perf_counter() - started
    stats.requests = len(examples)
    stats.input_tokens_estimate = estimate_tokens(examples, spec)
    stats.latencies_ms = [asked.latency_ms for asked in results if asked is not None]

    return [asked for asked in results if asked is not None], stats


def progress(label: str, done: int, total: int) -> None:
    """A single rewritten line, so a long run shows something."""
    print(f"\r  {label}: {done}/{total}", end="" if done < total else "\n", flush=True)


BackendFactory = Callable[[str], Backend]
