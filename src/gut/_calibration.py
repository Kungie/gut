"""Measuring whether a probability means what it says.

The cost rule and the posture presets both take `p` at face value. If a model says `0.9` and that
happens six times in ten, every threshold derived from a cost model is in the wrong place and
nothing in the code will say so -- the decisions still look confident, they are just wrong more
often than the numbers claim.

Three measures, because they answer different questions:

- **Accuracy** — did it pick the right side? Says nothing about the probability.
- **Brier score** — mean squared error of the probability against what happened. Rewards being
  right *and* committed: a model that answers `0.5` to everything scores `0.25` however well it
  ranks. Zero is perfect.
- **Expected calibration error** — bucket the predictions, compare what was claimed against what
  happened, weight by bucket size. This is the one that tells you *where* the model is wrong, and
  a reliability table makes it readable.

A note on what is being calibrated. For a yes/no question the probability is a claim about the
event, so this is calibration in the ordinary sense. For a choice or a score, the number is a
*confidence* -- a statistic over how peaked the answer's own distribution is, which the vendor
never claimed was a probability of being correct. Measuring it against correctness is how you find
out whether treating it as one is defensible on your data. The reports label the two differently.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

MIN_EXAMPLES: Final = 30
"""Below this many examples the calibration numbers are noise and are reported as unreliable.

Thirty is a convention, not a derivation: with five buckets it is the point where a bucket can hold
enough examples for its observed frequency to mean anything at all. See D22.
"""

DEFAULT_BINS: Final = 5


@dataclass(frozen=True, slots=True)
class Bin:
    """One probability bucket, and how it turned out."""

    low: float
    high: float
    count: int
    predicted: float
    """Mean probability the model claimed in this bucket."""
    observed: float
    """Fraction of this bucket where the thing actually happened."""

    @property
    def gap(self) -> float:
        """How far the claim was from reality here."""
        return abs(self.predicted - self.observed)

    @property
    def label(self) -> str:
        """The bucket's range, for a table."""
        return f"{self.low:.1f}-{self.high:.1f}"


@dataclass(frozen=True, slots=True)
class Calibration:
    """What a set of probabilities was worth."""

    count: int
    brier: float
    ece: float
    bins: tuple[Bin, ...]

    @property
    def reliable(self) -> bool:
        """Whether there are enough examples for these numbers to mean anything."""
        return self.count >= MIN_EXAMPLES

    @property
    def caveat(self) -> str | None:
        """A warning to print alongside the numbers, when they should not be trusted."""
        if self.reliable:
            return None
        return (
            f"only {self.count} examples: below {MIN_EXAMPLES} these numbers are noise. "
            f"Read the table as a hint about where to look, not as a measurement."
        )

    def table(self, indent: str = "  ") -> str:
        """The reliability table as text."""
        return "\n".join(
            f"{indent}p {bucket.label}   n={bucket.count:<4} "
            f"said {bucket.predicted:.2f}   happened {bucket.observed:.2f}"
            for bucket in self.bins
        )

    def to_json(self) -> dict[str, Any]:
        """A JSON-compatible form, for `--json` and for CI to consume."""
        return {
            "count": self.count,
            "brier": round(self.brier, 6),
            "ece": round(self.ece, 6),
            "reliable": self.reliable,
            "bins": [
                {
                    "low": bucket.low,
                    "high": bucket.high,
                    "count": bucket.count,
                    "predicted": round(bucket.predicted, 6),
                    "observed": round(bucket.observed, 6),
                }
                for bucket in self.bins
            ],
        }


def calibrate(pairs: Sequence[tuple[float, bool]], *, bins: int = DEFAULT_BINS) -> Calibration:
    """Score a set of `(claimed probability, what happened)` pairs.

    Args:
        pairs: One entry per prediction.
        bins: How many equal-width buckets to divide `[0, 1]` into. Empty buckets are dropped.

    Returns:
        A `Calibration`. An empty input scores zero on everything, which is the only honest answer
        to "how well calibrated is nothing".
    """
    if not pairs:
        return Calibration(count=0, brier=0.0, ece=0.0, bins=())
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}.")

    brier = statistics.fmean((probability - float(event)) ** 2 for probability, event in pairs)

    buckets: list[Bin] = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        last = index == bins - 1
        inside = [
            (probability, event)
            for probability, event in pairs
            if low <= probability < high or (last and probability == 1.0)
        ]
        if not inside:
            continue
        buckets.append(
            Bin(
                low=low,
                high=high,
                count=len(inside),
                predicted=statistics.fmean(probability for probability, _ in inside),
                observed=statistics.fmean(float(event) for _, event in inside),
            )
        )

    total = len(pairs)
    ece = sum(bucket.count / total * bucket.gap for bucket in buckets)
    return Calibration(count=total, brier=brier, ece=ece, bins=tuple(buckets))
