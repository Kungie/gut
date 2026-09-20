"""Run the triage handler over the whole dataset and report what it cost.

```bash
python report.py                 # stand-in model, no key needed
TYPESAFE_API_KEY=... python report.py    # the real thing
```

The headline numbers are the ones a team would actually argue about: how much of the queue was
handled without a person, how often that was wrong, what the mistakes cost, and how many model
calls it took.

The last block re-scores the *same answers* under `before.py`'s rule -- a hard 0.7 threshold with
no third branch. Same model, same probabilities, different decision: the difference between the
two cost figures is what the cost rule is worth.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any

from _simulate import configure_backend
from after import (
    COST_FALSE_ESCALATION,
    COST_HUMAN_REVIEW,
    COST_MISSED_THREAT,
    load_tickets,
    state_of,
    triage,
)

import gut

BEFORE_THRESHOLD = 0.7
"""The number `before.py` hard-codes."""

RELIABILITY_BINS = 5
"""How many probability buckets the calibration table uses."""


@dataclass
class Outcome:
    """One ticket, judged."""

    ticket_id: str
    hard: bool
    truth: bool
    churn_p: float
    escalated: bool
    to_human: bool
    team: str
    team_truth: str
    urgency: float
    urgency_truth: int

    @property
    def automatic(self) -> bool:
        """Whether it was resolved without a person."""
        return not self.to_human

    @property
    def wrong(self) -> bool:
        """Whether an automatic decision disagreed with the answer key."""
        return self.automatic and self.escalated != self.truth

    @property
    def cost(self) -> float:
        """What this decision cost, on the scale `after.py` declares."""
        if self.to_human:
            return COST_HUMAN_REVIEW
        if self.escalated and not self.truth:
            return COST_FALSE_ESCALATION
        if not self.escalated and self.truth:
            return COST_MISSED_THREAT
        return 0.0

    @property
    def before_cost(self) -> float:
        """What the same answer would have cost under a hard 0.7 threshold and no human."""
        escalated = self.churn_p > BEFORE_THRESHOLD
        if escalated and not self.truth:
            return COST_FALSE_ESCALATION
        if not escalated and self.truth:
            return COST_MISSED_THREAT
        return 0.0

    @property
    def before_wrong(self) -> bool:
        """Whether that rule would have got it wrong."""
        return (self.churn_p > BEFORE_THRESHOLD) != self.truth


def run(limit: int | None) -> tuple[list[Outcome], Any, str, float]:
    """Triage every ticket, collecting what each decision was and what it cost."""
    tickets = load_tickets()[:limit]
    backend, backend_label = configure_backend(load_tickets())
    sink = gut.MemorySink()
    gut.configure(sink=sink)

    outcomes: list[Outcome] = []
    started = time.perf_counter()
    for ticket in tickets:
        seen = len(sink.decisions)
        action = triage(state_of(ticket))
        churn_p = _churn_probability(sink.decisions[seen:])
        outcomes.append(
            Outcome(
                ticket_id=ticket["id"],
                hard="hard" in ticket,
                truth=bool(ticket["labels"]["cancel_threat"]),
                churn_p=churn_p,
                escalated=action.escalate,
                to_human=action.to_human,
                team=action.team,
                team_truth=str(ticket["labels"]["team"]),
                urgency=action.urgency,
                urgency_truth=int(ticket["labels"]["urgency"]),
            )
        )
    elapsed = time.perf_counter() - started
    return outcomes, backend, backend_label, elapsed


def _churn_probability(records: list[gut.DecisionRecord]) -> float:
    """Pull the churn judgment out of this ticket's decision records."""
    for record in records:
        instructions = str(record.question.get("instructions", ""))
        if "at risk of leaving" in instructions and record.p is not None:
            return record.p
    raise AssertionError("triage() always asks the churn question first")  # pragma: no cover


def _line(label: str, value: object, note: str = "") -> str:
    return f"  {label:<28} {value!s:>10}  {note}"


def _ratio(numerator: int, denominator: int) -> str:
    """A percentage, or a dash when there is nothing to divide by."""
    return f"{numerator / denominator:.0%}" if denominator else "-"


def brier_score(outcomes: list[Outcome]) -> float:
    """Mean squared error of the probabilities against what actually happened.

    Zero is perfect. It rewards being both right *and* confident, so a model that hedges at 0.5 on
    everything scores 0.25 however well it ranks.
    """
    return statistics.fmean((o.churn_p - float(o.truth)) ** 2 for o in outcomes)


def reliability(outcomes: list[Outcome]) -> list[tuple[str, int, float, float]]:
    """Predicted probability against observed frequency, bucketed.

    This is the question the cost rule quietly depends on: when the model says 0.9, does it happen
    nine times out of ten? Only non-empty buckets come back.
    """
    table: list[tuple[str, int, float, float]] = []
    for index in range(RELIABILITY_BINS):
        low = index / RELIABILITY_BINS
        high = (index + 1) / RELIABILITY_BINS
        last = index == RELIABILITY_BINS - 1
        bucket = [o for o in outcomes if low <= o.churn_p < high or (last and o.churn_p == 1.0)]
        if not bucket:
            continue
        table.append(
            (
                f"{low:.1f}-{high:.1f}",
                len(bucket),
                statistics.fmean(o.churn_p for o in bucket),
                statistics.fmean(float(o.truth) for o in bucket),
            )
        )
    return table


def calibration_error(outcomes: list[Outcome]) -> float:
    """Expected calibration error: the size-weighted gap between predicted and observed."""
    total = len(outcomes)
    return sum(
        count / total * abs(predicted - observed)
        for _, count, predicted, observed in reliability(outcomes)
    )


def main() -> None:
    """Print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tickets")
    limit = parser.parse_args().limit

    outcomes, backend, backend_label, elapsed = run(limit)
    total = len(outcomes)
    calls = backend.call_count
    questions = backend.question_count

    automatic = [o for o in outcomes if o.automatic]
    to_human = [o for o in outcomes if o.to_human]
    wrong = [o for o in automatic if o.wrong]
    missed = [o for o in wrong if o.truth]
    over = [o for o in wrong if not o.truth]

    print(f"\ngut demo - support ticket triage   ({total} tickets, {backend_label} model)")
    print("=" * 74)

    print("\nqueue")
    print(_line("resolved automatically", f"{len(automatic)}", f"{len(automatic) / total:.0%}"))
    print(_line("sent to a human", f"{len(to_human)}", f"{len(to_human) / total:.0%}"))

    print("\nautomatic decisions")
    print(_line("wrong", len(wrong), f"{len(wrong) / max(len(automatic), 1):.0%} of automatic"))
    print(_line("  churn risk missed", len(missed), f"at {COST_MISSED_THREAT:g} each"))
    print(_line("  escalated needlessly", len(over), f"at {COST_FALSE_ESCALATION:g} each"))

    print("\ncost")
    print(_line("total", f"{sum(o.cost for o in outcomes):.0f}", "on after.py's scale"))
    print(_line("per ticket", f"{sum(o.cost for o in outcomes) / total:.2f}"))

    print("\nmodel")
    print(_line("requests", calls, f"{calls / total:.1f} per ticket"))
    print(_line("judgments", questions, f"{questions / max(calls, 1):.1f} per request"))
    print(_line("wall clock", f"{elapsed:.2f}s", f"{elapsed / total * 1000:.0f} ms per ticket"))

    true_positives = sum(1 for o in automatic if o.escalated and o.truth)
    false_positives = sum(1 for o in automatic if o.escalated and not o.truth)
    false_negatives = sum(1 for o in automatic if not o.escalated and o.truth)
    threats = sum(1 for o in outcomes if o.truth)
    reached_someone = sum(1 for o in outcomes if o.truth and (o.escalated or o.to_human))

    print("\nchurn detection")
    print(
        _line(
            "precision",
            _ratio(true_positives, true_positives + false_positives),
            "of automatic escalations",
        )
    )
    print(
        _line(
            "recall",
            _ratio(true_positives, true_positives + false_negatives),
            "of threats it decided itself",
        )
    )
    print(_line("reached a person at all", f"{reached_someone}/{threats}", "escalated or reviewed"))

    print("\ncalibration")
    print(_line("Brier score", f"{brier_score(outcomes):.3f}", "0 is perfect, 0.25 is a coin flip"))
    print(
        _line(
            "calibration error",
            f"{calibration_error(outcomes):.3f}",
            "gap between claimed and observed",
        )
    )
    for label, count, predicted, observed in reliability(outcomes):
        print(f"    p {label}   n={count:<4} said {predicted:.2f}   happened {observed:.2f}")

    print("\nside judgments")
    team_right = sum(1 for o in outcomes if o.team == o.team_truth)
    print(_line("team routed correctly", f"{team_right}/{total}", f"{team_right / total:.0%}"))
    drift = statistics.fmean(abs(o.urgency - o.urgency_truth) for o in outcomes)
    print(_line("urgency mean error", f"{drift:.2f}", "levels"))

    hard = [o for o in outcomes if o.hard]
    if hard:
        hard_wrong = sum(1 for o in hard if o.wrong)
        hard_human = sum(1 for o in hard if o.to_human)
        print("\nthe ambiguous ones")
        print(_line("marked hard", len(hard), f"{len(hard) / total:.0%} of the dataset"))
        print(_line("  sent to a human", hard_human, f"{hard_human / len(hard):.0%} of them"))
        print(_line("  wrong automatically", hard_wrong))

    print("\nsame answers, before.py's rule")
    print(f"  a hard threshold of {BEFORE_THRESHOLD}, and no third branch")
    before_wrong = [o for o in outcomes if o.before_wrong]
    before_missed = sum(1 for o in before_wrong if o.truth)
    print(_line("wrong", len(before_wrong), f"vs {len(wrong)} with the cost rule"))
    print(_line("  churn risk missed", before_missed, f"vs {len(missed)}"))
    print(
        _line(
            "total cost",
            f"{sum(o.before_cost for o in outcomes):.0f}",
            f"vs {sum(o.cost for o in outcomes):.0f}",
        )
    )
    print(_line("requests", questions, f"vs {calls}, one per judgment"))
    print()


if __name__ == "__main__":
    main()
