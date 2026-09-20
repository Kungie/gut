"""Run the triage handler over the whole dataset at every posture, and report what each costs.

```bash
python report.py                                  # offline, from the committed cassette
TYPESAFE_API_KEY=... GUT_RECORD=1 python report.py   # against the real model, re-recording
```

The sweep is the point. A single hand-picked setting tells you what one configuration does; the
table tells you what the dial does, which is the thing you actually have to choose.

Every row asks the same five questions about the same 101 tickets, so the whole sweep costs one
pass over the data: the posture changes how an answer is *acted on*, never what was asked.
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from after import Team, load_tickets, state_of, triage

import gut

CASSETTE = Path(__file__).parent / "cassettes" / "predicates.json"
MODEL = "jev-1.13.0"
BEFORE_THRESHOLD = 0.7
"""The number `before.py` hard-codes."""

# What a mistake costs, for putting a single number on a row. A missed churn signal is ten times a
# needless escalation; a person reading a ticket is cheaper than either.
COST_FALSE_ESCALATION = 2.0
COST_MISSED_THREAT = 20.0
COST_HUMAN_REVIEW = 1.0

POSTURES: list[tuple[str, dict[str, Any]]] = [
    ("no arguments", {}),
    *[
        (
            f"stakes={stakes:<6} lean={lean or 'none':<4}",
            {
                "stakes": stakes,
                "lean": lean,
                "ask_human": True,
            },
        )
        for stakes in ("low", "medium", "high")
        for lean in (None, "yes", "no")
    ],
]


def configure_backend() -> Any:
    """Replay the committed cassette, or re-record it against the real model."""
    from gut._cassette import CassetteBackend, record_requested

    live = None
    if record_requested():
        if not os.environ.get("TYPESAFE_API_KEY", "").strip():
            raise SystemExit("GUT_RECORD=1 needs TYPESAFE_API_KEY to record from.")
        live = gut.JevBackend(model=MODEL)
    if not CASSETTE.exists() and live is None:
        raise SystemExit(f"No cassette at {CASSETTE}. Re-record with GUT_RECORD=1 and a key.")

    backend = CassetteBackend(CASSETTE, live, model=MODEL)
    gut.configure(backend=backend)
    return backend


@dataclass
class Row:
    """One posture, over the whole dataset."""

    label: str
    automatic: int = 0
    to_human: int = 0
    false_yes: int = 0
    false_no: int = 0
    cost: float = 0.0
    team_right: int = 0
    urgency_error: float = 0.0
    total: int = 0

    def record(self, action: Any, labels: dict[str, Any]) -> None:
        """Fold one ticket's outcome in."""
        self.total += 1
        truth = bool(labels["cancel_threat"])
        if action.to_human:
            self.to_human += 1
            self.cost += COST_HUMAN_REVIEW
        else:
            self.automatic += 1
            if action.escalate and not truth:
                self.false_yes += 1
                self.cost += COST_FALSE_ESCALATION
            elif not action.escalate and truth:
                self.false_no += 1
                self.cost += COST_MISSED_THREAT
        self.team_right += action.team == labels["team"]
        self.urgency_error += abs(action.urgency - labels["urgency"])

    def line(self) -> str:
        """The row, formatted for the table."""
        auto = self.automatic / self.total
        human = self.to_human / self.total
        return (
            f"  {self.label:<26} {auto:>5.0%} {human:>7.0%}"
            f" {self.false_yes:>10} {self.false_no:>9} {self.cost:>7.0f}"
            f" {self.team_right / self.total:>7.0%}"
        )


def sweep(tickets: list[dict[str, Any]]) -> list[Row]:
    """Run every posture over every ticket."""
    rows = []
    for label, posture in POSTURES:
        row = Row(label=label)
        for ticket in tickets:
            row.record(triage(state_of(ticket), **posture), ticket["labels"])
        rows.append(row)
    return rows


def before_rule(tickets: list[dict[str, Any]], backend: Any) -> tuple[int, float]:
    """What a hard 0.7 threshold with no third branch would have done, on the same answers."""
    from after import AT_RISK

    from gut._batching import fetch
    from gut._questions import NoulSpec

    spec = NoulSpec(AT_RISK)
    missed = 0
    cost = 0.0
    for ticket in tickets:
        probability = fetch(state_of(ticket), [spec], backend)[spec].answer.p  # type: ignore[union-attr]
        truth = bool(ticket["labels"]["cancel_threat"])
        escalated = probability > BEFORE_THRESHOLD
        if escalated and not truth:
            cost += COST_FALSE_ESCALATION
        elif not escalated and truth:
            missed += 1
            cost += COST_MISSED_THREAT
    return missed, cost


def main() -> None:
    """Print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tickets")
    limit = parser.parse_args().limit

    backend = configure_backend()
    tickets = load_tickets()[:limit]
    rows = sweep(tickets)
    backend.save()

    total = len(tickets)
    threats = sum(1 for t in tickets if t["labels"]["cancel_threat"])
    print(f"\ngut demo - support ticket triage   ({total} tickets, {threats} real churn risks)")
    print("=" * 84)
    print("\nthe dial: the same answers, acted on differently")
    print(
        f"  {'posture':<26} {'auto':>5} {'human':>7} {'false yes':>10} "
        f"{'missed':>9} {'cost':>7} {'team':>7}"
    )
    print("  " + "-" * 74)
    for row in rows:
        print(row.line())
    print(
        f"\n  missed = churn risks decided as 'no'. cost = {COST_FALSE_ESCALATION:g} per needless"
    )
    print(f"  escalation, {COST_MISSED_THREAT:g} per miss, {COST_HUMAN_REVIEW:g} per human review.")

    missed, cost = before_rule(tickets, backend)
    print("\nbefore.py: a hard 0.7 threshold, no third branch")
    print(
        f"  {'threshold=0.7':<26} {1.0:>5.0%} {0.0:>7.0%} "
        f"{'-':>10} {missed:>9} {cost:>7.0f} {'-':>7}"
    )

    print("\nmodel")
    print(
        f"  {'judgments asked':<26} {len(backend.cassette):>5}  "
        f"5 per ticket, reused across all {len(rows)} postures"
    )
    print(
        f"  {'requests they fit in':<26} {total:>5}  one per ticket, because @semantic batches them"
    )
    drift = statistics.fmean(row.urgency_error / row.total for row in rows)
    print(f"  {'urgency mean error':<26} {drift:>5.2f}  levels, over {len(Team)} teams\n")


if __name__ == "__main__":
    main()
