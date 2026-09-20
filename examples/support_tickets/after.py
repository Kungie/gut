"""The same triage handler, written with `gut`.

Compare with `before.py`. The differences worth noticing are not about line count:

1. **The thresholds are gone.** Nothing here says `0.7`. It says what a mistake costs, and the
   runtime works out the threshold -- which for these numbers is `0.038`, not a figure anyone
   guesses.
2. **There is a third branch.** `UNSURE` is a real outcome with a real destination, so a decision
   the model is not confident enough to make reaches a person instead of silently defaulting.
3. **One request per ticket.** `@semantic` collapses all five judgments into a single call, which
   `before.py` cannot do without restructuring the function.
4. **Nothing is parsed.** No prompt strings, no JSON scraping, no fallbacks for output that came
   back in the wrong shape.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gut
from gut import classify, likely, rate, semantic

TICKETS = Path(__file__).parent / "tickets.json"


class Team(enum.Enum):
    """Where a ticket should go. The member values are what the model is shown."""

    BILLING = "invoices, charges, refunds, payment methods, plan pricing"
    PLATFORM = "outages, errors, bugs, slowness, anything technically broken"
    ACCOUNT = "seats, access, permissions, login, ownership, provisioning"
    OTHER = "anything that fits none of the others, including sales and how-to questions"


URGENCY = [
    "can wait, no one is blocked",
    "needs attention this week",
    "needs attention today, someone is blocked right now",
]

# What each mistake costs us, on one scale. These are the only numbers in the file.
COST_FALSE_ESCALATION = 2.0
"""A CSM spends twenty minutes on a customer who was never going to leave."""
COST_MISSED_THREAT = 20.0
"""A churn signal goes unanswered. Ten times worse, which is why the threshold ends up at 0.09."""
COST_HUMAN_REVIEW = 1.0
"""An agent reads the ticket and decides. Cheap, but not free -- and it has to be cheaper than a
needless escalation, or asking a person is never the cheapest option and UNSURE becomes
unreachable. `gut` warns when that happens."""


@dataclass(frozen=True)
class Action:
    """What triage decided to do with a ticket."""

    escalate: bool
    to_human: bool
    team: str
    urgency: float
    reason: str


@semantic
def triage(ticket: dict[str, Any]) -> Action:
    """Decide what happens to one ticket.

    Five judgments, one request. The decorator reads this function once and asks everything up
    front; the calls below find their answers already waiting.
    """
    at_risk = likely(
        ticket,
        "the customer is at risk of leaving: they threaten to cancel, say they will not renew, "
        "or describe being close to giving up on the product",
        cost_false_yes=COST_FALSE_ESCALATION,
        cost_false_no=COST_MISSED_THREAT,
        cost_human=COST_HUMAN_REVIEW,
    )
    team = classify(ticket, Team, min_confidence=0.55)
    urgency = rate(ticket, URGENCY)

    match at_risk:
        case gut.YES:
            return Action(
                escalate=True,
                to_human=False,
                team=_name(team),
                urgency=urgency.score,
                reason="churn risk",
            )
        case gut.UNSURE:
            return Action(
                escalate=False,
                to_human=True,
                team=_name(team),
                urgency=urgency.score,
                reason="unsure about churn risk",
            )

    # Not a churn risk. Route it, and let the two remaining judgments shape the handling.
    if likely(ticket, "the customer is asking for money back or a credit"):
        return Action(False, False, "BILLING", urgency.score, "refund request")
    if likely(ticket, "this reports something in the product behaving incorrectly"):
        return Action(False, False, _name(team), urgency.score, "bug report")
    return Action(False, False, _name(team), urgency.score, "routine")


def _name(team: gut.ChoiceDecision[Team]) -> str:
    """The chosen team, or `UNSURE` when the model was not confident enough to pick one."""
    return team.value.name if team.value is not gut.UNSURE else "UNSURE"


def load_tickets() -> list[dict[str, Any]]:
    """The demo dataset, labels included."""
    return json.loads(TICKETS.read_text(encoding="utf-8"))


def state_of(ticket: dict[str, Any]) -> dict[str, str]:
    """What the model is shown: the ticket, never its labels."""
    return {"subject": ticket["subject"], "body": ticket["body"]}


if __name__ == "__main__":
    import sys

    from _simulate import configure_backend

    configure_backend(load_tickets())
    for ticket in load_tickets()[:5]:
        action = triage(state_of(ticket))
        print(
            f"{ticket['id']}  {action.reason:<24} team={action.team:<9} "
            f"urgency={action.urgency:.1f} "
            f"{'ESCALATE' if action.escalate else 'HUMAN' if action.to_human else ''}"
        )
    sys.exit(0)
