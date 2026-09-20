"""The same triage handler, written with `gut`.

Compare with `before.py`. The differences are not about line count:

1. **No thresholds.** Nothing here says `0.7`. It says how bad each mistake is, in words, and the
   runtime works out where the boundaries go.
2. **There is a third branch.** `UNSURE` is a real outcome with a real destination, so a judgment
   the model is not sure enough to make reaches a person instead of becoming a confident guess.
3. **One request per ticket.** `@semantic` collapses all five judgments into a single call.
4. **Nothing is parsed.** No prompt strings, no JSON scraping, no fallbacks for the wrong shape.

The handler takes its posture as arguments so `report.py` can sweep every setting over the same
tickets. Written for one setting it is just:

```python
at_risk = likely(ticket, "the customer is at risk of leaving...",
                 stakes="high", lean="yes", ask_human=True)
```
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

# The five judgments, worded once. predicates/ asks exactly these, so the eval suite measures the
# questions the handler actually asks rather than a paraphrase of them.
AT_RISK = (
    "the customer is at risk of leaving: they threaten to cancel, say they will not renew, or "
    "describe being close to giving up on the product"
)
WANTS_MONEY_BACK = "the customer is asking for money back or a credit"
REPORTS_A_BUG = "this reports something in the product behaving incorrectly"
OWNING_TEAM = "which team should own this ticket"
HOW_URGENT = "how urgent is this ticket"


@dataclass(frozen=True)
class Action:
    """What triage decided to do with a ticket."""

    escalate: bool
    to_human: bool
    team: str
    urgency: float
    reason: str


@semantic
def triage(
    ticket: dict[str, Any],
    *,
    stakes: gut.Stakes | None = None,
    lean: gut.Lean | None = None,
    ask_human: bool = False,
) -> Action:
    """Decide what happens to one ticket.

    Five judgments, one request. The decorator reads this function once and asks everything up
    front; the calls below find their answers already waiting.
    """
    at_risk = likely(ticket, AT_RISK, stakes=stakes, lean=lean, ask_human=ask_human)
    team = classify(ticket, Team, question=OWNING_TEAM, ask_human=ask_human, stakes=stakes)
    urgency = rate(ticket, URGENCY, question=HOW_URGENT)

    match at_risk:
        case gut.YES:
            return Action(True, False, _name(team), urgency.score, "churn risk")
        case gut.UNSURE:
            return Action(False, True, _name(team), urgency.score, "unsure about churn risk")

    # Not a churn risk. Route it, and let the two remaining judgments shape the handling.
    if likely(ticket, WANTS_MONEY_BACK):
        return Action(False, False, "BILLING", urgency.score, "refund request")
    if likely(ticket, REPORTS_A_BUG):
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
    from report import configure_backend

    configure_backend()
    for ticket in load_tickets()[:5]:
        action = triage(state_of(ticket), stakes="medium", lean="yes", ask_human=True)
        print(
            f"{ticket['id']}  {action.reason:<24} team={action.team:<9} "
            f"urgency={action.urgency:.1f} "
            f"{'ESCALATE' if action.escalate else 'HUMAN' if action.to_human else ''}"
        )
