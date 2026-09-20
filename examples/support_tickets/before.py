"""The triage handler as you would write it without `gut`.

This is not a straw man. It is what the code looks like when the judgment calls are bolted on: a
call per question, a threshold someone picked, a mapping from the model's label back to your own
type, and a defensive layer for when the answer is not the shape you expected.

Run it with a real key:

```bash
export TYPESAFE_API_KEY=...
python before.py
```

Four things are wrong with it, and none of them are style:

1. **`0.7` is made up.** Nobody derived it. It is the number that felt about right, and it encodes
   a claim about the relative cost of escalating a calm customer versus losing an angry one that
   nobody wrote down or checked. The cost-equivalent threshold here turns out to be `0.038`.
2. **There is no third branch.** A probability of 0.52 and a probability of 0.99 take the same
   path. "We don't know" is not expressible, so it silently becomes "no".
3. **Five calls per ticket.** The state is re-sent and re-billed every time, because nothing here
   knows the questions belong together.
4. **The plumbing is load-bearing.** The label mapping, the `.get` fallbacks and the try/except
   are all places a change in the model's output becomes a change in your control flow.

A text LLM instead of a typed one adds a fifth: parsing a probability out of prose, and deciding
what to do when it says "about 70%" or returns markdown.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TICKETS = Path(__file__).parent / "tickets.json"

# Someone picked these. There is no record of why.
CHURN_THRESHOLD = 0.7
REFUND_THRESHOLD = 0.7
BUG_THRESHOLD = 0.7
TEAM_CONFIDENCE_FLOOR = 0.55

TEAMS = {
    "BILLING": "invoices, charges, refunds, payment methods, plan pricing",
    "PLATFORM": "outages, errors, bugs, slowness, anything technically broken",
    "ACCOUNT": "seats, access, permissions, login, ownership, provisioning",
    "OTHER": "anything that fits none of the others, including sales and how-to questions",
}

URGENCY = [
    "can wait, no one is blocked",
    "needs attention this week",
    "needs attention today, someone is blocked right now",
]


@dataclass(frozen=True)
class Action:
    """What triage decided to do with a ticket."""

    escalate: bool
    to_human: bool
    team: str
    urgency: float
    reason: str


def _client() -> Any:
    from typesafe_sdk import TypeSafeClient

    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("before.py talks to the real API; set TYPESAFE_API_KEY.")
    return TypeSafeClient(model="jev-1.13.0")


def triage(ticket: dict[str, Any], client: Any) -> Action:
    """Decide what happens to one ticket. Five round trips, all of them paying for the state."""
    from typesafe_sdk import Choice, Noul, Score

    churn = client.system_one(
        state=ticket,
        questions={
            "q": Noul(
                instructions="the customer is at risk of leaving: they threaten to cancel, say "
                "they will not renew, or describe being close to giving up on the product"
            )
        },
    )
    team_response = client.system_one(
        state=ticket, questions={"q": Choice(instructions="which team owns this", criteria=TEAMS)}
    )
    urgency_response = client.system_one(
        state=ticket, questions={"q": Score(instructions="how urgent is this", criteria=URGENCY)}
    )
    refund = client.system_one(
        state=ticket,
        questions={"q": Noul(instructions="the customer is asking for money back or a credit")},
    )
    bug = client.system_one(
        state=ticket,
        questions={
            "q": Noul(instructions="this reports something in the product behaving incorrectly")
        },
    )

    # Map the model's label back to our own vocabulary, and hope it is one we know.
    try:
        chosen = team_response.answers["q"].choice
        confidence = team_response.answers["q"].confidence
    except (KeyError, AttributeError):
        chosen, confidence = "OTHER", 0.0
    team = chosen if chosen in TEAMS and confidence >= TEAM_CONFIDENCE_FLOOR else "OTHER"

    urgency = float(getattr(urgency_response.answers.get("q"), "score", 0.0))
    churn_p = float(getattr(churn.answers.get("q"), "noul", 0.0))

    # Two outcomes. A 0.69 and a 0.01 are treated identically.
    if churn_p > CHURN_THRESHOLD:
        return Action(True, False, team, urgency, "churn risk")
    if float(getattr(refund.answers.get("q"), "noul", 0.0)) > REFUND_THRESHOLD:
        return Action(False, False, "BILLING", urgency, "refund request")
    if float(getattr(bug.answers.get("q"), "noul", 0.0)) > BUG_THRESHOLD:
        return Action(False, False, team, urgency, "bug report")
    return Action(False, False, team, urgency, "routine")


if __name__ == "__main__":
    tickets = json.loads(TICKETS.read_text(encoding="utf-8"))
    client = _client()
    for ticket in tickets[:5]:
        state = {"subject": ticket["subject"], "body": ticket["body"]}
        action = triage(state, client)
        print(
            f"{ticket['id']}  {action.reason:<18} team={action.team:<9} "
            f"urgency={action.urgency:.1f} {'ESCALATE' if action.escalate else ''}"
        )
