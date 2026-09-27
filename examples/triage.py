"""Support ticket triage: which queue, how soon, and whether a person should look first.

    python examples/triage.py                    # a 70M-parameter NLI model, on this machine
    python examples/triage.py --backend qwen     # Qwen3-0.6B, on this machine
    python examples/triage.py --backend openai   # gpt-4.1-nano

`triage()` never names a model. `@gut.semantic` turns its three judgments into one request per
ticket, and the judgments that would hurt if wrong can answer UNSURE instead of guessing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import gut


class Team(enum.Enum):
    """The member values are what the model reads; the names are what comes back."""

    BILLING = "payments, invoices, refunds and pricing"
    ENGINEERING = "bugs, errors, outages and anything technically broken"
    ACCOUNT = "logins, passwords, access and permissions"
    OTHER = "anything else"


URGENCY = [
    "no one is waiting on this",
    "someone needs this within a few days",
    "someone is blocked right now",
]


@dataclass(frozen=True)
class Route:
    queue: str
    priority: int
    """0 is most urgent."""
    note: str = ""


@gut.semantic
def triage(ticket: str) -> Route:
    leaving = gut.likely(ticket, "the customer threatens to cancel", lean="yes", ask_human=True)
    team = gut.classify(ticket, Team, ask_human=True)
    urgency = gut.rate(ticket, URGENCY)

    match leaving:
        case gut.YES:
            return Route("retention", 0, "said they might leave")
        case gut.UNSURE:
            note = "might be a churn risk"
        case _:
            note = ""

    queue = team.value.name.lower() if isinstance(team.value, Team) else "front desk"
    return Route(queue, len(URGENCY) - 1 - urgency.nearest_level, note)


TICKETS = [
    "I was charged twice for September. Please refund the duplicate payment.",
    "Since this morning every export to CSV fails with a 500 error. Finance needs it by 5pm.",
    "I can't log in: the password reset email never arrives.",
    "Third outage this month. If it happens again we're moving to another vendor.",
    "Do you have a discount for nonprofits?",
]


def run(tickets: list[str]) -> list[tuple[str, Route]]:
    return [(ticket, triage(ticket)) for ticket in tickets]


def main() -> None:
    from _pick import backend_from_argv

    gut.configure(backend=backend_from_argv(__doc__ or ""))
    for ticket, route in run(TICKETS):
        print(f"{route.queue:<12} P{route.priority}  {ticket[:64]:<64}  {route.note}")


if __name__ == "__main__":
    main()
