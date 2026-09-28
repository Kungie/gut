"""Could this meeting have been an email? A second opinion for your calendar.

    python examples/meeting_or_email.py
    python examples/meeting_or_email.py --backend jev

One concrete question does the work: does the invite ask people to decide something together? If
not, it could have been an email. `ask_human=True` leaves room for the invites that are too vague to
call -- which is its own kind of answer.
"""

from __future__ import annotations

import gut

INVITES = [
    "Sync (60 min). Let's sync.",
    "Quick update on Q3 numbers (45 min). I'll walk through the slides I already emailed.",
    "Choose the launch date (30 min). Marketing needs July or September; legal and ops both "
    "have constraints. We leave with a date.",
    "Brainstorm (90 min). Bring ideas!",
    "FYI: the office wifi password is changing (30 min).",
]


def advise(invite: str) -> str:
    match gut.likely(invite, "asks the attendees to make a decision", ask_human=True):
        case gut.YES:
            return "Go. There is something to decide, and deciding needs people."
        case gut.NO:
            return "Email. Reply with 'Could you send this over instead?'"
        case _:
            return "Ask for an agenda. Nobody, including the model, can tell what this is for."


def main() -> None:
    from _pick import backend_from_argv

    gut.configure(backend=backend_from_argv(__doc__ or "", default="qwen"))
    for invite in INVITES:
        print(f"{invite[:58]:<58}  {advise(invite)}")


if __name__ == "__main__":
    main()
