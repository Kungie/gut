"""Your own model as a backend, in a page -- here, the keyword rules you already had.

A backend is anything with a `model_id` and an `ask()` that returns probabilities. This one knows a
few phrases and is sure about them, and says "no idea" about everything else. On its own that is a
poor classifier. As the first stage of a `gut.Cascade` it is free, instant and right about what it
recognises, and the model behind it only sees the rest.

    python examples/custom_backend.py                  # keywords first, then an NLI model
    python examples/custom_backend.py --backend qwen   # keywords first, then Qwen3-0.6B
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

import gut
from gut import QuestionSpec, State


class KeywordBackend:
    """Certain when a rule matches, and honestly clueless when none does."""

    model_id = "keywords-1"

    def __init__(self, rules: Mapping[str, Sequence[str]]) -> None:
        self.rules = {
            question: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for question, patterns in rules.items()
        }

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> gut.BackendResponse:
        text = state if isinstance(state, str) else json.dumps(state)
        answers: dict[str, gut.Answer] = {}
        for name, spec in questions.items():
            patterns = self.rules.get(spec.instructions or "", [])
            if isinstance(spec, gut.NoulSpec) and any(p.search(text) for p in patterns):
                answers[name] = gut.NoulAnswer(p=0.99)
            else:
                answers[name] = no_idea(spec)
        return gut.BackendResponse(answers=answers, model=self.model_id)


def no_idea(spec: QuestionSpec) -> gut.Answer:
    """An answer that says nothing, so a cascade always escalates it."""
    match spec:
        case gut.NoulSpec():
            return gut.NoulAnswer(p=0.5)
        case gut.ChoiceSpec():
            share = 1 / len(spec.criteria)
            return gut.ChoiceAnswer(
                choice=next(iter(spec.criteria)),
                confidence=share,
                probabilities=dict.fromkeys(spec.criteria, share),
            )
        case _:
            share = 1 / len(spec.criteria)
            levels = dict.fromkeys(range(len(spec.criteria)), share)
            return gut.ScoreAnswer(
                score=sum(level * share for level in levels),
                confidence=share,
                probabilities=levels,
                legend=dict(enumerate(spec.criteria)),
            )


RULES = {
    "asks for a refund": [r"\brefund\b", r"money back", r"charged twice"],
    "is a bug report": [r"\b5\d\d\b error", r"\bcrash(es|ed)?\b", r"stack ?trace"],
}

TICKETS = [
    "I was charged twice this month.",
    "The app crashes when I rotate my phone.",
    "This isn't what I ordered and I'd like to return it for my money.",
    "Clicking save does nothing at all.",
]


def run(
    fallback: gut.Backend, tickets: list[str]
) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    cascade = gut.Cascade(KeywordBackend(RULES), fallback)
    gut.configure(backend=cascade)
    rows = []
    for ticket in tickets:
        with gut.judge(ticket) as j:
            refund = j.likely("asks for a refund")
            bug = j.likely("is a bug report")
        rows.append(
            (
                ticket,
                f"refund={refund.outcome} ({refund.model})",
                f"bug={bug.outcome} ({bug.model})",
            )
        )
    return rows, dict(cascade.answered_by)


def main() -> None:
    from _pick import backend_from_argv

    rows, answered_by = run(backend_from_argv(__doc__ or ""), TICKETS)
    for ticket, refund, bug in rows:
        print(f"{ticket[:58]:<58}  {refund:<40}  {bug}")
    print("\nanswers per model:", answered_by)


if __name__ == "__main__":
    main()
