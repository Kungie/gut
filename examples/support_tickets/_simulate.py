"""A stand-in model, so the demo runs without an API key.

With `TYPESAFE_API_KEY` set the report talks to the real Jev. Without it, this builds a
`FakeBackend` that answers from the dataset's own labels -- imperfectly, and deterministically.

The imperfection is the point. A simulator that is always right produces a report where every
number is trivial: nothing is ever escalated wrongly, nothing ever reaches a human, and the cost
column is zero. So this one is confidently right on the straightforward tickets, much less sure on
the ones the dataset marks `hard`, and wrong often enough on those to exercise the parts of the
report that matter.

It is a demo prop, not a model. Every number it produces is derived from the answer key.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from typing import Any

import gut
from gut._backends.fake import FakeValue
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State, canonical_json

CONFIDENT = 0.96
"""How sure the stand-in is on a straightforward ticket."""
HESITANT = 0.58
"""How sure it is on one the dataset marks hard -- well inside the band where a person is
the cheapest option."""
WRONG_WHEN_HARD = 0.35
"""How often it gets a hard ticket outright wrong."""


def _unit(*parts: str) -> float:
    """A stable pseudo-random number in [0, 1) from the given strings."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class Simulator:
    """Answers questions about the demo tickets from their labels."""

    def __init__(self, tickets: list[dict[str, Any]]) -> None:
        self._by_state = {
            canonical_json({"subject": t["subject"], "body": t["body"]}): t for t in tickets
        }

    def _ticket(self, state: State) -> dict[str, Any] | None:
        return self._by_state.get(canonical_json(state))

    def __call__(self, state: State, name: str, spec: QuestionSpec) -> FakeValue | None:
        ticket = self._ticket(state)
        if ticket is None:
            return None
        labels: Mapping[str, Any] = ticket["labels"]
        hard = "hard" in ticket
        seed = _unit(ticket["id"], spec.fingerprint)
        confidence = HESITANT if hard else CONFIDENT
        mistaken = hard and seed < WRONG_WHEN_HARD

        if isinstance(spec, NoulSpec):
            truth = self._noul_truth(spec, labels)
            if truth is None:
                return None
            if mistaken:
                truth = not truth
            spread = 0.04 * (2 * _unit(spec.fingerprint, ticket["id"]) - 1)
            return max(0.0, min(1.0, (confidence if truth else 1 - confidence) + spread))

        if isinstance(spec, ChoiceSpec):
            options = list(spec.criteria)
            correct = str(labels["team"])
            if correct not in options:  # pragma: no cover - the demo enum covers every label
                return None
            if mistaken:
                others = [option for option in options if option != correct]
                return others[int(seed * len(others)) % len(others)]
            return correct

        if isinstance(spec, ScoreSpec):
            level = float(labels["urgency"])
            drift = (0.8 if hard else 0.25) * (2 * seed - 1)
            top = len(spec.criteria) - 1
            return max(0.0, min(float(top), level + drift))

        return None  # pragma: no cover - there is no fourth question type

    @staticmethod
    def _noul_truth(spec: NoulSpec, labels: Mapping[str, Any]) -> bool | None:
        question = (spec.instructions or "").lower()
        if "at risk of leaving" in question:
            return bool(labels["cancel_threat"])
        if "money back" in question:
            return bool(labels["refund_request"])
        if "behaving incorrectly" in question:
            return bool(labels["bug_report"])
        return None  # pragma: no cover - the demo asks no other yes/no question


class Counting:
    """Wraps a backend and counts what it was asked.

    The report claims batching collapses five judgments into one request. That claim needs a
    number, and the real backend does not keep one.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.call_count = 0
        self.question_count = 0

    @property
    def model_id(self) -> str:
        """The wrapped backend's model."""
        return str(self.inner.model_id)

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> Any:
        """Count the request, then pass it along."""
        self.call_count += 1
        self.question_count += len(questions)
        return self.inner.ask(state, questions)


def configure_backend(tickets: list[dict[str, Any]]) -> tuple[Any, str]:
    """Point `gut` at the real model if a key is present, and at the stand-in otherwise.

    Returns:
        The backend and a one-word description of what it is, for the report header.
    """
    if os.environ.get("TYPESAFE_API_KEY", "").strip():
        inner: Any = gut.JevBackend(model="jev-1.13.0")
        label = "live"
    else:
        inner = gut.FakeBackend(rule=Simulator(tickets))
        label = "simulated"
    backend = Counting(inner)
    # Caching off so the call count in the report measures batching, not repetition.
    gut.configure(backend=backend, cache=gut.NullCache())
    return backend, label
