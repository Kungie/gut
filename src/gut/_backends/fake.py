"""A deterministic backend for tests and offline development.

`FakeBackend` answers from fixtures you supply, never from the network, and always the same way for
the same input. It is strict by default: a question nobody configured raises rather than quietly
returning 0.5, because a test that passes on an invented probability is worse than one that fails.

It also counts calls. That is not a convenience -- "these five questions became one request" is a
claim `gut` makes, and `backend.call_count` is how it gets checked.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

from gut._backends.base import Answer, BackendResponse, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._errors import BackendError
from gut._questions import (
    ChoiceSpec,
    NoulSpec,
    QuestionSpec,
    ScoreSpec,
    State,
    canonical_json,
)

FakeValue: TypeAlias = "float | int | bool | str | Answer"
"""A fixture value: a full `Answer`, or a shorthand -- a probability or bool for a noul, an option
name for a choice, a level number for a score."""

Rule: TypeAlias = "Callable[[State, str, QuestionSpec], FakeValue | None]"
"""A callable consulted for every question; returning `None` falls through to the fixtures."""

DEFAULT_CONFIDENCE: float = 0.9
"""Probability given to the named option when a choice fixture is just an option name."""


def deterministic_rule(state: State, name: str, spec: QuestionSpec) -> FakeValue:
    """Answer any question reproducibly, derived from the state and the question itself.

    For offline development, where you want the code path to run rather than a particular answer.
    The answers are arbitrary but stable: the same state and question always produce the same
    result, across processes and across machines.

    Seeded with SHA-256 rather than `hash()`, whose string hashing is salted per process and would
    make "deterministic" true only within a single run.
    """
    digest = hashlib.sha256(canonical_json([state, spec.canonical()]).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    match spec:
        case NoulSpec():
            return seed / float(1 << 64)
        case ChoiceSpec():
            return list(spec.criteria)[seed % len(spec.criteria)]
        case ScoreSpec():  # pragma: no branch - exhaustive, proven by mypy
            return float(seed % len(spec.criteria))


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One call made to the backend, kept so tests can assert how many there were."""

    state: State
    names: tuple[str, ...]
    specs: Mapping[str, QuestionSpec]


def _noul_from(value: FakeValue, name: str) -> NoulAnswer:
    if isinstance(value, NoulAnswer):
        return value
    if isinstance(value, bool):
        return NoulAnswer(p=1.0 if value else 0.0)
    if isinstance(value, int | float):
        p = float(value)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise BackendError(f"Fixture for {name!r} must be a probability in [0, 1], got {p!r}.")
        return NoulAnswer(p=p)
    raise BackendError(
        f"Fixture for the yes/no question {name!r} must be a probability, a bool, or a NoulAnswer; "
        f"got {type(value).__name__}."
    )


def _choice_from(value: FakeValue, name: str, spec: ChoiceSpec) -> ChoiceAnswer:
    if isinstance(value, ChoiceAnswer):
        return value
    if not isinstance(value, str):
        raise BackendError(
            f"Fixture for the choice question {name!r} must be an option name or a ChoiceAnswer; "
            f"got {type(value).__name__}."
        )
    if value not in spec.criteria:
        options = ", ".join(repr(option) for option in spec.criteria)
        raise BackendError(f"Fixture {value!r} for {name!r} is not one of its options: {options}.")
    others = len(spec.criteria) - 1
    remainder = (1.0 - DEFAULT_CONFIDENCE) / others
    probabilities = {
        option: DEFAULT_CONFIDENCE if option == value else remainder for option in spec.criteria
    }
    return ChoiceAnswer(choice=value, confidence=DEFAULT_CONFIDENCE, probabilities=probabilities)


def _score_from(value: FakeValue, name: str, spec: ScoreSpec) -> ScoreAnswer:
    if isinstance(value, ScoreAnswer):
        return value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BackendError(
            f"Fixture for the score question {name!r} must be a level number or a ScoreAnswer; "
            f"got {type(value).__name__}."
        )
    top = len(spec.criteria) - 1
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= top:
        raise BackendError(
            f"Fixture for {name!r} must be a level between 0 and {top}, got {value!r}."
        )
    legend = dict(enumerate(spec.criteria))
    lower = math.floor(score)
    upper = math.ceil(score)
    if lower == upper:
        probabilities = {lower: 1.0}
    else:
        fraction = score - lower
        probabilities = {lower: 1.0 - fraction, upper: fraction}
    return ScoreAnswer(
        score=score,
        confidence=max(probabilities.values()),
        probabilities=probabilities,
        legend=legend,
    )


def _coerce(value: FakeValue, name: str, spec: QuestionSpec) -> Answer:
    """Turn a fixture into the answer shape the question calls for."""
    match spec:
        case NoulSpec():
            return _noul_from(value, name)
        case ChoiceSpec():
            return _choice_from(value, name, spec)
        case ScoreSpec():  # pragma: no branch - exhaustive, proven by mypy
            return _score_from(value, name, spec)


@dataclass
class FakeBackend:
    """A `Backend` that answers from fixtures.

    Questions are looked up by their text first and by their fingerprint second, so fixtures read
    naturally:

    ```python
    backend = FakeBackend(answers={
        "is a bug report": 0.91,
        "which team owns this": "platform",
        "how urgent is this": 2,
    })
    ```

    Args:
        answers: Fixture values keyed by question text or fingerprint.
        rule: Consulted before `answers` for every question; return `None` to fall through.
        default: Used when nothing else matched. Left unset, an unmatched question raises. A
            single scalar cannot serve all three question types, so for mixed batches pass
            `rule=deterministic_rule` instead.
        model: The model id reported back, so recorded decisions are distinguishable from real ones.
    """

    answers: Mapping[str, FakeValue] = field(default_factory=dict)
    rule: Rule | None = None
    default: FakeValue | None = None
    model: str = "fake-1.0"
    calls: list[RecordedCall] = field(default_factory=list, init=False)

    @property
    def model_id(self) -> str:
        """The model this backend reports; there is no alias to resolve."""
        return self.model

    @property
    def call_count(self) -> int:
        """How many backend calls have been made -- the batching assertion in one number."""
        return len(self.calls)

    def reset(self) -> None:
        """Forget recorded calls, leaving the fixtures in place."""
        self.calls.clear()

    def _lookup(self, state: State, name: str, spec: QuestionSpec) -> FakeValue:
        if self.rule is not None:
            value = self.rule(state, name, spec)
            if value is not None:
                return value
        for key in (spec.instructions, spec.fingerprint):
            if key is not None and key in self.answers:
                return self.answers[key]
        if self.default is not None:
            return self.default
        described = (
            repr(spec.instructions)
            if spec.instructions
            else f"a {spec.canonical()['type']} question"
        )
        raise BackendError(
            f"FakeBackend has no fixture for {described} (fingerprint {spec.fingerprint}). Add it "
            f"to answers=, return it from rule=, or set default= to answer everything."
        )

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """Answer `questions` about `state` from the configured fixtures."""
        if not questions:
            raise BackendError("A backend call needs at least one question.")
        self.calls.append(RecordedCall(state=state, names=tuple(questions), specs=dict(questions)))
        answers = {
            name: _coerce(self._lookup(state, name, spec), name, spec)
            for name, spec in questions.items()
        }
        return BackendResponse(answers=answers, model=self.model)
