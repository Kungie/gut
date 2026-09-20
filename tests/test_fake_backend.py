"""Tests for the offline backend: fixture lookup, coercion, strictness, and call counting."""

from __future__ import annotations

import subprocess
import sys

import pytest

from gut._backends import (
    ChoiceAnswer,
    FakeBackend,
    NoulAnswer,
    ScoreAnswer,
    deterministic_rule,
)
from gut._backends.base import Backend
from gut._errors import BackendError
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State

BUG = NoulSpec("is a bug report")
TEAM = ChoiceSpec("which team owns this", {"billing": "invoices", "platform": "outages"})
URGENCY = ScoreSpec("how urgent is this", ["can wait", "this week", "today"])


def test_fake_backend_satisfies_the_protocol() -> None:
    assert isinstance(FakeBackend(), Backend)


# --------------------------------------------------------------------------- lookup


def test_fixtures_are_found_by_question_text() -> None:
    backend = FakeBackend(answers={"is a bug report": 0.91})
    response = backend.ask("a ticket", {"bug": BUG})
    assert response.answers["bug"] == NoulAnswer(p=0.91)
    assert response.model == "fake-1.0"


def test_fixtures_are_found_by_fingerprint() -> None:
    backend = FakeBackend(answers={BUG.fingerprint: 0.2})
    assert backend.ask("a ticket", {"bug": BUG}).answers["bug"] == NoulAnswer(p=0.2)


def test_a_rule_takes_precedence_over_fixtures() -> None:
    seen: list[tuple[str, str]] = []

    def rule(state: State, name: str, spec: QuestionSpec) -> float | None:
        seen.append((str(state), name))
        return 1.0 if "bug" in str(state) else None

    backend = FakeBackend(answers={"is a bug report": 0.0}, rule=rule)
    assert backend.ask("a bug", {"q": BUG}).answers["q"] == NoulAnswer(p=1.0)
    # Returning None falls through to the fixtures.
    assert backend.ask("a question", {"q": BUG}).answers["q"] == NoulAnswer(p=0.0)
    assert seen == [("a bug", "q"), ("a question", "q")]


def test_default_answers_everything_else() -> None:
    backend = FakeBackend(default=0.5)
    assert backend.ask("anything", {"q": BUG}).answers["q"] == NoulAnswer(p=0.5)


def test_an_unconfigured_question_raises_rather_than_inventing_an_answer() -> None:
    with pytest.raises(BackendError, match="no fixture for 'is a bug report'"):
        FakeBackend().ask("a ticket", {"bug": BUG})


def test_an_empty_batch_is_rejected() -> None:
    with pytest.raises(BackendError, match="at least one question"):
        FakeBackend().ask("a ticket", {})


# --------------------------------------------------------------------------- coercion


@pytest.mark.parametrize(("value", "expected"), [(True, 1.0), (False, 0.0), (0.3, 0.3), (1, 1.0)])
def test_noul_shorthands(value: float | bool, expected: float) -> None:
    backend = FakeBackend(answers={"is a bug report": value})
    assert backend.ask("t", {"q": BUG}).answers["q"] == NoulAnswer(p=expected)


def test_choice_shorthand_builds_a_distribution() -> None:
    backend = FakeBackend(answers={"which team owns this": "platform"})
    answer = backend.ask("t", {"q": TEAM}).answers["q"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "platform"
    assert answer.probabilities == {"billing": pytest.approx(0.1), "platform": 0.9}
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_score_shorthand_on_a_level() -> None:
    backend = FakeBackend(answers={"how urgent is this": 2})
    answer = backend.ask("t", {"q": URGENCY}).answers["q"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == 2.0
    assert answer.probabilities == {2: 1.0}
    assert answer.confidence == 1.0
    assert answer.legend == {0: "can wait", 1: "this week", 2: "today"}


def test_score_shorthand_between_levels() -> None:
    backend = FakeBackend(answers={"how urgent is this": 1.25})
    answer = backend.ask("t", {"q": URGENCY}).answers["q"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == 1.25
    assert answer.probabilities == {1: pytest.approx(0.75), 2: pytest.approx(0.25)}
    assert answer.confidence == pytest.approx(0.75)


def test_a_full_answer_is_passed_through_untouched() -> None:
    exact = ChoiceAnswer(
        choice="billing", confidence=0.55, probabilities={"billing": 0.55, "platform": 0.45}
    )
    backend = FakeBackend(answers={"which team owns this": exact})
    assert backend.ask("t", {"q": TEAM}).answers["q"] is exact


@pytest.mark.parametrize(
    ("spec", "answer"),
    [
        (BUG, NoulAnswer(p=0.42)),
        (
            URGENCY,
            ScoreAnswer(
                score=1.5,
                confidence=0.5,
                probabilities={1: 0.5, 2: 0.5},
                legend={0: "can wait", 1: "this week", 2: "today"},
            ),
        ),
    ],
)
def test_every_answer_type_passes_through_untouched(spec: QuestionSpec, answer: object) -> None:
    backend = FakeBackend(answers={spec.instructions: answer})  # type: ignore[dict-item]
    assert backend.ask("t", {"q": spec}).answers["q"] is answer


@pytest.mark.parametrize(
    ("spec", "value", "message"),
    [
        (BUG, 1.5, r"probability in \[0, 1\]"),
        (BUG, "yes", "must be a probability, a bool"),
        (TEAM, 1, "must be an option name"),
        (TEAM, "shipping", "is not one of its options"),
        (URGENCY, "today", "must be a level number"),
        (URGENCY, True, "must be a level number"),
        (URGENCY, 5, "must be a level between 0 and 2"),
        (URGENCY, -1, "must be a level between 0 and 2"),
    ],
)
def test_bad_fixtures_are_reported_against_the_question(
    spec: QuestionSpec, value: object, message: str
) -> None:
    backend = FakeBackend(answers={spec.instructions: value})  # type: ignore[dict-item]
    with pytest.raises(BackendError, match=message):
        backend.ask("t", {"q": spec})


def test_the_deterministic_rule_answers_anything_reproducibly() -> None:
    backend = FakeBackend(rule=deterministic_rule)
    first = backend.ask("a ticket", {"a": BUG, "b": TEAM, "c": URGENCY})
    second = FakeBackend(rule=deterministic_rule).ask(
        "a ticket", {"a": BUG, "b": TEAM, "c": URGENCY}
    )
    assert first.answers == second.answers

    noul = first.answers["a"]
    assert isinstance(noul, NoulAnswer)
    assert 0.0 <= noul.p <= 1.0
    choice = first.answers["b"]
    assert isinstance(choice, ChoiceAnswer)
    assert choice.choice in TEAM.criteria
    score = first.answers["c"]
    assert isinstance(score, ScoreAnswer)
    assert 0 <= score.score <= len(URGENCY.criteria) - 1

    # Different state, and the answers are free to differ.
    other = backend.ask("another ticket", {"a": BUG})
    assert isinstance(other.answers["a"], NoulAnswer)


def test_the_deterministic_rule_is_stable_across_processes() -> None:
    """Seeded with SHA-256, not `hash()`, whose string hashing is salted per process."""
    script = (
        "from gut._backends import FakeBackend, deterministic_rule;"
        "from gut._questions import NoulSpec;"
        "b = FakeBackend(rule=deterministic_rule);"
        "print(b.ask('a ticket', {'a': NoulSpec('is a bug report')}).answers['a'].p)"
    )
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    ]

    in_process = FakeBackend(rule=deterministic_rule).ask("a ticket", {"a": BUG}).answers["a"]
    assert isinstance(in_process, NoulAnswer)
    assert outputs == [str(in_process.p), str(in_process.p)]


# --------------------------------------------------------------------------- call counting


def test_calls_are_recorded_so_batching_can_be_asserted() -> None:
    backend = FakeBackend(rule=deterministic_rule)
    assert backend.call_count == 0

    backend.ask("ticket", {"a": BUG, "b": TEAM, "c": URGENCY})
    assert backend.call_count == 1
    assert backend.calls[0].names == ("a", "b", "c")
    assert backend.calls[0].state == "ticket"

    backend.ask("ticket", {"d": BUG})
    assert backend.call_count == 2

    backend.reset()
    assert backend.call_count == 0
    assert backend.rule is deterministic_rule  # configuration survives a reset
