"""Tests for answer serialisation, including the string-keys-to-int-levels asymmetry."""

from __future__ import annotations

from typing import Any

import pytest

from gut._backends.base import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._errors import BackendError
from gut._serde import answer_from_json, answer_to_json

ANSWERS: list[Answer] = [
    NoulAnswer(p=0.83),
    ChoiceAnswer(choice="billing", confidence=0.9, probabilities={"billing": 0.9, "other": 0.1}),
    ScoreAnswer(
        score=1.3,
        confidence=0.7,
        probabilities={0: 0.0, 1: 0.7, 2: 0.3},
        legend={0: "can wait", 1: "this week", 2: "today"},
    ),
]


@pytest.mark.parametrize("answer", ANSWERS, ids=lambda a: type(a).__name__)
def test_answers_round_trip(answer: Answer) -> None:
    assert answer_from_json(answer_to_json(answer)) == answer


def test_score_levels_survive_the_trip_through_string_keys() -> None:
    """JSON object keys are strings; score levels are integers. The coercion must be exact."""
    original = ANSWERS[2]
    assert isinstance(original, ScoreAnswer)
    encoded = answer_to_json(original)
    assert set(encoded["probabilities"]) == {"0", "1", "2"}

    decoded = answer_from_json(encoded)
    assert isinstance(decoded, ScoreAnswer)
    assert set(decoded.probabilities) == {0, 1, 2}
    assert decoded.legend[2] == "today"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "unknown type None"),
        ({"type": "wat"}, "unknown type 'wat'"),
        ({"type": "noul"}, "missing 'p'"),
        ({"type": "noul", "p": "high"}, "'p' as str"),
        ({"type": "noul", "p": True}, "'p' as bool"),
        ({"type": "choice", "choice": "a"}, "missing 'confidence'"),
        (
            {"type": "choice", "choice": 1, "confidence": 0.5, "probabilities": {}},
            "'choice' as int",
        ),
        (
            {
                "type": "score",
                "score": 1.0,
                "confidence": 0.5,
                "probabilities": {"x": 1.0},
                "legend": {},
            },
            "non-integer level keys",
        ),
    ],
)
def test_corrupt_payloads_are_rejected(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(BackendError, match=message):
        answer_from_json(payload)
