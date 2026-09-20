"""Tests for question specs: validation at the client, and stable fingerprints."""

from __future__ import annotations

import pytest

from gut._errors import QuestionError
from gut._questions import (
    ChoiceSpec,
    NoulSpec,
    ScoreSpec,
    canonical_json,
    state_fingerprint,
)


def test_canonical_json_is_independent_of_key_order() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"a": 1}) == '{"a":1}'


def test_canonical_json_keeps_non_ascii_readable() -> None:
    assert canonical_json({"q": "iptal mi?"}) == '{"q":"iptal mi?"}'


def test_fingerprints_are_stable_and_question_specific() -> None:
    first = NoulSpec("the customer threatens to cancel")
    same = NoulSpec("the customer threatens to cancel")
    other = NoulSpec("the customer asks how to cancel")

    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != other.fingerprint
    # Pinned: changing this string silently invalidates every stored decision id.
    assert first.fingerprint == "07bbb8dda08376eb"


def test_criteria_are_part_of_the_fingerprint() -> None:
    plain = NoulSpec("is this spam")
    described = NoulSpec("is this spam", yes_means="unsolicited advertising")
    assert plain.fingerprint != described.fingerprint
    assert "criteria" not in plain.canonical()
    assert described.canonical()["criteria"] == {"true": "unsolicited advertising"}


def test_score_levels_are_normalised_to_a_tuple() -> None:
    from_list = ScoreSpec("how urgent", ["low", "high"])
    from_tuple = ScoreSpec("how urgent", ("low", "high"))
    assert from_list == from_tuple
    assert from_list.fingerprint == from_tuple.fingerprint
    assert isinstance(from_list.criteria, tuple)


def test_choice_canonical_keeps_undescribed_options() -> None:
    spec = ChoiceSpec("which team", {"billing": "invoices", "platform": None})
    assert spec.canonical()["criteria"] == {"billing": "invoices", "platform": None}


def test_a_subject_json_cannot_hold_is_reported_clearly() -> None:
    """Coercing it would let two different subjects share a cached answer."""
    with pytest.raises(QuestionError, match="must be JSON-compatible"):
        canonical_json({"when": object()})


def test_state_fingerprint_covers_every_allowed_shape() -> None:
    assert state_fingerprint("hello") != state_fingerprint(["hello"])
    assert state_fingerprint({"a": 1, "b": 2}) == state_fingerprint({"b": 2, "a": 1})


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: NoulSpec(""), "non-empty string"),
        (lambda: NoulSpec("   "), "non-empty string"),
        (lambda: NoulSpec(42), "non-empty string"),  # type: ignore[arg-type]
        (lambda: NoulSpec("q", yes_means=""), "yes_means must be"),
        (lambda: NoulSpec("q", no_means=" "), "no_means must be"),
        (lambda: ScoreSpec("q", ["only one"]), "between 2 and 10 levels"),
        (lambda: ScoreSpec("q", [str(i) for i in range(11)]), "between 2 and 10 levels"),
        (lambda: ScoreSpec("q", "not a list"), "must be a sequence of strings"),
        (lambda: ScoreSpec("q", 5), "must be a sequence"),  # type: ignore[arg-type]
        (lambda: ScoreSpec("q", ["ok", ""]), "Level 1 must be"),
        (lambda: ChoiceSpec("q", {"only": None}), "between 2 and 255 options"),
        (lambda: ChoiceSpec("q", {str(i): None for i in range(256)}), "between 2 and 255"),
        (lambda: ChoiceSpec("q", ["a", "b"]), "must be a mapping"),  # type: ignore[arg-type]
        (lambda: ChoiceSpec("q", {"": None, "b": None}), "option name must be"),
        (lambda: ChoiceSpec("q", {"a": "", "b": None}), "description for option 'a'"),
    ],
)
def test_invalid_questions_are_rejected_before_any_request(build: object, message: str) -> None:
    with pytest.raises(QuestionError, match=message):
        build()  # type: ignore[operator]


def test_a_bare_string_is_not_a_rubric() -> None:
    """`ScoreSpec("q", "abc")` would otherwise be read as three one-character levels."""
    with pytest.raises(QuestionError):
        ScoreSpec("q", "abc")
