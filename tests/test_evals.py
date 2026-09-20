"""Tests for predicate example files: parsing, scoring, and what counts as correct."""

from __future__ import annotations

from pathlib import Path

import pytest

import gut
from gut._backends.fake import FakeBackend
from gut._errors import EvalError
from gut._evals import (
    DEFAULT_MIN_ACCURACY,
    format_failures,
    load_suite,
    looks_like_suite,
    run_suite,
)
from gut._questions import ChoiceSpec, NoulSpec, ScoreSpec

NOUL_FILE = """
question: "the customer threatens to cancel"
min_accuracy: 0.5
examples:
  - text: "I'm cancelling."
    expected: yes
  - text: "How do I cancel?"
    expected: no
"""

CHOICE_FILE = """
question: "which team owns this"
options:
  BILLING: "invoices and charges"
  PLATFORM: "outages and errors"
examples:
  - text: "charged twice"
    expected: BILLING
"""

SCORE_FILE = """
question: "how urgent"
levels: ["can wait", "this week", "today"]
tolerance: 0.6
examples:
  - text: "everything is down"
    expected: 2
"""


@pytest.fixture(autouse=True)
def _no_cache() -> None:
    """These tests score the same question twice with different backends on purpose."""
    gut.configure(cache=gut.NullCache())


def write(tmp_path: Path, body: str, name: str = "suite.yaml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- parsing


def test_a_yes_no_file_is_the_default_shape(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, NOUL_FILE))
    assert suite.kind == "noul"
    assert isinstance(suite.spec, NoulSpec)
    assert suite.spec.instructions == "the customer threatens to cancel"
    assert [example.expected for example in suite.examples] == [True, False]
    assert suite.min_accuracy == 0.5
    assert suite.name == "suite"


def test_options_make_it_a_choice_file(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, CHOICE_FILE))
    assert suite.kind == "choice"
    assert isinstance(suite.spec, ChoiceSpec)
    assert suite.spec.criteria == {
        "BILLING": "invoices and charges",
        "PLATFORM": "outages and errors",
    }


def test_levels_make_it_a_score_file(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, SCORE_FILE))
    assert suite.kind == "score"
    assert isinstance(suite.spec, ScoreSpec)
    assert suite.tolerance == 0.6


def test_the_kind_can_be_stated_outright(tmp_path: Path) -> None:
    body = 'type: noul\nquestion: "q"\nexamples:\n  - text: "t"\n    expected: yes\n'
    assert load_suite(write(tmp_path, body)).kind == "noul"


@pytest.mark.parametrize(
    ("declared", "extra"),
    [
        ("choice", "options:\n  A: first\n  B: second\n"),
        ("score", "levels: [low, high]\n"),
    ],
)
def test_a_stated_kind_overrides_the_inference(tmp_path: Path, declared: str, extra: str) -> None:
    body = f"type: {declared}\n{extra}examples:\n  - text: t\n    expected: A\n"
    if declared == "score":
        body = body.replace("expected: A", "expected: 1")
    assert load_suite(write(tmp_path, body)).kind == declared


@pytest.mark.parametrize(
    ("body", "missing"),
    [
        ("type: choice\nexamples:\n  - text: t\n    expected: A\n", "options"),
        ("type: score\nexamples:\n  - text: t\n    expected: 1\n", "levels"),
    ],
)
def test_a_stated_kind_still_needs_its_own_keys(tmp_path: Path, body: str, missing: str) -> None:
    with pytest.raises(EvalError, match=f"missing required key '{missing}'"):
        load_suite(write(tmp_path, body))


def test_criteria_descriptions_are_optional(tmp_path: Path) -> None:
    body = (
        'question: "is this spam"\nyes_means: "unsolicited advertising"\n'
        'examples:\n  - text: "buy now"\n    expected: yes\n'
    )
    suite = load_suite(write(tmp_path, body))
    assert suite.spec.canonical()["criteria"] == {"true": "unsolicited advertising"}


def test_a_structured_subject_is_allowed(tmp_path: Path) -> None:
    body = (
        'question: "is a bug report"\nexamples:\n'
        '  - state: {subject: "down", body: "502"}\n    expected: yes\n'
    )
    suite = load_suite(write(tmp_path, body))
    assert suite.examples[0].state == {"subject": "down", "body": "502"}


def test_min_accuracy_defaults_to_everything(tmp_path: Path) -> None:
    """A threshold that silently tolerates failures is worse than one you had to lower."""
    body = 'question: "q"\nexamples:\n  - text: "t"\n    expected: yes\n'
    assert load_suite(write(tmp_path, body)).min_accuracy == DEFAULT_MIN_ACCURACY == 1.0


def test_looks_like_suite_only_accepts_documents_with_examples() -> None:
    assert looks_like_suite({"examples": []}) is True
    assert looks_like_suite({"services": {"web": {}}}) is False
    assert looks_like_suite(["a", "list"]) is False
    assert looks_like_suite(None) is False


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("question: q\n", "no 'examples'"),
        ("examples: []\nquestion: q\n", "examples is empty"),
        ("examples: 5\nquestion: q\n", "examples must be a list"),
        ("examples:\n  - 5\nquestion: q\n", "must be a mapping"),
        ("examples:\n  - expected: yes\nquestion: q\n", "needs a 'text' or 'state' key"),
        ("examples:\n  - text: t\nquestion: q\n", "needs an 'expected' key"),
        ("examples:\n  - text: t\n    expected: 5\nquestion: q\n", "must be yes or no"),
        ("examples:\n  - text: t\n    expected: yes\n", "needs a 'question'"),
        ("type: wat\nquestion: q\nexamples:\n  - text: t\n    expected: yes\n", "type must be"),
        ("question: 5\nexamples:\n  - text: t\n    expected: yes\n", "question must be a string"),
        (
            "options: [a, b]\nexamples:\n  - text: t\n    expected: A\n",
            "options must be a mapping",
        ),
        (
            "options:\n  A: first\n  B: second\nexamples:\n  - text: t\n    expected: 5\n",
            "must be an option name",
        ),
        ("levels: nope\nexamples:\n  - text: t\n    expected: 1\n", "levels must be a list"),
        (
            "levels: [a]\nexamples:\n  - text: t\n    expected: 1\n",
            "between 2 and 10 levels",
        ),
        (
            "question: q\nmin_accuracy: high\nexamples:\n  - text: t\n    expected: yes\n",
            "min_accuracy must be a number",
        ),
        (
            "levels: [a, b]\nexamples:\n  - text: t\n    expected: 1\n    tolerance: loose\n",
            "tolerance must be a number",
        ),
        ("options:\n  A: first\n", "no 'examples'"),
        ("[1, 2, 3]\n", "no 'examples'"),
    ],
)
def test_a_malformed_file_says_what_is_wrong(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(EvalError, match=message):
        load_suite(write(tmp_path, body))


def test_unparseable_yaml_is_reported_as_such(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="not valid YAML"):
        load_suite(write(tmp_path, "question: [unclosed\n"))


# --------------------------------------------------------------------------- scoring


def test_a_yes_no_file_scores_against_its_threshold(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, NOUL_FILE))
    backend = FakeBackend(
        answers={"I'm cancelling.": 0.9, "How do I cancel?": 0.2},
        rule=lambda state, name, spec: {"I'm cancelling.": 0.9}.get(str(state), 0.2),
    )
    result = run_suite(suite, backend)

    assert result.accuracy == 1.0
    assert result.correct == 2
    assert result.passed is True
    assert result.failures == ()
    assert "p=0.900" in result.results[0].actual


def test_the_threshold_can_be_moved(tmp_path: Path) -> None:
    body = NOUL_FILE + "threshold: 0.95\n"
    suite = load_suite(write(tmp_path, body))
    assert suite.threshold == 0.95

    # Both answers now fall below the bar, so the "yes" example is wrong.
    result = run_suite(suite, FakeBackend(default=0.9))
    assert result.correct == 1


def test_a_choice_file_scores_on_the_chosen_option(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, CHOICE_FILE))
    assert run_suite(suite, FakeBackend(default="BILLING")).passed is True
    assert run_suite(suite, FakeBackend(default="PLATFORM")).passed is False


def test_a_score_file_scores_within_tolerance(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, SCORE_FILE))
    assert run_suite(suite, FakeBackend(default=1.5)).passed is True  # |1.5 - 2| <= 0.6
    assert run_suite(suite, FakeBackend(default=1.0)).passed is False  # |1.0 - 2| > 0.6


def test_an_example_can_set_its_own_tolerance(tmp_path: Path) -> None:
    body = SCORE_FILE.replace(
        '  - text: "everything is down"\n    expected: 2\n',
        '  - text: "everything is down"\n    expected: 2\n    tolerance: 1.5\n',
    )
    suite = load_suite(write(tmp_path, body))
    assert suite.examples[0].tolerance == 1.5
    assert run_suite(suite, FakeBackend(default=1.0)).passed is True


def test_accuracy_is_compared_against_the_files_own_bar(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, NOUL_FILE))
    result = run_suite(suite, FakeBackend(default=0.9))  # gets the "no" example wrong

    assert result.accuracy == 0.5
    assert result.passed is True  # min_accuracy is 0.5
    assert len(result.failures) == 1


def test_failures_report_what_the_model_actually_said(tmp_path: Path) -> None:
    suite = load_suite(write(tmp_path, NOUL_FILE.replace("min_accuracy: 0.5", "min_accuracy: 1.0")))
    report = format_failures(run_suite(suite, FakeBackend(default=0.9)))

    assert "accuracy 50%" in report
    assert "below min_accuracy 100%" in report
    assert "the customer threatens to cancel" in report
    assert "expected False, got yes (p=0.900)" in report
    assert "How do I cancel?" in report


def test_a_long_subject_is_trimmed_in_reports(tmp_path: Path) -> None:
    body = 'question: "q"\nexamples:\n  - text: "' + "x" * 200 + '"\n    expected: no\n'
    suite = load_suite(write(tmp_path, body))
    report = format_failures(run_suite(suite, FakeBackend(default=0.9)))
    assert "..." in report
    assert "x" * 200 not in report


def test_a_structured_subject_is_labelled_in_reports(tmp_path: Path) -> None:
    body = 'question: "q"\nexamples:\n  - state: {subject: "down"}\n    expected: no\n'
    suite = load_suite(write(tmp_path, body))
    assert "subject" in format_failures(run_suite(suite, FakeBackend(default=0.9)))


def test_answers_are_reported_with_their_confidence(tmp_path: Path) -> None:
    choice = run_suite(load_suite(write(tmp_path, CHOICE_FILE)), FakeBackend(default="BILLING"))
    assert "confidence=0.90" in choice.results[0].actual

    score = run_suite(load_suite(write(tmp_path, SCORE_FILE, "s.yaml")), FakeBackend(default=1.5))
    assert "tolerance 0.6" in score.results[0].actual
