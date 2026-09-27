"""Tests for the pure parts of the local backends: everything but the forward pass.

The forward pass itself needs PyTorch and a downloaded model; `test_local_models.py` covers it
wherever the `local` extra is installed.
"""

from __future__ import annotations

import math

import pytest

import gut
from gut import BackendError, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._backends.local import (
    answer_from_entailment,
    as_hypothesis,
    hypotheses_for,
    nli_labels,
)
from gut._questions import ChoiceSpec, NoulSpec, ScoreSpec


@pytest.mark.parametrize(
    ("question", "hypothesis"),
    [
        ("is a bug report", "This text is a bug report."),
        ("mentions a safety problem.", "This text mentions a safety problem."),
        ("has reproduction steps", "This text has reproduction steps."),
        ("the customer threatens to cancel", "the customer threatens to cancel"),
        ("Is this spam?", "Is this spam?"),
        ("  ", ""),
    ],
)
def test_a_bare_predicate_gets_a_subject(question: str, hypothesis: str) -> None:
    assert as_hypothesis(question) == hypothesis


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ({0: "entailment", 1: "not_entailment"}, (0, 1)),
        ({0: "contradiction", 1: "neutral", 2: "entailment"}, (2, 0)),
        ({0: "ENTAILMENT", 1: "Not Entailment"}, (0, 1)),
    ],
)
def test_the_entailment_head_is_found_by_name(
    labels: dict[int, str], expected: tuple[int, int]
) -> None:
    assert nli_labels(labels) == expected


def test_a_model_that_is_not_nli_is_refused() -> None:
    with pytest.raises(BackendError, match="natural-language-inference model"):
        nli_labels({0: "POSITIVE", 1: "NEGATIVE"})


def test_hypotheses_per_question_shape() -> None:
    template = "This text is about {}."
    assert hypotheses_for(NoulSpec("is spam"), template) == ["This text is spam."]
    assert hypotheses_for(NoulSpec("is urgent", yes_means="someone is blocked"), template) == [
        "someone is blocked"
    ]
    choice = ChoiceSpec(None, {"BILLING": "invoices and refunds", "BUG_REPORT": None})
    assert hypotheses_for(choice, template) == [
        "This text is about invoices and refunds.",
        "This text is about bug report.",
    ]
    score = ScoreSpec(None, ["no one is blocked", "is blocking a team"])
    assert hypotheses_for(score, template) == ["no one is blocked", "This text is blocking a team."]


def test_yes_no_is_the_sigmoid_of_the_log_odds() -> None:
    answer = answer_from_entailment(NoulSpec("q"), [math.log(3)])
    assert isinstance(answer, NoulAnswer)
    assert answer.p == pytest.approx(0.75)


def test_options_compete_through_one_softmax() -> None:
    spec = ChoiceSpec(None, {"A": None, "B": None, "C": None})
    answer = answer_from_entailment(spec, [2.0, 0.0, 0.0])
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "A"
    total = math.exp(2) + 2
    assert answer.probabilities == pytest.approx(
        {"A": math.exp(2) / total, "B": 1 / total, "C": 1 / total}
    )
    assert answer.confidence == pytest.approx(math.exp(2) / total)


def test_levels_compete_and_the_score_is_their_mean() -> None:
    spec = ScoreSpec(None, ["low", "mid", "high"])
    answer = answer_from_entailment(spec, [0.0, 0.0, 1000.0])
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == pytest.approx(2.0)
    assert answer.confidence == pytest.approx(1.0)
    assert answer.legend == {0: "low", 1: "mid", 2: "high"}


@pytest.mark.parametrize("name", ["TransformersBackend", "ZeroShotBackend"])
def test_a_missing_extra_says_how_to_install_it(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def without_torch(module: str, *args: object, **kwargs: object) -> object:
        if module in {"torch", "transformers"}:
            raise ImportError(f"No module named {module!r}")
        return real_import(module, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", without_torch)
    with pytest.raises(BackendError, match=r'pip install "gut\[local\]"'):
        getattr(gut, name)()
