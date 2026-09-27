"""Tests for `Cascade`: the cheap model first, the expensive one only when it has to."""

from __future__ import annotations

import enum
import logging

import pytest

import gut
from gut import BackendError, Cascade, FakeBackend, PolicyError
from gut._questions import ChoiceSpec, NoulSpec, ScoreSpec


class Team(enum.Enum):
    BILLING = "invoices"
    PLATFORM = "outages"
    OTHER = "anything else"


def small(**answers: float | str) -> FakeBackend:
    return FakeBackend(answers=dict(answers), model="small-1")


def big(**answers: float | str) -> FakeBackend:
    return FakeBackend(answers=dict(answers), model="big-1")


def test_a_decisive_answer_never_reaches_the_second_model() -> None:
    first, second = FakeBackend(default=0.95, model="small-1"), FakeBackend(default=0.5)
    cascade = Cascade(first, second)
    response = cascade.ask("t", {"q": NoulSpec("is spam")})
    assert response.answers["q"] == gut.NoulAnswer(p=0.95)
    assert response.model_for("q") == "small-1"
    assert second.call_count == 0


def test_only_the_unsettled_questions_are_escalated() -> None:
    first = FakeBackend(answers={"is spam": 0.03, "is a bug": 0.55}, model="small-1")
    second = FakeBackend(answers={"is a bug": 0.9}, model="big-1")
    cascade = Cascade(first, second)

    response = cascade.ask("t", {"spam": NoulSpec("is spam"), "bug": NoulSpec("is a bug")})

    assert second.calls[0].names == ("bug",)
    assert response.models == {"spam": "small-1", "bug": "big-1"}
    assert list(response.answers) == ["spam", "bug"]
    assert cascade.answered_by == {"small-1": 1, "big-1": 1}


def test_the_band_edges_count_as_unsure() -> None:
    cascade = Cascade(small(), big(), unsure_band=(0.2, 0.8))
    assert not cascade.settled(gut.NoulAnswer(p=0.2))
    assert not cascade.settled(gut.NoulAnswer(p=0.8))
    assert cascade.settled(gut.NoulAnswer(p=0.19))
    assert cascade.settled(gut.NoulAnswer(p=0.81))


def test_choices_and_ratings_escalate_on_confidence() -> None:
    # FakeBackend gives a named option 0.9 confidence and an integer level full confidence.
    first = FakeBackend(answers={"which team": "PLATFORM", "how urgent": 1.5}, model="small-1")
    second = FakeBackend(answers={"how urgent": 2}, model="big-1")
    cascade = Cascade(first, second, min_confidence=0.7)
    response = cascade.ask(
        "t",
        {
            "team": ChoiceSpec("which team", {m.name: m.value for m in Team}),
            "urgency": ScoreSpec("how urgent", ["later", "soon", "now"]),
        },
    )
    assert response.models == {"team": "small-1", "urgency": "big-1"}


def test_the_last_model_is_believed_whatever_it_says() -> None:
    cascade = Cascade(FakeBackend(default=0.5), FakeBackend(default=0.5, model="big-1"))
    response = cascade.ask("t", {"q": NoulSpec("is spam")})
    assert response.answers["q"] == gut.NoulAnswer(p=0.5)
    assert response.model_for("q") == "big-1"


def test_three_stages_escalate_step_by_step() -> None:
    stages = [
        FakeBackend(default=0.5, model="tiny"),
        FakeBackend(default=0.6, model="small"),
        FakeBackend(default=0.99, model="large"),
    ]
    cascade = Cascade(*stages)
    assert cascade.ask("t", {"q": NoulSpec("q")}).model_for("q") == "large"
    assert [stage.call_count for stage in stages] == [1, 1, 1]


def test_a_stage_that_fails_hands_its_questions_on(caplog: pytest.LogCaptureFixture) -> None:
    broken = FakeBackend(model="local")  # no fixtures: every question raises BackendError
    cascade = Cascade(broken, big(**{"is spam": 0.9}))
    with caplog.at_level(logging.INFO, logger="gut"):
        response = cascade.ask("t", {"q": NoulSpec("is spam")})
    assert response.model_for("q") == "big-1"
    assert "local failed, escalating" in caplog.text


def test_the_last_stage_failing_is_raised() -> None:
    cascade = Cascade(FakeBackend(default=0.5), FakeBackend(model="big-1"))
    with pytest.raises(BackendError, match="no fixture"):
        cascade.ask("t", {"q": NoulSpec("is spam")})


def test_a_last_stage_that_skips_a_question_is_an_error() -> None:
    class Forgetful(FakeBackend):
        def ask(self, state, questions):  # type: ignore[no-untyped-def]
            response = super().ask(state, questions)
            return gut.BackendResponse(answers={}, model=response.model)

    cascade = Cascade(FakeBackend(default=0.5), Forgetful(default=0.9))
    with pytest.raises(BackendError, match="No stage of the cascade answered 'q'"):
        cascade.ask("t", {"q": NoulSpec("is spam")})


def test_tokens_are_added_up_across_stages() -> None:
    class Billed(FakeBackend):
        def ask(self, state, questions):  # type: ignore[no-untyped-def]
            response = super().ask(state, questions)
            return gut.BackendResponse(
                answers=response.answers, model=response.model, input_tokens=10
            )

    cascade = Cascade(Billed(default=0.5), Billed(default=0.9))
    assert cascade.ask("t", {"q": NoulSpec("q")}).input_tokens == 20
    assert Cascade(small(q=0.99), big()).ask("t", {"q": NoulSpec("q")}).input_tokens is None


def test_decisions_name_the_model_that_actually_answered() -> None:
    first = FakeBackend(answers={"is spam": 0.02, "is a bug": 0.5}, model="small-1")
    second = FakeBackend(answers={"is a bug": 0.97}, model="big-1")
    gut.configure(backend=Cascade(first, second), cache=gut.NullCache())

    with gut.judge("a ticket") as j:
        spam = j.likely("is spam")
        bug = j.likely("is a bug")
    assert spam == gut.NO
    assert spam.model == "small-1"
    assert bug == gut.YES
    assert bug.model == "big-1"

    single = gut.likely("a ticket", "is a bug")
    assert single.model == "big-1"


def test_the_cache_key_changes_with_the_thresholds() -> None:
    a = Cascade(small(), big())
    b = Cascade(small(), big(), unsure_band=(0.1, 0.9))
    assert a.model_id == "cascade(small-1 > big-1; band=0.2-0.8; min_confidence=0.7)"
    assert a.model_id != b.model_id


@pytest.mark.parametrize(
    ("backends", "options", "message"),
    [
        (1, {}, "at least two backends"),
        (2, {"unsure_band": (0.8, 0.2)}, "unsure_band must be"),
        (2, {"unsure_band": (-0.1, 0.5)}, "unsure_band must be"),
        (2, {"min_confidence": 1.5}, "min_confidence must be"),
    ],
)
def test_bad_arguments_are_refused(backends: int, options: dict[str, object], message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        Cascade(*[small() for _ in range(backends)], **options)  # type: ignore[arg-type]


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(BackendError, match="at least one question"):
        Cascade(small(), big()).ask("t", {})


def test_a_cascade_is_a_backend() -> None:
    assert isinstance(Cascade(small(), big()), gut.Backend)
