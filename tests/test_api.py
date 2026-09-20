"""Tests for likely / classify / rate: decision construction, ids, and backend contract."""

from __future__ import annotations

import contextlib
import enum
import warnings
from collections.abc import Iterator, Mapping

import pytest

import gut
from gut import FakeBackend, Outcome
from gut._backends.base import BackendResponse, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._backends.fake import FakeValue
from gut._errors import BackendError, ConfigurationError, PolicyError, QuestionError
from gut._questions import ChoiceSpec, QuestionSpec, ScoreSpec, State

URGENCY = ["can wait", "this week", "today"]


class Team(enum.Enum):
    BILLING = "questions about invoices, charges, refunds"
    PLATFORM = "outages, latency, API errors"
    OTHER = "anything else"


def _by_question_type(state: State, name: str, spec: QuestionSpec) -> FakeValue | None:
    """Answer choice and score questions generically; leave noul to the fixtures.

    Picks whatever options the question actually offers, so it works for any enum a test defines.
    """
    match spec:
        case ChoiceSpec():
            return next(iter(spec.criteria))
        case ScoreSpec():
            return len(spec.criteria) - 1
        case _:
            return None


@contextlib.contextmanager
def warnings_as_errors() -> Iterator[None]:
    """Turn any warning raised inside the block into a failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.fixture
def backend() -> FakeBackend:
    fake = FakeBackend(
        answers={
            "is a bug report": 0.91,
            "the customer threatens to cancel": 0.05,
        },
        rule=_by_question_type,
    )
    gut.configure(backend=fake)
    return fake


# --------------------------------------------------------------------------- likely


def test_likely_builds_a_decision_from_the_probability(backend: FakeBackend) -> None:
    decision = gut.likely("a ticket", "is a bug report")
    assert decision.p == 0.91
    assert decision.outcome is Outcome.YES
    assert decision.model == "fake-1.0"
    assert decision.cached is False
    assert decision.latency_ms is not None and decision.latency_ms >= 0


def test_likely_applies_the_cost_rule(backend: FakeBackend) -> None:
    """p=0.05 sits above the 2/52 threshold that these costs imply, so it escalates."""
    decision = gut.likely(
        "an email",
        "the customer threatens to cancel",
        cost_false_yes=2,
        cost_false_no=50,
    )
    assert decision.outcome is Outcome.YES
    assert decision.policy.implied_threshold == pytest.approx(2 / 52)

    # Symmetric costs, and the same probability reads the other way.
    assert gut.likely("an email", "the customer threatens to cancel").outcome is Outcome.NO


def test_likely_reaches_unsure_when_a_human_is_cheap_enough(backend: FakeBackend) -> None:
    decision = gut.likely(
        "an email",
        "the customer threatens to cancel",
        cost_false_yes=2,
        cost_false_no=50,
        cost_human=0.5,
    )
    assert decision.outcome is Outcome.UNSURE


def test_likely_passes_criteria_through_to_the_question(backend: FakeBackend) -> None:
    gut.likely("a ticket", "is a bug report", yes_means="a defect report", no_means="a question")
    spec = backend.calls[-1].specs["q"]
    assert spec.canonical()["criteria"] == {"true": "a defect report", "false": "a question"}


def test_likely_rejects_contradictory_costs(backend: FakeBackend) -> None:
    with pytest.raises(PolicyError, match="not both"):
        gut.likely("a ticket", "is a bug report", cost_false_yes=1, cost_false_no=1, threshold=0.5)


def test_likely_rejects_an_empty_question(backend: FakeBackend) -> None:
    with pytest.raises(QuestionError, match="non-empty string"):
        gut.likely("a ticket", "   ")


# --------------------------------------------------------------------------- decision ids


def test_the_same_question_from_the_same_place_keeps_its_id(backend: FakeBackend) -> None:
    def site() -> str:
        return gut.likely("a ticket", "is a bug report").id

    assert site() == site()


def test_a_different_question_gets_a_different_id(backend: FakeBackend) -> None:
    first = gut.likely("a ticket", "is a bug report").id
    second = gut.likely("a ticket", "the customer threatens to cancel").id
    assert first != second


def test_the_same_question_from_a_different_place_gets_a_different_id(
    backend: FakeBackend,
) -> None:
    def here() -> str:
        return gut.likely("a ticket", "is a bug report").id

    def there() -> str:
        return gut.likely("a ticket", "is a bug report").id

    assert here() != there()


def test_the_id_does_not_depend_on_the_subject(backend: FakeBackend) -> None:
    def site(subject: str) -> str:
        return gut.likely(subject, "is a bug report").id

    assert site("one ticket") == site("a completely different ticket")


# --------------------------------------------------------------------------- classify


def test_classify_maps_the_label_back_to_an_enum_member(backend: FakeBackend) -> None:
    decision = gut.classify("an outage report", Team)
    assert decision.value is Team.BILLING
    assert decision.outcome is Outcome.YES
    assert set(decision.probabilities) == set(Team)
    assert sum(decision.probabilities.values()) == pytest.approx(1.0)


def test_classify_sends_member_values_as_the_descriptions(backend: FakeBackend) -> None:
    gut.classify("a ticket", Team)
    criteria = backend.calls[-1].specs["q"].canonical()["criteria"]
    assert criteria == {member.name: member.value for member in Team}


def test_classify_goes_unsure_below_min_confidence(backend: FakeBackend) -> None:
    decision = gut.classify("a ticket", Team, min_confidence=0.95)
    assert decision.outcome is Outcome.UNSURE
    assert decision.value is Outcome.UNSURE
    assert decision.confidence == 0.9
    # The distribution is still there to inspect; only the verdict is withheld.
    assert decision.probabilities[Team.BILLING] == 0.9


def test_classify_passes_an_optional_question(backend: FakeBackend) -> None:
    gut.classify("a ticket", Team, question="Which team owns this?")
    assert backend.calls[-1].specs["q"].instructions == "Which team owns this?"
    gut.classify("a ticket", Team)
    assert backend.calls[-1].specs["q"].instructions is None


def test_classify_warns_when_nothing_can_be_none_of_these(backend: FakeBackend) -> None:
    class Sentiment(enum.Enum):
        POSITIVE = "pleased or complimentary"
        NEGATIVE = "annoyed or complaining"

    with pytest.warns(UserWarning, match="no catch-all member"):
        gut.classify("a ticket", Sentiment)


def test_classify_warns_only_once_per_enum(backend: FakeBackend) -> None:
    class Tone(enum.Enum):
        WARM = "friendly"
        COLD = "curt"

    with pytest.warns(UserWarning, match="no catch-all member"):
        gut.classify("a ticket", Tone)
    with warnings_as_errors():
        gut.classify("a ticket", Tone)


def test_classify_stays_quiet_with_a_catch_all(backend: FakeBackend) -> None:
    with warnings_as_errors():
        gut.classify("a ticket", Team)


def test_classify_rejects_enums_whose_values_are_not_descriptions(
    backend: FakeBackend,
) -> None:
    class Numbered(enum.Enum):
        A = 1
        B = 2

    with pytest.raises(QuestionError, match="non-empty string value"):
        gut.classify("a ticket", Numbered)


def test_classify_rejects_an_empty_enum(backend: FakeBackend) -> None:
    Empty = enum.Enum("Empty", [])  # type: ignore[misc]  # noqa: N806
    with pytest.raises(QuestionError, match="no members"):
        gut.classify("a ticket", Empty)


# --------------------------------------------------------------------------- rate


def test_rate_builds_a_score_decision(backend: FakeBackend) -> None:
    decision = gut.rate("a ticket", URGENCY)
    assert decision.score == 2.0  # the fixture rule answers with the top level
    assert decision.levels == tuple(URGENCY)
    assert decision.outcome is Outcome.YES
    assert decision.nearest_level == 2


def test_rate_goes_unsure_below_min_confidence(backend: FakeBackend) -> None:
    assert gut.rate("a ticket", URGENCY, min_confidence=1.1).outcome is Outcome.UNSURE


def test_rate_enforces_the_level_limits(backend: FakeBackend) -> None:
    with pytest.raises(QuestionError, match="between 2 and 10 levels"):
        gut.rate("a ticket", ["only one"])


# --------------------------------------------------------------------------- backend contract


def test_the_backend_can_be_overridden_per_call(backend: FakeBackend) -> None:
    other = FakeBackend(answers={"is a bug report": 0.0}, model="other-1.0")
    decision = gut.likely("a ticket", "is a bug report", backend=other)
    assert decision.model == "other-1.0"
    assert decision.p == 0.0
    assert backend.call_count == 0


def test_no_backend_configured_says_what_to_do() -> None:
    with pytest.raises(ConfigurationError, match="No backend is configured"):
        gut.likely("a ticket", "is a bug report")


class WrongShapeBackend:
    """Returns an answer of the wrong kind, as a misbehaving backend would."""

    model_id = "wrong-1.0"

    def __init__(self, answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> None:
        self.answer = answer

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        return BackendResponse(answers=dict.fromkeys(questions, self.answer), model="wrong-1.0")


class EmptyBackend:
    """Returns no answers at all."""

    model_id = "empty-1.0"

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        return BackendResponse(answers={}, model="empty-1.0")


def test_a_backend_answering_the_wrong_shape_is_caught() -> None:
    score = ScoreAnswer(score=1.0, confidence=1.0, probabilities={1: 1.0}, legend={1: "x"})
    with pytest.raises(BackendError, match="Expected a NoulAnswer"):
        gut.likely("a ticket", "is a bug report", backend=WrongShapeBackend(score))

    noul = NoulAnswer(p=0.5)
    with pytest.raises(BackendError, match="Expected a ChoiceAnswer"):
        gut.classify("a ticket", Team, backend=WrongShapeBackend(noul))

    with pytest.raises(BackendError, match="Expected a ScoreAnswer"):
        gut.rate("a ticket", URGENCY, backend=WrongShapeBackend(noul))


def test_a_backend_answering_nothing_is_caught() -> None:
    with pytest.raises(BackendError, match="returned no answer"):
        gut.likely("a ticket", "is a bug report", backend=EmptyBackend())


def test_a_backend_inventing_a_label_is_caught() -> None:
    bogus = ChoiceAnswer(choice="SHIPPING", confidence=0.9, probabilities={"SHIPPING": 0.9})
    with pytest.raises(BackendError, match="not a member of Team"):
        gut.classify("a ticket", Team, backend=WrongShapeBackend(bogus))
