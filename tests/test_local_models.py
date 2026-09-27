"""The local backends, end to end, against real models.

Skipped unless the `local` extra is installed: these download two small models (about 1.7 GB
together) the first time they run. They check plumbing, not quality -- that an obvious case comes
out the obvious way, and that sharing the subject's computation across a batch changes nothing.

    pip install "gutfeel[local]" && pytest -m local
"""

from __future__ import annotations

import enum

import pytest

import gut
from gut._questions import ChoiceSpec, NoulSpec

pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.local

SPAM = "WIN a FREE iPhone!!! Click the link below and enter your card details to claim your prize"
MEETING = "Hi Sam, can we move tomorrow's design review to 3pm? The room is booked until then."
REFUND = "The charger you sold me melted after two days. I want my money back."


class Team(enum.Enum):
    BILLING = "payments, invoices, refunds and money"
    ENGINEERING = "bugs, crashes, errors and broken software"
    OTHER = "anything else"


@pytest.fixture(scope="module")
def nli() -> gut.Backend:
    return gut.ZeroShotBackend(device="cpu")


@pytest.fixture(scope="module")
def causal() -> gut.Backend:
    return gut.TransformersBackend(dtype="float32", device="cpu")


@pytest.fixture(params=["nli", "causal"])
def backend(request: pytest.FixtureRequest) -> gut.Backend:
    chosen: gut.Backend = request.getfixturevalue(request.param)
    gut.configure(backend=chosen, cache=gut.NullCache())
    return chosen


def test_obvious_yes_and_no(backend: gut.Backend) -> None:
    assert gut.likely(SPAM, "is spam") == gut.YES
    assert gut.likely(MEETING, "is spam") == gut.NO
    assert gut.likely(REFUND, "asks for a refund") == gut.YES


def test_obvious_classification(backend: gut.Backend) -> None:
    assert gut.classify(REFUND, Team).value is Team.BILLING


def test_the_answering_model_is_pinned_to_a_commit(backend: gut.Backend) -> None:
    decision = gut.likely(MEETING, "asks to reschedule a meeting")
    assert decision.model.startswith(backend.model_id + "@")


def test_a_batch_answers_exactly_what_single_questions_do(causal: gut.Backend) -> None:
    """The shared key-value cache must be an optimisation and nothing else."""
    questions: dict[str, gut.QuestionSpec] = {
        "spam": NoulSpec("is spam"),
        "prize": NoulSpec("mentions a prize"),
        "team": ChoiceSpec(None, {member.name: member.value for member in Team}),
    }
    batched = causal.ask(SPAM, questions)
    for name, spec in questions.items():
        alone = causal.ask(SPAM, {name: spec}).answers[name]
        together = batched.answers[name]
        if isinstance(alone, gut.NoulAnswer):
            assert isinstance(together, gut.NoulAnswer)
            assert together.p == pytest.approx(alone.p, abs=1e-4)
        else:
            assert isinstance(together, gut.ChoiceAnswer)
            assert isinstance(alone, gut.ChoiceAnswer)
            assert dict(together.probabilities) == pytest.approx(
                dict(alone.probabilities), abs=1e-4
            )


def test_a_cascade_of_two_local_models(nli: gut.Backend, causal: gut.Backend) -> None:
    cascade = gut.Cascade(nli, causal)
    gut.configure(backend=cascade, cache=gut.NullCache())
    assert gut.likely(SPAM, "is spam") == gut.YES
    assert sum(cascade.answered_by.values()) == 1
