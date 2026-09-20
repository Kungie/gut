"""The backend protocol and the raw answer shapes every backend returns.

These types are `gut`'s own, deliberately not the vendor's. A backend's job is narrow: take one
state and a batch of named questions, return one typed answer per name plus the exact model version
that produced them. Everything above this line -- cost rules, caching, batching, logging -- is
backend-agnostic, which is what keeps a future local-model or logprob backend a drop-in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from gut._questions import QuestionSpec, State


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    """A yes/no answer. Carries no confidence: the probability is the whole answer."""

    p: float
    """Probability that the answer is yes, from 0 to 1."""


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """A selected option and the distribution it came from."""

    choice: str
    """The option with the highest probability."""
    confidence: float
    """How concentrated `probabilities` is -- a spread statistic, not an accuracy estimate."""
    probabilities: Mapping[str, float]
    """Probability of every option, summing to approximately 1."""


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """A rating against an ordered rubric."""

    score: float
    """Probability-weighted mean of the levels; may land between two of them."""
    confidence: float
    """How concentrated `probabilities` is -- a spread statistic, not an accuracy estimate."""
    probabilities: Mapping[int, float]
    """Probability of each level, keyed by level number."""
    legend: Mapping[int, str]
    """The rubric that was asked, keyed by level number."""


Answer: TypeAlias = NoulAnswer | ChoiceAnswer | ScoreAnswer
"""Any answer a backend can return."""


@dataclass(frozen=True, slots=True)
class BackendResponse:
    """One backend call's worth of answers."""

    answers: Mapping[str, Answer]
    """Answers keyed by the question names that were asked."""
    model: str
    """The exact versioned model that answered, never an alias."""
    input_tokens: int | None = None
    """Billable input tokens, when the backend reports them."""


@runtime_checkable
class Backend(Protocol):
    """What `gut` needs from anything that can answer questions.

    One call answers many questions about a single state, because that is the shape that makes
    batching worth doing: billing is on input, so the state is paid for once however many questions
    ride along with it.
    """

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """Answer `questions` about `state`.

        Args:
            state: The content every question refers to.
            questions: A nonempty mapping of caller-chosen names to question specs.

        Returns:
            One answer per name in `questions`, plus the model that produced them.
        """
        ...
