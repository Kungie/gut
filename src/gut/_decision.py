"""Decision objects -- what `likely`, `classify` and `rate` hand back.

A `Decision` is not a bool with extra fields. It is a three-valued answer that *usually* collapses
to a bool, and the point of the type is to make the third value impossible to ignore by accident:

- `match d: case gut.YES / gut.NO / gut.UNSURE` handles all three explicitly.
- `if d:` works, and what it does with UNSURE is a configured decision, not a silent default.

Every decision also carries what it took to produce it -- the probability, the exact model version
that answered, the policy applied, whether it came from cache -- because a judgment you cannot audit
later is a judgment you cannot calibrate.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Literal, TypeAlias, TypeVar

from gut._config import current_on_unsure, current_sink
from gut._errors import UnsureDecision
from gut._log import ResolutionRecord, _jsonable, _now, emit_resolution
from gut._outcomes import Outcome
from gut._rule import DEFAULT_POLICY, Policy

DecisionSource: TypeAlias = Literal["backend", "cache", "prefetch"]
"""Where an answer came from: its own request, the cache, or a batch fetched ahead of it."""

E = TypeVar("E", bound=Enum)


@dataclass(frozen=True, eq=False, kw_only=True)
class BaseDecision:
    """Fields and behaviour common to every kind of decision."""

    outcome: Outcome
    """What the decision resolved to."""
    id: str
    """Stable decision-site identifier: the same question asked from the same place keeps this id
    across runs and across releases, so decisions can be tracked and later calibrated."""
    model: str
    """The exact versioned model that answered, never an alias. Aliases move; recorded answers
    should not silently change meaning underneath a stored id."""
    source: DecisionSource = "backend"
    """How the answer was obtained.

    `"prefetch"` means it rode along in a batch fetched by `@semantic` or `judge()` -- distinct from
    `"cache"`, because it still cost a request, just a shared one.
    """
    latency_ms: float | None = None
    """How long this decision's own backend call took, or `None` when it did not make one."""

    @property
    def cached(self) -> bool:
        """Whether this decision avoided a request of its own."""
        return self.source != "backend"

    def __bool__(self) -> bool:
        """Collapse to a bool, consulting the UNSURE policy when there is no honest answer.

        Raises:
            UnsureDecision: The decision is UNSURE and the policy is `"raise"` (the default).
        """
        if self.outcome is Outcome.YES:
            return True
        if self.outcome is Outcome.NO:
            return False

        policy = current_on_unsure()
        if callable(policy):
            return bool(policy(self))
        if policy == "raise":
            raise UnsureDecision(self)
        return policy == "true"

    def __eq__(self, other: object) -> bool:
        """Compare against an `Outcome`, or against another decision of the same kind.

        Comparing equal to an outcome is what makes `match d: case gut.YES:` work, since a value
        pattern is an `==` test. It does mean equality is not transitive across decisions -- two
        different decisions can both equal `gut.YES` without equalling each other. That is the
        documented trade for `match` reading the way it should. See D10 in DECISIONS.md.
        """
        if isinstance(other, Outcome):
            return self.outcome is other
        if other.__class__ is self.__class__:
            return self._identity() == other._identity()
        return NotImplemented

    def __hash__(self) -> int:
        """Hash on the outcome alone, so that `d == gut.YES` implies `hash(d) == hash(gut.YES)`."""
        return hash(self.outcome)

    def resolve(self, actual: object, *, note: str | None = None) -> ResolutionRecord:
        """Record what actually happened, under this decision's id.

        Nothing consumes resolutions yet. They exist so that the question "is this model calibrated
        on *my* data?" is answerable later from logs already being written, rather than from an
        instrumentation project started after the fact.

        Args:
            actual: Ground truth, however you express it -- a bool, an enum member, a level number.
            note: Anything worth keeping alongside it.

        Returns:
            The record that was emitted, whether or not a sink kept it.
        """
        record = ResolutionRecord(
            id=self.id,
            timestamp=_now(),
            actual=_jsonable(actual),
            decided=self.outcome.value,
            note=note,
        )
        emit_resolution(record, current_sink())
        return record

    def _identity(self) -> tuple[object, ...]:
        return tuple(getattr(self, f.name) for f in dataclasses.fields(self))


@dataclass(frozen=True, eq=False, kw_only=True)
class Decision(BaseDecision):
    """A yes/no judgment, backed by Jev's noul primitive.

    Example:
        ```python
        d = gut.likely(email, "the customer threatens to cancel",
                       cost_false_yes=2, cost_false_no=50, cost_human=1)
        match d:
            case gut.YES:    escalate()
            case gut.NO:     auto_reply()
            case gut.UNSURE: ask_human()
        ```
    """

    p: float
    """Probability that the answer is yes, and the number the cost rule was applied to.

    When a calibrator is configured this is the corrected value; `raw_p` is what the model said.
    """
    raw_p: float | None = None
    """What the model said before correction, or `None` when nothing corrected it."""
    policy: Policy = field(default_factory=lambda: DEFAULT_POLICY)
    """The cost policy that turned `p` into `outcome`."""


@dataclass(frozen=True, eq=False, kw_only=True)
class ChoiceDecision(BaseDecision, Generic[E]):
    """A categorical judgment over the members of an `Enum`, backed by the choice primitive.

    `outcome` is `YES` when a member was selected and `UNSURE` when `confidence` fell below
    `min_confidence`; there is no meaningful `NO` for a classification.
    """

    value: E | Literal[Outcome.UNSURE]
    """The selected enum member, or `UNSURE` when confidence fell below `min_confidence`."""
    probabilities: Mapping[E, float]
    """Probability of every member, summing to approximately 1."""
    confidence: float
    """How concentrated `probabilities` is, from 0 to 1.

    Uncorrected, this is a spread statistic computed from the distribution the answer already
    gives, **not** a probability of being correct. A fitted calibrator turns it into one, for the
    task it was fitted on; `raw_confidence` then holds what the model said.
    """
    raw_confidence: float | None = None
    """What the model said before correction, or `None` when nothing corrected it."""
    min_confidence: float | None = None
    """The confidence floor applied, if any."""


@dataclass(frozen=True, eq=False, kw_only=True)
class ScoreDecision(BaseDecision):
    """An ordinal judgment against a rubric, backed by the score primitive.

    `outcome` is `YES` when a score was accepted and `UNSURE` when `confidence` fell below
    `min_confidence`.
    """

    score: float
    """Probability-weighted mean of the rubric levels; may land between two levels."""
    probabilities: Mapping[int, float]
    """Probability of each level, keyed by level number."""
    confidence: float
    """How concentrated `probabilities` is, from 0 to 1 -- see `ChoiceDecision.confidence`."""
    raw_confidence: float | None = None
    """What the model said before correction, or `None` when nothing corrected it."""
    levels: tuple[str, ...] = ()
    """The rubric that was asked, in order, so a stored score stays interpretable."""
    min_confidence: float | None = None
    """The confidence floor applied, if any."""

    @property
    def nearest_level(self) -> int:
        """The rubric level the score is closest to.

        Rounds half away from zero, so a score of exactly 1.5 reads as level 2 rather than Python's
        banker's rounding to 2 in some cases and 1 in others.
        """
        return int(self.score + 0.5)
