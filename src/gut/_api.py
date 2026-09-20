"""The three primitives: `likely`, `classify`, `rate`.

Each one asks a single question and comes back with a decision you can branch on. They are the
whole public surface for making a judgment; everything else -- batching, caching, logging -- changes
how these get answered, not how they are written.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from gut._backends.base import Answer, Backend, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._cache import CacheEntry, cache_key
from gut._config import current_backend, current_cache
from gut._decision import ChoiceDecision, Decision, DecisionSource, ScoreDecision
from gut._errors import BackendError, QuestionError
from gut._outcomes import Outcome
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State
from gut._rule import Policy, policy
from gut._scope import current_prefetch
from gut._site import CallSite, caller_site, decision_id

E = TypeVar("E", bound=Enum)
A = TypeVar("A", NoulAnswer, ChoiceAnswer, ScoreAnswer)

CATCH_ALL_NAMES = frozenset({"OTHER", "UNKNOWN", "NONE", "MISC", "MISCELLANEOUS", "CATCH_ALL"})
"""Member names taken as a catch-all, so `classify` stays quiet about forced choices."""

_warned_enums: set[type[Enum]] = set()


@dataclass(frozen=True, slots=True)
class _Resolved:
    """One answer and how it was obtained."""

    answer: Answer
    model: str
    source: DecisionSource
    latency_ms: float | None


def _ask(state: State, spec: QuestionSpec, backend: Backend | None) -> _Resolved:
    """Answer one question, from cache when possible."""
    chosen = current_backend() if backend is None else backend

    # A batch fetched by @semantic or judge() is checked first: it is already paid for, and it is
    # correct even when caching is switched off.
    prefetch = current_prefetch()
    if prefetch is not None:
        prefetched = prefetch.take(state, spec)
        if prefetched is not None:
            return _Resolved(
                answer=prefetched.answer,
                model=prefetched.model,
                source="prefetch",
                latency_ms=None,
            )

    cache = current_cache()
    key = cache_key(state, spec, chosen.model_id)

    hit = cache.get(key)
    if hit is not None:
        # The stored model is the version that actually answered, which is more specific than
        # the alias in the key. Reporting the alias would make a recorded decision unfalsifiable.
        return _Resolved(answer=hit.answer, model=hit.model, source="cache", latency_ms=None)

    started = time.perf_counter()
    response = chosen.ask(state, {"q": spec})
    latency_ms = (time.perf_counter() - started) * 1000.0
    if "q" not in response.answers:
        raise BackendError(f"{type(chosen).__name__} returned no answer for the question asked.")
    answer = response.answers["q"]
    cache.set(key, CacheEntry(answer=answer, model=response.model))
    return _Resolved(answer=answer, model=response.model, source="backend", latency_ms=latency_ms)


def _expect(answer: Answer, kind: type[A], spec: QuestionSpec) -> A:
    """Narrow a raw answer to the shape its question called for."""
    if not isinstance(answer, kind):
        raise BackendError(
            f"Expected a {kind.__name__} for {spec.canonical()['type']} question "
            f"{spec.fingerprint}, got {type(answer).__name__}."
        )
    return answer


def _min_confidence_outcome(confidence: float, min_confidence: float | None) -> Outcome:
    """UNSURE below the floor, YES above it.

    There is no NO here: a classification or a rating always names something, and the only question
    is whether the distribution was peaked enough to act on.
    """
    if min_confidence is not None and confidence < min_confidence:
        return Outcome.UNSURE
    return Outcome.YES


def build_decision(
    spec: NoulSpec, resolved: _Resolved, site: CallSite, *, rule: Policy
) -> Decision:
    """Assemble a yes/no decision from a raw answer."""
    noul = _expect(resolved.answer, NoulAnswer, spec)
    return Decision(
        outcome=rule.decide(noul.p),
        id=decision_id(spec.fingerprint, site),
        model=resolved.model,
        source=resolved.source,
        latency_ms=resolved.latency_ms,
        p=noul.p,
        policy=rule,
    )


def build_choice_decision(
    spec: ChoiceSpec,
    enum_class: type[E],
    resolved: _Resolved,
    site: CallSite,
    *,
    min_confidence: float | None,
) -> ChoiceDecision[E]:
    """Assemble a categorical decision, mapping the chosen label back to its enum member."""
    choice = _expect(resolved.answer, ChoiceAnswer, spec)
    by_name = {member.name: member for member in enum_class}
    if choice.choice not in by_name:
        raise BackendError(
            f"Backend chose {choice.choice!r}, which is not a member of {enum_class.__name__}."
        )
    outcome = _min_confidence_outcome(choice.confidence, min_confidence)
    return ChoiceDecision(
        outcome=outcome,
        id=decision_id(spec.fingerprint, site),
        model=resolved.model,
        source=resolved.source,
        latency_ms=resolved.latency_ms,
        value=by_name[choice.choice] if outcome is Outcome.YES else Outcome.UNSURE,
        probabilities={
            by_name[name]: value for name, value in choice.probabilities.items() if name in by_name
        },
        confidence=choice.confidence,
        min_confidence=min_confidence,
    )


def build_score_decision(
    spec: ScoreSpec, resolved: _Resolved, site: CallSite, *, min_confidence: float | None
) -> ScoreDecision:
    """Assemble an ordinal decision from a raw answer."""
    score = _expect(resolved.answer, ScoreAnswer, spec)
    return ScoreDecision(
        outcome=_min_confidence_outcome(score.confidence, min_confidence),
        id=decision_id(spec.fingerprint, site),
        model=resolved.model,
        source=resolved.source,
        latency_ms=resolved.latency_ms,
        score=score.score,
        probabilities=dict(score.probabilities),
        confidence=score.confidence,
        levels=tuple(spec.criteria),
        min_confidence=min_confidence,
    )


def likely(
    subject: State,
    question: str,
    *,
    cost_false_yes: float | None = None,
    cost_false_no: float | None = None,
    cost_human: float | None = None,
    threshold: float | None = None,
    unsure_band: tuple[float, float] | None = None,
    yes_means: str | None = None,
    no_means: str | None = None,
    backend: Backend | None = None,
) -> Decision:
    """Judge whether `question` is true of `subject`.

    Args:
        subject: What to judge: text, a JSON-shaped dict, or a list of strings.
        question: The claim to evaluate, written as a statement or a question.
        cost_false_yes: What it costs to act on a yes that turns out wrong.
        cost_false_no: What it costs to act on a no that turns out wrong.
        cost_human: What it costs to ask a person instead. Omit it and `UNSURE` is impossible.
        threshold: Escape hatch -- decide yes strictly above this probability. Not combinable
            with costs.
        unsure_band: Escape hatch -- `(lo, hi)`, UNSURE inside the closed interval.
        yes_means: Optional description of what counts as yes, for an ambiguous question.
        no_means: Optional description of what counts as no.
        backend: Override the configured backend for this call.

    Returns:
        A `Decision` whose `outcome` is YES, NO or UNSURE.

    Example:
        ```python
        d = gut.likely(email, "the customer threatens to cancel",
                       cost_false_yes=2, cost_false_no=50, cost_human=5)
        ```
    """
    rule = policy(
        cost_false_yes=cost_false_yes,
        cost_false_no=cost_false_no,
        cost_human=cost_human,
        threshold=threshold,
        unsure_band=unsure_band,
    )
    spec = NoulSpec(instructions=question, yes_means=yes_means, no_means=no_means)
    site = caller_site()
    return build_decision(spec, _ask(subject, spec, backend), site, rule=rule)


def _criteria_from_enum(enum_class: type[E]) -> Mapping[str, str | None]:
    members = list(enum_class)
    if not members:
        raise QuestionError(f"{enum_class.__name__} has no members to choose between.")
    criteria: dict[str, str | None] = {}
    for member in members:
        if not isinstance(member.value, str) or not member.value.strip():
            raise QuestionError(
                f"{enum_class.__name__}.{member.name} must have a non-empty string value: the "
                f"value is the description the model is shown. Got {member.value!r}."
            )
        criteria[member.name] = member.value
    return criteria


def _warn_without_catch_all(enum_class: type[E]) -> None:
    if enum_class in _warned_enums:
        return
    _warned_enums.add(enum_class)
    if any(member.name.upper() in CATCH_ALL_NAMES for member in enum_class):
        return
    warnings.warn(
        f"{enum_class.__name__} has no catch-all member, so every subject is forced into one of "
        f"its categories even when none fits. Consider adding an OTHER member, or set "
        f"min_confidence to route weak matches to UNSURE.",
        UserWarning,
        stacklevel=3,
    )


def classify(
    subject: State,
    enum_class: type[E],
    *,
    question: str | None = None,
    min_confidence: float | None = None,
    backend: Backend | None = None,
) -> ChoiceDecision[E]:
    """Pick the member of `enum_class` that fits `subject`.

    Each member's **value** is the description shown to the model, and its **name** is the label
    that comes back. So the enum is both the type and the prompt:

    ```python
    class Team(Enum):
        BILLING = "questions about invoices, charges, refunds"
        PLATFORM = "outages, latency, API errors"
        OTHER = "anything else"
    ```

    Args:
        subject: What to classify.
        enum_class: The categories, as an `Enum` whose values are descriptions.
        question: What the model should decide. Omit it and the descriptions speak for themselves.
        min_confidence: Below this confidence the outcome is UNSURE and `value` is `UNSURE`.
            Confidence is how peaked the distribution is, **not** a probability of being right.
        backend: Override the configured backend for this call.

    Returns:
        A `ChoiceDecision` whose `value` is an enum member, or `UNSURE`.

    Warns:
        UserWarning: `enum_class` has no catch-all member, so nothing can be "none of these".
    """
    criteria = _criteria_from_enum(enum_class)
    _warn_without_catch_all(enum_class)
    spec = ChoiceSpec(instructions=question, criteria=criteria)
    site = caller_site()
    return build_choice_decision(
        spec, enum_class, _ask(subject, spec, backend), site, min_confidence=min_confidence
    )


def rate(
    subject: State,
    levels: Sequence[str],
    *,
    question: str | None = None,
    min_confidence: float | None = None,
    backend: Backend | None = None,
) -> ScoreDecision:
    """Rate `subject` against an ordered rubric.

    Args:
        subject: What to rate.
        levels: Between 2 and 10 level descriptions, in order; the first is level 0.
        question: What the model should rate. Omit it and the rubric speaks for itself.
        min_confidence: Below this confidence the outcome is UNSURE.
        backend: Override the configured backend for this call.

    Returns:
        A `ScoreDecision` whose `score` may land between levels.

    Example:
        ```python
        r = gut.rate(ticket, ["calm", "annoyed", "angry", "threatening to leave"])
        ```
    """
    spec = ScoreSpec(instructions=question, criteria=levels)
    site = caller_site()
    return build_score_decision(
        spec, _ask(subject, spec, backend), site, min_confidence=min_confidence
    )


__all__ = ["classify", "likely", "rate"]
