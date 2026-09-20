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
from typing import Any, TypeVar

from gut._backends.base import Answer, Backend, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._cache import CacheEntry, cache_key
from gut._calibrators import correct
from gut._config import (
    current_backend,
    current_cache,
    current_calibration,
    current_sink,
    recording,
)
from gut._decision import ChoiceDecision, Decision, DecisionSource, ScoreDecision
from gut._errors import BackendError, PolicyError, QuestionError
from gut._log import DecisionRecord, _now, emit_decision, policy_to_json, site_to_json
from gut._outcomes import Outcome
from gut._posture import Lean, Stakes, min_confidence_for, policy_for
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
    """What the decision is made on: corrected, if a calibrator applies."""
    model: str
    source: DecisionSource
    latency_ms: float | None
    raw_answer: Answer | None = None
    """What the model said before correction, when something corrected it."""


def _corrected(answer: Answer, spec: QuestionSpec, model: str) -> tuple[Answer, Answer | None]:
    """Apply the configured correction to one answer.

    Corrections are applied on the way out of the cache rather than on the way in: a cached answer
    stays valid when the calibrator changes, because what was stored is what the model said.
    """
    return correct(answer, spec.fingerprint, model, current_calibration())


def _ask(state: State, spec: QuestionSpec, backend: Backend | None) -> _Resolved:
    """Answer one question, from cache when possible."""
    chosen = current_backend() if backend is None else backend

    # A batch fetched by @semantic or judge() is checked first: it is already paid for, and it is
    # correct even when caching is switched off.
    prefetch = current_prefetch()
    if prefetch is not None:
        prefetched = prefetch.take(state, spec)
        if prefetched is not None:
            answer, raw = _corrected(prefetched.answer, spec, prefetched.model)
            return _Resolved(
                answer=answer,
                model=prefetched.model,
                source="prefetch",
                latency_ms=None,
                raw_answer=raw,
            )

    cache = current_cache()
    key = cache_key(state, spec, chosen.model_id)

    hit = cache.get(key)
    if hit is not None:
        # The stored model is the version that actually answered, which is more specific than
        # the alias in the key. Reporting the alias would make a recorded decision unfalsifiable.
        answer, raw = _corrected(hit.answer, spec, hit.model)
        return _Resolved(
            answer=answer, model=hit.model, source="cache", latency_ms=None, raw_answer=raw
        )

    started = time.perf_counter()
    response = chosen.ask(state, {"q": spec})
    latency_ms = (time.perf_counter() - started) * 1000.0
    if "q" not in response.answers:
        raise BackendError(f"{type(chosen).__name__} returned no answer for the question asked.")
    answer = response.answers["q"]
    # Cache what the model said, then correct on the way out, so a cached answer survives a refit.
    cache.set(key, CacheEntry(answer=answer, model=response.model))
    corrected, raw = _corrected(answer, spec, response.model)
    return _Resolved(
        answer=corrected,
        model=response.model,
        source="backend",
        latency_ms=latency_ms,
        raw_answer=raw,
    )


def _expect(answer: Answer, kind: type[A], spec: QuestionSpec) -> A:
    """Narrow a raw answer to the shape its question called for."""
    if not isinstance(answer, kind):
        raise BackendError(
            f"Expected a {kind.__name__} for {spec.canonical()['type']} question "
            f"{spec.fingerprint}, got {type(answer).__name__}."
        )
    return answer


def _resolve_policy(
    *,
    stakes: Stakes | None,
    lean: Lean | None,
    ask_human: bool,
    cost_false_yes: float | None,
    cost_false_no: float | None,
    cost_human: float | None,
    threshold: float | None,
    unsure_band: tuple[float, float] | None,
) -> Policy:
    """Pick the layer the caller asked for, and refuse a call that asks for both."""
    exact = {
        "cost_false_yes": cost_false_yes,
        "cost_false_no": cost_false_no,
        "cost_human": cost_human,
        "threshold": threshold,
        "unsure_band": unsure_band,
    }
    posture = {"stakes": stakes, "lean": lean, "ask_human": ask_human or None}
    given_exact = sorted(name for name, value in exact.items() if value is not None)
    given_posture = sorted(name for name, value in posture.items() if value is not None)

    if given_exact and given_posture:
        raise PolicyError(
            f"Describe the posture or give the costs, not both: got "
            f"{', '.join(given_posture)} alongside {', '.join(given_exact)}. "
            f"stakes/lean/ask_human are shorthand for exactly these costs -- use "
            f"gut.presets() to see which -- so mixing them leaves it ambiguous which wins."
        )
    if given_posture:
        return policy_for(stakes=stakes, lean=lean, ask_human=ask_human)
    return policy(
        cost_false_yes=cost_false_yes,
        cost_false_no=cost_false_no,
        cost_human=cost_human,
        threshold=threshold,
        unsure_band=unsure_band,
    )


def _resolve_min_confidence(
    *,
    stakes: Stakes | None,
    ask_human: bool,
    min_confidence: float | None,
    kind: str,
) -> float | None:
    """The same choice for `classify` and `rate`, where the gate is confidence, not cost."""
    if min_confidence is not None and (stakes is not None or ask_human):
        raise PolicyError(
            f"Describe the posture or give min_confidence, not both. For {kind} the posture is "
            f"shorthand for a confidence floor; gut.presets() shows the mapping."
        )
    if stakes is not None or ask_human:
        return min_confidence_for(stakes, ask_human)
    return min_confidence


def _min_confidence_outcome(confidence: float, min_confidence: float | None) -> Outcome:
    """UNSURE below the floor, YES above it.

    There is no NO here: a classification or a rating always names something, and the only question
    is whether the distribution was peaked enough to act on.
    """
    if min_confidence is not None and confidence < min_confidence:
        return Outcome.UNSURE
    return Outcome.YES


def _record(
    decision: Decision | ChoiceDecision[Any] | ScoreDecision,
    spec: QuestionSpec,
    site: CallSite,
    **extra: Any,
) -> None:
    """Write a decision to the configured sink, if anything is listening."""
    if not recording():
        return
    record = DecisionRecord(
        id=decision.id,
        kind=str(spec.canonical()["type"]),  # type: ignore[arg-type]
        timestamp=_now(),
        outcome=decision.outcome.value,
        model=decision.model,
        source=decision.source,
        question=spec.canonical(),
        site=site_to_json(site),
        latency_ms=decision.latency_ms,
        **extra,
    )
    emit_decision(record, current_sink())


def build_decision(
    spec: NoulSpec, resolved: _Resolved, site: CallSite, *, rule: Policy
) -> Decision:
    """Assemble a yes/no decision from a raw answer."""
    noul = _expect(resolved.answer, NoulAnswer, spec)
    raw = _expect(resolved.raw_answer, NoulAnswer, spec) if resolved.raw_answer else None
    decision = Decision(
        outcome=rule.decide(noul.p),
        id=decision_id(spec.fingerprint, site),
        model=resolved.model,
        source=resolved.source,
        latency_ms=resolved.latency_ms,
        p=noul.p,
        raw_p=raw.p if raw else None,
        policy=rule,
    )
    _record(
        decision,
        spec,
        site,
        p=noul.p,
        raw_p=raw.p if raw else None,
        costs=policy_to_json(rule),
    )
    return decision


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
    raw = _expect(resolved.raw_answer, ChoiceAnswer, spec) if resolved.raw_answer else None
    by_name = {member.name: member for member in enum_class}
    if choice.choice not in by_name:
        raise BackendError(
            f"Backend chose {choice.choice!r}, which is not a member of {enum_class.__name__}."
        )
    outcome = _min_confidence_outcome(choice.confidence, min_confidence)
    decision = ChoiceDecision(
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
        raw_confidence=raw.confidence if raw else None,
        min_confidence=min_confidence,
    )
    _record(
        decision,
        spec,
        site,
        value=decision.value.name if outcome is Outcome.YES else None,
        confidence=choice.confidence,
        raw_confidence=raw.confidence if raw else None,
        probabilities=dict(choice.probabilities),
        min_confidence=min_confidence,
    )
    return decision


def build_score_decision(
    spec: ScoreSpec, resolved: _Resolved, site: CallSite, *, min_confidence: float | None
) -> ScoreDecision:
    """Assemble an ordinal decision from a raw answer."""
    score = _expect(resolved.answer, ScoreAnswer, spec)
    raw = _expect(resolved.raw_answer, ScoreAnswer, spec) if resolved.raw_answer else None
    decision = ScoreDecision(
        outcome=_min_confidence_outcome(score.confidence, min_confidence),
        id=decision_id(spec.fingerprint, site),
        model=resolved.model,
        source=resolved.source,
        latency_ms=resolved.latency_ms,
        score=score.score,
        probabilities=dict(score.probabilities),
        confidence=score.confidence,
        raw_confidence=raw.confidence if raw else None,
        levels=tuple(spec.criteria),
        min_confidence=min_confidence,
    )
    _record(
        decision,
        spec,
        site,
        score=score.score,
        confidence=score.confidence,
        raw_confidence=raw.confidence if raw else None,
        probabilities={str(level): value for level, value in score.probabilities.items()},
        min_confidence=min_confidence,
    )
    return decision


def likely(
    subject: State,
    question: str,
    *,
    stakes: Stakes | None = None,
    lean: Lean | None = None,
    ask_human: bool = False,
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

    With nothing but a subject and a question it answers yes or no, above and below `0.5`. The
    keyword arguments come in two layers and you pick one: words, or numbers.

    Args:
        subject: What to judge: text, a JSON-shaped dict, or a list of strings.
        question: The claim to evaluate, written as a statement or a question.
        stakes: `"low"`, `"medium"` or `"high"` -- how costly an automatic mistake is next to a
            person deciding instead. Only widens the range where a person is asked, so it needs
            `ask_human=True` to do anything.
        lean: `"yes"` or `"no"` -- which way to err when in doubt, meaning which mistake is worse.
            Moves the point where yes overtakes no to `0.25` or `0.75`.
        ask_human: Whether `UNSURE` is a possible outcome at all. Off by default.
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

    Raises:
        PolicyError: Posture words and exact costs were both given.

    Example:
        ```python
        if gut.likely(email, "the customer threatens to cancel"):
            escalate()

        d = gut.likely(email, "the customer threatens to cancel",
                       stakes="high", lean="yes", ask_human=True)
        ```
    """
    rule = _resolve_policy(
        stakes=stakes,
        lean=lean,
        ask_human=ask_human,
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
    stakes: Stakes | None = None,
    ask_human: bool = False,
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
        stakes: `"low"`, `"medium"` or `"high"` -- how peaked the answer must be before it is acted
            on. Needs `ask_human=True`. There is no `lean` here: a four-way choice has no direction
            to err in.
        ask_human: Whether `UNSURE` is a possible outcome at all. Off by default.
        min_confidence: Below this confidence the outcome is UNSURE and `value` is `UNSURE`.
            Confidence is how peaked the distribution is, **not** a probability of being right.
        backend: Override the configured backend for this call.

    Returns:
        A `ChoiceDecision` whose `value` is an enum member, or `UNSURE`.

    Warns:
        UserWarning: `enum_class` has no catch-all member, so nothing can be "none of these".
    """
    min_confidence = _resolve_min_confidence(
        stakes=stakes, ask_human=ask_human, min_confidence=min_confidence, kind="classify"
    )
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
    stakes: Stakes | None = None,
    ask_human: bool = False,
    min_confidence: float | None = None,
    backend: Backend | None = None,
) -> ScoreDecision:
    """Rate `subject` against an ordered rubric.

    Args:
        subject: What to rate.
        levels: Between 2 and 10 level descriptions, in order; the first is level 0.
        question: What the model should rate. Omit it and the rubric speaks for itself.
        stakes: `"low"`, `"medium"` or `"high"` -- how peaked the answer must be before it is acted
            on. Needs `ask_human=True`.
        ask_human: Whether `UNSURE` is a possible outcome at all. Off by default.
        min_confidence: Below this confidence the outcome is UNSURE.
        backend: Override the configured backend for this call.

    Returns:
        A `ScoreDecision` whose `score` may land between levels.

    Example:
        ```python
        r = gut.rate(ticket, ["calm", "annoyed", "angry", "threatening to leave"])
        ```
    """
    min_confidence = _resolve_min_confidence(
        stakes=stakes, ask_human=ask_human, min_confidence=min_confidence, kind="rate"
    )
    spec = ScoreSpec(instructions=question, criteria=levels)
    site = caller_site()
    return build_score_decision(
        spec, _ask(subject, spec, backend), site, min_confidence=min_confidence
    )


__all__ = ["classify", "likely", "rate"]
