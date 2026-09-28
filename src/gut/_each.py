"""`each()`: the same question about many subjects, in as few model calls as the backend allows.

```python
for comment, decision in zip(comments, gut.each(comments).likely("is spam")):
    if decision:
        hide(comment)
```

A loop of `gut.likely(...)` calls works, one model call at a time. `each()` hands every subject to
the backend together: Jev and any server behind a network get concurrent requests, a local model
runs them through batched forward passes, and a `Cascade` sends the whole lot to its cheap stage and
only the unsettled ones onwards. Subjects already in the cache are not asked again, and a subject
that appears twice is asked once.

The decisions come back as a list in the same order as the subjects, each exactly what
`gut.likely(subject, ...)` would have returned for it -- same outcome rules, same posture words,
same `on_decision` hook -- with `source="batch"`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum
from typing import TypeVar

from gut._api import (
    _criteria_from_enum,
    _resolve_min_confidence,
    _resolve_policy,
    _Resolved,
    _warn_without_catch_all,
    build_choice_decision,
    build_decision,
    build_score_decision,
)
from gut._backends._many import DEFAULT_CONCURRENCY, ask_all
from gut._backends.base import Backend
from gut._cache import CacheEntry, cache_key
from gut._config import current_backend, current_cache
from gut._decision import ChoiceDecision, Decision, ScoreDecision
from gut._errors import BackendError
from gut._posture import Lean, Stakes
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State, canonical_json
from gut._site import caller_site

E = TypeVar("E", bound=Enum)


class Each:
    """Many subjects, waiting for a question. Created by `each()`."""

    def __init__(
        self,
        subjects: Iterable[State],
        *,
        backend: Backend | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self.subjects: list[State] = list(subjects)
        self._backend = backend
        self._concurrency = concurrency

    def __len__(self) -> int:
        return len(self.subjects)

    def likely(
        self,
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
    ) -> list[Decision]:
        """`gut.likely(subject, question, ...)` for every subject. See `gut.likely`."""
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
        return [build_decision(spec, resolved, site, rule=rule) for resolved in self._answers(spec)]

    def classify(
        self,
        enum_class: type[E],
        *,
        question: str | None = None,
        stakes: Stakes | None = None,
        ask_human: bool = False,
        min_confidence: float | None = None,
    ) -> list[ChoiceDecision[E]]:
        """`gut.classify(subject, enum_class, ...)` for every subject. See `gut.classify`."""
        min_confidence = _resolve_min_confidence(
            stakes=stakes, ask_human=ask_human, min_confidence=min_confidence, kind="classify"
        )
        criteria = _criteria_from_enum(enum_class)
        _warn_without_catch_all(enum_class)
        spec = ChoiceSpec(instructions=question, criteria=criteria)
        site = caller_site()
        return [
            build_choice_decision(spec, enum_class, resolved, site, min_confidence=min_confidence)
            for resolved in self._answers(spec)
        ]

    def rate(
        self,
        levels: Sequence[str],
        *,
        question: str | None = None,
        stakes: Stakes | None = None,
        ask_human: bool = False,
        min_confidence: float | None = None,
    ) -> list[ScoreDecision]:
        """`gut.rate(subject, levels, ...)` for every subject. See `gut.rate`."""
        min_confidence = _resolve_min_confidence(
            stakes=stakes, ask_human=ask_human, min_confidence=min_confidence, kind="rate"
        )
        spec = ScoreSpec(instructions=question, criteria=levels)
        site = caller_site()
        return [
            build_score_decision(spec, resolved, site, min_confidence=min_confidence)
            for resolved in self._answers(spec)
        ]

    def _answers(self, spec: QuestionSpec) -> list[_Resolved]:
        """One answer per subject, in order: from the cache where possible, the rest in one go."""
        if not self.subjects:
            return []
        backend = current_backend() if self._backend is None else self._backend
        cache = current_cache()

        # A subject that appears twice is one question, asked once.
        unique: list[State] = []
        position: dict[str, int] = {}
        order: list[int] = []
        for subject in self.subjects:
            key = canonical_json(subject)
            if key not in position:
                position[key] = len(unique)
                unique.append(subject)
            order.append(position[key])

        resolved: list[_Resolved | None] = [None] * len(unique)
        misses: list[int] = []
        for index, subject in enumerate(unique):
            hit = cache.get(cache_key(subject, spec, backend.model_id))
            if hit is None:
                misses.append(index)
            else:
                resolved[index] = _Resolved(
                    answer=hit.answer, model=hit.model, source="cache", latency_ms=None
                )

        responses = ask_all(
            backend, [(unique[i], {"q": spec}) for i in misses], concurrency=self._concurrency
        )
        for index, response in zip(misses, responses, strict=True):
            if "q" not in response.answers:
                raise BackendError(f"{type(backend).__name__} returned no answer for a subject.")
            answer, model = response.answers["q"], response.model_for("q")
            cache.set(cache_key(unique[index], spec, backend.model_id), CacheEntry(answer, model))
            resolved[index] = _Resolved(answer=answer, model=model, source="batch", latency_ms=None)

        return [resolved[index] for index in order]  # type: ignore[misc]


def each(
    subjects: Iterable[State],
    *,
    backend: Backend | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Each:
    """Ask the same question about many subjects at once.

    Args:
        subjects: Anything a single `likely` accepts, as many as you like.
        backend: Override the configured backend.
        concurrency: Subjects in flight at once, for a backend that takes one per request.

    Example:
        ```python
        spam = gut.each(comments).likely("is spam")
        teams = gut.each(tickets).classify(Team)
        ```
    """
    return Each(subjects, backend=backend, concurrency=concurrency)


__all__ = ["Each", "each"]
