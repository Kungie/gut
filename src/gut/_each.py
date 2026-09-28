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

Every method has an awaitable twin -- `alikely`, `aclassify`, `arate` -- for code running on an
event loop: Jev and the OpenAI-compatible backend are then asked natively, the rest from a worker
thread.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
from gut._backends._many import DEFAULT_CONCURRENCY, aask_all, ask_all
from gut._backends.base import Backend, BackendResponse
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

    # ------------------------------------------------------------------ blocking

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

    # ------------------------------------------------------------------ awaitable

    async def alikely(
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
        """`likely`, awaitable. Jev and servers get concurrent requests on the event loop."""
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
        answers = await self._aanswers(spec)
        return [build_decision(spec, resolved, site, rule=rule) for resolved in answers]

    async def aclassify(
        self,
        enum_class: type[E],
        *,
        question: str | None = None,
        stakes: Stakes | None = None,
        ask_human: bool = False,
        min_confidence: float | None = None,
    ) -> list[ChoiceDecision[E]]:
        """`classify`, awaitable."""
        min_confidence = _resolve_min_confidence(
            stakes=stakes, ask_human=ask_human, min_confidence=min_confidence, kind="classify"
        )
        criteria = _criteria_from_enum(enum_class)
        _warn_without_catch_all(enum_class)
        spec = ChoiceSpec(instructions=question, criteria=criteria)
        site = caller_site()
        return [
            build_choice_decision(spec, enum_class, resolved, site, min_confidence=min_confidence)
            for resolved in await self._aanswers(spec)
        ]

    async def arate(
        self,
        levels: Sequence[str],
        *,
        question: str | None = None,
        stakes: Stakes | None = None,
        ask_human: bool = False,
        min_confidence: float | None = None,
    ) -> list[ScoreDecision]:
        """`rate`, awaitable."""
        min_confidence = _resolve_min_confidence(
            stakes=stakes, ask_human=ask_human, min_confidence=min_confidence, kind="rate"
        )
        spec = ScoreSpec(instructions=question, criteria=levels)
        site = caller_site()
        return [
            build_score_decision(spec, resolved, site, min_confidence=min_confidence)
            for resolved in await self._aanswers(spec)
        ]

    # ------------------------------------------------------------------ shared

    def _answers(self, spec: QuestionSpec) -> list[_Resolved]:
        """One answer per subject, in order: from the cache where possible, the rest in one go."""
        batch = _Batch.plan(self.subjects, spec, self._backend)
        responses = ask_all(batch.backend, batch.items(), concurrency=self._concurrency)
        return batch.finish(responses)

    async def _aanswers(self, spec: QuestionSpec) -> list[_Resolved]:
        """`_answers` without blocking the event loop."""
        batch = _Batch.plan(self.subjects, spec, self._backend)
        responses = await aask_all(batch.backend, batch.items(), concurrency=self._concurrency)
        return batch.finish(responses)


@dataclass
class _Batch:
    """One question about many subjects: what the cache already knows, and what is left to ask."""

    spec: QuestionSpec
    backend: Backend
    unique: list[State]
    """Each distinct subject once."""
    order: list[int]
    """For every subject asked about, its position in `unique`."""
    resolved: list[_Resolved | None]
    misses: list[int]
    """Positions in `unique` that the cache could not answer."""

    @classmethod
    def plan(cls, subjects: Sequence[State], spec: QuestionSpec, chosen: Backend | None) -> _Batch:
        backend = chosen if chosen is not None else current_backend()
        cache = current_cache()

        # A subject that appears twice is one question, asked once.
        unique: list[State] = []
        position: dict[str, int] = {}
        order: list[int] = []
        for subject in subjects:
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
        return cls(spec, backend, unique, order, resolved, misses)

    def items(self) -> list[tuple[State, dict[str, QuestionSpec]]]:
        """What still has to be asked, one item per subject the cache did not know."""
        return [(self.unique[index], {"q": self.spec}) for index in self.misses]

    def finish(self, responses: Sequence[BackendResponse]) -> list[_Resolved]:
        """Cache the fresh answers and return one answer per subject, in the original order."""
        cache = current_cache()
        for index, response in zip(self.misses, responses, strict=True):
            if "q" not in response.answers:
                raise BackendError(
                    f"{type(self.backend).__name__} returned no answer for a subject."
                )
            answer, model = response.answers["q"], response.model_for("q")
            key = cache_key(self.unique[index], self.spec, self.backend.model_id)
            cache.set(key, CacheEntry(answer, model))
            self.resolved[index] = _Resolved(
                answer=answer, model=model, source="batch", latency_ms=None
            )
        return [self.resolved[index] for index in self.order]  # type: ignore[misc]


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
