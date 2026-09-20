"""`judge()`: the same batching as `@semantic`, asked for explicitly.

```python
with gut.judge(ticket) as j:
    bug = j.likely("is a bug report")
    team = j.classify(Team)
    if bug:          # <- everything registered so far goes out in one request, here
        route(team)  # <- already answered
```

**The lazy-resolution rule, stated once.** `j.likely(...)` registers a question and hands back a
handle. Nothing is sent. The first time any handle is *used* -- tested for truth, matched, compared,
or read for a field -- every question registered up to that moment is sent in a single request.
Questions registered afterwards form a new group, sent on the next use.

So where you first read a result decides what got batched with what. Register everything you might
need before reading any of it, and you get one request. Interleave registering and reading, and you
get one request per read. That is a deliberate exposure rather than a hidden heuristic: with
`@semantic` the grouping is inferred, and here it is yours to place.

Two consequences worth knowing:

- `repr()` never resolves. Printing a handle in a debugger must not cost a request, so it reports
  whether it is pending instead.
- Leaving the `with` block resolves nothing. A question nobody reads is never asked and never
  billed. The block only stops further registration; handles still resolve afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Generic, Literal, TypeVar, cast

from gut._api import (
    _criteria_from_enum,
    _Resolved,
    _warn_without_catch_all,
    build_choice_decision,
    build_decision,
    build_score_decision,
)
from gut._backends.base import Backend
from gut._batching import fetch
from gut._config import current_backend
from gut._decision import BaseDecision, ChoiceDecision, Decision, ScoreDecision
from gut._errors import JudgeClosedError
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State
from gut._rule import Policy, policy
from gut._site import CallSite, caller_site

E = TypeVar("E", bound=Enum)
D = TypeVar("D", bound=BaseDecision)


@dataclass
class _Registration:
    """A question waiting to be asked, and how to turn its answer into a decision."""

    spec: QuestionSpec
    site: CallSite
    rule: Policy | None = None
    enum_class: type[Enum] | None = None
    min_confidence: float | None = None


class Lazy(Generic[D]):
    """A decision that has been registered but not yet asked.

    Behaves like the decision it will become: truthy, matchable, comparable, and readable field by
    field. Any of those triggers resolution. `repr()` deliberately does not.
    """

    __slots__ = ("_index", "_judge")

    def __init__(self, judge: Judge, index: int) -> None:
        self._judge = judge
        self._index = index

    @property
    def decision(self) -> D:
        """The resolved decision, asking for it if necessary."""
        return cast("D", self._judge._decision_for(self._index))

    @property
    def pending(self) -> bool:
        """Whether this handle has yet to be resolved. Reading it costs nothing."""
        return self._index not in self._judge._decisions

    def __getattr__(self, name: str) -> object:
        if name.startswith("__"):  # pragma: no cover - protocol lookups go through the type
            raise AttributeError(name)
        return getattr(self.decision, name)

    def __bool__(self) -> bool:
        return bool(self.decision)

    def __eq__(self, other: object) -> bool:
        return self.decision == other

    def __hash__(self) -> int:
        return hash(self.decision)

    def __repr__(self) -> str:
        """Describe the handle without resolving it: a debugger must not spend money."""
        if self.pending:
            return f"<Lazy pending #{self._index}>"
        return f"<Lazy {self.decision!r}>"


class Judge:
    """Collects questions about one subject and answers them in as few requests as possible.

    Created by `judge()`; see this module's documentation for the lazy-resolution rule.
    """

    def __init__(self, subject: State, *, backend: Backend | None = None) -> None:
        self.subject = subject
        self._backend = backend
        self._registrations: list[_Registration] = []
        self._decisions: dict[int, BaseDecision] = {}
        self._pending: list[int] = []
        self._closed = False
        self.requests = 0
        """How many backend requests this judge has made."""

    def __enter__(self) -> Judge:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        # Nothing is forced: a question nobody reads is never asked and never billed.
        self._closed = True
        return False

    def _register(self, registration: _Registration) -> int:
        if self._closed:
            raise JudgeClosedError(
                "This judge() block has ended; register questions inside it, or open a new one."
            )
        index = len(self._registrations)
        self._registrations.append(registration)
        self._pending.append(index)
        return index

    def likely(
        self,
        question: str,
        *,
        cost_false_yes: float | None = None,
        cost_false_no: float | None = None,
        cost_human: float | None = None,
        threshold: float | None = None,
        unsure_band: tuple[float, float] | None = None,
        yes_means: str | None = None,
        no_means: str | None = None,
    ) -> Decision:
        """Register a yes/no judgment. See `gut.likely` for the arguments."""
        rule = policy(
            cost_false_yes=cost_false_yes,
            cost_false_no=cost_false_no,
            cost_human=cost_human,
            threshold=threshold,
            unsure_band=unsure_band,
        )
        spec = NoulSpec(instructions=question, yes_means=yes_means, no_means=no_means)
        index = self._register(_Registration(spec=spec, site=caller_site(), rule=rule))
        return cast("Decision", Lazy[Decision](self, index))

    def classify(
        self,
        enum_class: type[E],
        *,
        question: str | None = None,
        min_confidence: float | None = None,
    ) -> ChoiceDecision[E]:
        """Register a categorical judgment. See `gut.classify` for the arguments."""
        criteria = _criteria_from_enum(enum_class)
        _warn_without_catch_all(enum_class)
        spec = ChoiceSpec(instructions=question, criteria=criteria)
        index = self._register(
            _Registration(
                spec=spec,
                site=caller_site(),
                enum_class=enum_class,
                min_confidence=min_confidence,
            )
        )
        return cast("ChoiceDecision[E]", Lazy[ChoiceDecision[E]](self, index))

    def rate(
        self,
        levels: Sequence[str],
        *,
        question: str | None = None,
        min_confidence: float | None = None,
    ) -> ScoreDecision:
        """Register an ordinal judgment. See `gut.rate` for the arguments."""
        spec = ScoreSpec(instructions=question, criteria=levels)
        index = self._register(
            _Registration(spec=spec, site=caller_site(), min_confidence=min_confidence)
        )
        return cast("ScoreDecision", Lazy[ScoreDecision](self, index))

    def resolve(self) -> None:
        """Ask everything registered so far, without reading any of it."""
        if self._pending:
            self._flush()

    def _decision_for(self, index: int) -> BaseDecision:
        if index not in self._decisions:
            self._flush()
        return self._decisions[index]

    def _flush(self) -> None:
        indices = self._pending
        self._pending = []
        if not indices:  # pragma: no cover - _decision_for only flushes for an unresolved index
            return

        backend = current_backend() if self._backend is None else self._backend
        registrations = [self._registrations[index] for index in indices]
        answers = fetch(self.subject, [r.spec for r in registrations], backend)
        self.requests += 1

        for index, registration in zip(indices, registrations, strict=True):
            entry = answers[registration.spec]
            resolved = _Resolved(
                answer=entry.answer, model=entry.model, source="prefetch", latency_ms=None
            )
            self._decisions[index] = _build(registration, resolved)


def _build(registration: _Registration, resolved: _Resolved) -> BaseDecision:
    """Turn one raw answer into the decision its registration asked for."""
    spec = registration.spec
    match spec:
        case NoulSpec():
            assert registration.rule is not None
            return build_decision(spec, resolved, registration.site, rule=registration.rule)
        case ChoiceSpec():
            assert registration.enum_class is not None
            return build_choice_decision(
                spec,
                registration.enum_class,
                resolved,
                registration.site,
                min_confidence=registration.min_confidence,
            )
        case ScoreSpec():  # pragma: no branch - exhaustive, proven by mypy
            return build_score_decision(
                spec, resolved, registration.site, min_confidence=registration.min_confidence
            )


def judge(subject: State, *, backend: Backend | None = None) -> Judge:
    """Ask several questions about one subject in as few requests as possible.

    Args:
        subject: What every question in the block is about.
        backend: Override the configured backend for this block.

    Example:
        ```python
        with gut.judge(ticket) as j:
            bug = j.likely("is a bug report")
            team = j.classify(Team)
            if bug:            # one request, covering both
                route(team)
        ```
    """
    return Judge(subject, backend=backend)


__all__ = ["Judge", "Lazy", "judge"]
