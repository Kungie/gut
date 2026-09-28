"""Counting what gut's model calls cost, and stopping at a budget.

```python
with gut.usage(max_cost=0.50) as spent:
    flagged = gut.each(comments).likely("is spam")
print(spent)            # 1,200 calls, $0.0114
```

Every call gut makes on its own behalf -- a `likely`, a batch from `each()`, a prefetch for
`@semantic` or `judge()` -- is counted by every `usage()` block it runs inside. Answers from the
cache cost nothing and are not counted. The cost is what the backend reports: `0.0` for a model in
your process, OpenRouter's own figure for Jev through OpenRouter, and unknown for a service that
does not say, which is counted separately rather than guessed at.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from gut._backends.base import BackendResponse
from gut._errors import BudgetExceeded, ConfigurationError


@dataclass
class Usage:
    """What the calls made inside one `usage()` block have cost so far."""

    max_cost: float | None = None
    """The budget in US dollars, or `None` for none."""
    calls: int = 0
    """Backend calls made: one per subject asked, whatever number of questions rode along."""
    input_tokens: int = 0
    """Billable input tokens, summed over the calls that reported them."""
    cost: float = 0.0
    """US dollars, summed over the calls that reported a cost."""
    unpriced_calls: int = 0
    """Calls whose backend did not say what they cost. While this is above zero, `cost` is a floor
    rather than the total, and a budget cannot be kept."""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def cost_known(self) -> bool:
        """Whether every call so far reported its cost, so that `cost` is the whole of it."""
        return self.unpriced_calls == 0

    def record(self, response: BackendResponse) -> None:
        """Count one call."""
        with self._lock:
            self.calls += 1
            self.input_tokens += response.input_tokens or 0
            if response.cost is None:
                self.unpriced_calls += 1
            else:
                self.cost += response.cost

    def check(self) -> None:
        """Refuse to start another call once the budget is spent, or cannot be known.

        Raises:
            BudgetExceeded: `max_cost` is reached, or a call did not report its cost.
        """
        if self.max_cost is None:
            return
        if not self.cost_known:
            raise BudgetExceeded(
                self,
                f"max_cost=${self.max_cost:g} cannot be kept: the backend does not report what "
                f"its calls cost. OpenRouter does, and a model in your process costs nothing.",
            )
        # Free calls never exhaust a budget, so max_cost=0 means "only what costs nothing".
        if self.cost > 0 and self.cost >= self.max_cost:
            raise BudgetExceeded(
                self,
                f"Stopped before the next call: ${self.cost:.6g} spent of a "
                f"${self.max_cost:g} budget, over {self.calls} calls.",
            )

    def to_dict(self) -> dict[str, Any]:
        """The counts as JSON-compatible data."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cost": round(self.cost, 8) if self.cost_known else None,
            "unpriced_calls": self.unpriced_calls,
            "max_cost": self.max_cost,
        }

    def __str__(self) -> str:
        calls = f"{self.calls:,} call{'' if self.calls == 1 else 's'}"
        if self.calls and not self.cost_known:
            tokens = f", {self.input_tokens:,} input tokens" if self.input_tokens else ""
            return f"{calls}{tokens}, cost not reported"
        return f"{calls}, ${self.cost:.6g}"


_active: ContextVar[tuple[Usage, ...]] = ContextVar("gut_usage", default=())


@contextmanager
def usage(*, max_cost: float | None = None) -> Iterator[Usage]:
    """Count what the calls inside the block cost, and optionally stop at a budget.

    Blocks nest: a call counts toward every block it runs inside, and each block keeps its own
    budget. The count follows the code into threads and tasks started with `asyncio`, since it
    lives in a context variable.

    Args:
        max_cost: A budget in US dollars. Once it is reached, the next call gut would make raises
            `BudgetExceeded` instead. A call already under way finishes, and an `each()` batch is
            one decision to spend, so a budget can be passed by the batch that reaches it -- ask in
            smaller batches for a tighter stop.

    Raises:
        ConfigurationError: `max_cost` is negative.
    """
    if max_cost is not None and max_cost < 0:
        raise ConfigurationError(f"max_cost must be zero or more, got {max_cost}.")
    meter = Usage(max_cost=max_cost)
    token = _active.set((*_active.get(), meter))
    try:
        yield meter
    finally:
        _active.reset(token)


def before_call() -> None:
    """Called before gut asks a backend anything: every active budget must allow it."""
    for meter in _active.get():
        meter.check()


def after_call(responses: Iterable[BackendResponse]) -> None:
    """Called with what came back: every active block counts it."""
    meters = _active.get()
    if not meters:
        return
    for response in responses:
        for meter in meters:
            meter.record(response)


__all__ = ["Usage", "after_call", "before_call", "usage"]
