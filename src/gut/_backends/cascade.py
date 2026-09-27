"""`Cascade`: ask the cheapest model first, and only escalate what it is unsure of.

Most judgments a program makes are easy. A 70M-parameter NLI model on a laptop gets the obvious
cases of "is this spam?" right, and paying a bigger model to read those is waste. The
cascade asks each backend in order and keeps an answer as soon as it is decisive enough -- a
probability outside `unsure_band`, or a confidence at or above `min_confidence`. Only what is left
goes on to the next backend, and whatever reaches the last one is kept as it is.

```python
gut.configure(backend=gut.Cascade(
    gut.ZeroShotBackend(),                        # free, local, milliseconds
    gut.OpenAICompatibleBackend("gpt-4.1-nano"),  # only for what the first one could not settle
))
```

It is an ordinary backend, so everything else composes with it: `@semantic` hands it a whole batch,
the cache stores what it returns, and `Decision.model` names the model that actually answered each
question. The escalation band is the cascade's own; your `stakes` / `ask_human` still apply to the
final answer, so a question the last model is unsure of can still reach a person.

A backend that fails is treated like one that was unsure: its questions move on to the next. The
last backend's errors are raised, since there is nobody left to ask.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from collections.abc import Mapping

from gut._backends.base import Answer, Backend, BackendResponse, NoulAnswer
from gut._errors import BackendError, GutError, PolicyError
from gut._questions import QuestionSpec, State

logger = logging.getLogger("gut")


class Cascade:
    """A backend that tries `backends` in order, escalating only the answers that are not settled.

    Args:
        backends: Two or more backends, cheapest first.
        unsure_band: A yes/no answer with `p` inside this closed interval is escalated.
        min_confidence: A choice or rating below this confidence is escalated.

    Raises:
        PolicyError: Fewer than two backends, or a band or floor outside `[0, 1]`.
    """

    def __init__(
        self,
        *backends: Backend,
        unsure_band: tuple[float, float] = (0.2, 0.8),
        min_confidence: float = 0.7,
    ) -> None:
        if len(backends) < 2:
            raise PolicyError("A cascade needs at least two backends; use the one directly.")
        low, high = unsure_band
        if not 0.0 <= low <= high <= 1.0:
            raise PolicyError(
                f"unsure_band must be (lo, hi) with 0 <= lo <= hi <= 1, got {unsure_band!r}."
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise PolicyError(f"min_confidence must be between 0 and 1, got {min_confidence!r}.")
        self.backends = backends
        self.unsure_band = (float(low), float(high))
        self.min_confidence = float(min_confidence)
        self.answered_by: Counter[str] = Counter()
        """How many answers each model has given. The number that shows what the cascade saves."""
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        """Every stage and both thresholds, since all of them decide which answer comes back."""
        chain = " > ".join(backend.model_id for backend in self.backends)
        low, high = self.unsure_band
        return f"cascade({chain}; band={low:g}-{high:g}; min_confidence={self.min_confidence:g})"

    def settled(self, answer: Answer) -> bool:
        """Whether `answer` is decisive enough to keep without asking further."""
        if isinstance(answer, NoulAnswer):
            low, high = self.unsure_band
            return not low <= answer.p <= high
        return answer.confidence >= self.min_confidence

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """Ask each stage what the previous ones left unsettled."""
        if not questions:
            raise BackendError("A backend call needs at least one question.")
        pending = dict(questions)
        answers: dict[str, Answer] = {}
        models: dict[str, str] = {}
        tokens: int | None = None
        last = len(self.backends) - 1

        for stage, backend in enumerate(self.backends):
            if not pending:
                break
            try:
                response = backend.ask(state, pending)
            except GutError as error:
                if stage == last:
                    raise
                logger.info("gut: %s failed, escalating: %s", backend.model_id, error)
                continue
            if response.input_tokens is not None:
                tokens = (tokens or 0) + response.input_tokens
            for name in list(pending):
                answer = response.answers.get(name)
                if answer is None or not (stage == last or self.settled(answer)):
                    continue
                answers[name] = answer
                models[name] = response.model_for(name)
                del pending[name]

        if pending:
            raise BackendError(f"No stage of the cascade answered {', '.join(map(repr, pending))}.")
        with self._lock:
            self.answered_by.update(models.values())
        ordered = {name: answers[name] for name in questions}
        return BackendResponse(
            answers=ordered,
            model=models[next(iter(questions))],
            input_tokens=tokens,
            models=models,
        )
