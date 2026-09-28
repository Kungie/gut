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
from collections.abc import Mapping, Sequence

from gut._backends._many import Item, ask_all
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
        return self.ask_many([(state, questions)])[0]

    def ask_many(self, items: Sequence[Item]) -> list[BackendResponse]:
        """The same, for many subjects: each stage sees every subject it still has questions for.

        So over a thousand comments the cheap model reads all thousand, and the next one reads only
        the ones it could not settle -- in one batch, by its own fastest route. A stage that fails
        hands the whole batch on.
        """
        if any(not questions for _, questions in items):
            raise BackendError("A backend call needs at least one question.")
        pending = [dict(questions) for _, questions in items]
        answers: list[dict[str, Answer]] = [{} for _ in items]
        models: list[dict[str, str]] = [{} for _ in items]
        tokens: list[int | None] = [None for _ in items]
        last = len(self.backends) - 1

        for stage, backend in enumerate(self.backends):
            open_items = [index for index, left in enumerate(pending) if left]
            if not open_items:
                break
            try:
                responses = ask_all(backend, [(items[i][0], pending[i]) for i in open_items])
            except GutError as error:
                if stage == last:
                    raise
                logger.info("gut: %s failed, escalating: %s", backend.model_id, error)
                continue
            for index, response in zip(open_items, responses, strict=True):
                if response.input_tokens is not None:
                    tokens[index] = (tokens[index] or 0) + response.input_tokens
                for name in list(pending[index]):
                    answer = response.answers.get(name)
                    if answer is None or not (stage == last or self.settled(answer)):
                        continue
                    answers[index][name] = answer
                    models[index][name] = response.model_for(name)
                    del pending[index][name]

        unanswered = sorted({name for left in pending for name in left})
        if unanswered:
            raise BackendError(
                f"No stage of the cascade answered {', '.join(map(repr, unanswered))}."
            )
        with self._lock:
            for per_item in models:
                self.answered_by.update(per_item.values())
        return [
            BackendResponse(
                answers={name: answers[index][name] for name in questions},
                model=models[index][next(iter(questions))],
                input_tokens=tokens[index],
                models=models[index],
            )
            for index, (_, questions) in enumerate(items)
        ]
