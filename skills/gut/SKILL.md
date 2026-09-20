---
name: gut
description: Write judgment calls in Python with gut — "is this a churn threat?", "which team owns this?", "how urgent is this?" — as one readable line that can also answer UNSURE. Use when code needs a decision that is judgment rather than logic, or when replacing a hand-rolled LLM call plus a hard-coded threshold.
---

# gut

Judgment as one line of Python. `pip install "gut[jev]"`, `export TYPESAFE_API_KEY=...`.

## Start here

```python
import gut

if gut.likely(email, "the customer threatens to cancel"):
    escalate()
```

**Write this form first.** No arguments, no threshold, no prompt. Reach for anything below only when
the decision genuinely warrants it.

Three question shapes:

```python
gut.likely(ticket, "is a bug report")                    # yes / no
gut.classify(ticket, Team)                               # which one — an Enum
gut.rate(ticket, ["can wait", "this week", "today"])     # how much
```

For `classify`, the Enum member **values** are what the model is shown and the **names** come back:

```python
class Team(Enum):
    BILLING  = "invoices, charges, refunds"
    PLATFORM = "outages, latency, API errors"
    OTHER    = "anything else"        # always include a catch-all; gut warns without one
```

## Saying how careful to be

Three words. Never a number.

```python
gut.likely(email, "the customer threatens to cancel",
           stakes="high", lean="yes", ask_human=True)
```

- `lean="yes"` / `"no"` — which mistake is worse, so which way to err.
- `ask_human=True` — makes `UNSURE` possible. **Off by default**; without it there is no third branch.
- `stakes="low"|"medium"|"high"` — how bad an automatic mistake is next to a person deciding.
  Needs `ask_human=True`, and only widens the range where a person is asked.

`classify` and `rate` take `stakes` and `ask_human`, not `lean`.

## Branching

```python
import gut

decision = gut.likely(email, "the customer threatens to cancel", ask_human=True)

match decision:
    case gut.YES:    escalate()
    case gut.NO:     auto_reply()
    case gut.UNSURE: review_queue.add(email)
```

**`case gut.YES:` must be dotted.** A bare `case YES:` is a capture pattern, not a value pattern —
Python rejects it with `SyntaxError: name capture 'YES' makes remaining patterns unreachable`.
`case Outcome.YES:` also works.

`if decision:` works too. `UNSURE` then follows `gut.configure(on_unsure=...)`, which raises
`UnsureDecision` by default rather than guessing.

## Asking several things about one subject

```python
from gut import likely, semantic

@semantic
def handle(ticket):
    if likely(ticket, "is a bug report"):
        if likely(ticket, "has reproduction steps"):   # already answered
            ...
    elif likely(ticket, "asks for a refund"):          # already answered
        ...
```

One request instead of three. For `@semantic` to collect a judgment:

- the subject must be a **parameter** of the decorated function, never reassigned inside it;
- the question must be a **literal or a module-level name**, not an f-string;
- the call must not pass `backend=`, `*args` or `**kwargs`.

Anything else is asked normally — it never breaks, it just doesn't batch. `handle.gut_plan` shows
what was collected. Coroutine functions work; the fetch runs off the event loop.

No parameter to batch against? Ask explicitly:

```python
import gut

with gut.judge(ticket) as j:
    bug = j.likely("is a bug report")
    team = j.classify(Team)
    if bug:          # everything registered so far goes out here, in one request
        route(team)
```

## Exact costs — only when you know the numbers

```python
gut.likely(email, "the customer threatens to cancel",
           cost_false_yes=2, cost_false_no=50, cost_human=1)
```

**Never mix these with `stakes` / `lean` / `ask_human` in one call** — that raises `PolicyError`.

`cost_human` must be below `cost_false_yes * cost_false_no / (cost_false_yes + cost_false_no)`, or
`UNSURE` can never fire. `gut` warns; `Policy.max_useful_cost_human` gives the ceiling. The posture
words cannot hit this, which is one reason to prefer them.

## Testing

```yaml
# predicates/cancel_threat.yaml
question: "the customer threatens to cancel"
min_accuracy: 0.9
examples:
  - text: "If this happens again I'm cancelling my subscription."
    expected: yes
  - text: "How do I cancel my subscription? I want to downgrade."
    expected: no
```

```bash
gut eval predicates/                  # accuracy, Brier score, calibration error
pytest --gut-evals predicates/        # the same, as tests
```

Offline development needs no key:

```python
import gut

gut.configure(backend=gut.FakeBackend(answers={"is a bug report": 0.91}))
```

Full documentation: [`docs/`](../../docs/README.md).

## Rules of thumb

1. **Start with no arguments.** Add `ask_human=True` when a wrong answer is expensive, then `lean`,
   then `stakes`. Reach for explicit costs only when you actually know them.
2. **Never invent a threshold.** If you find yourself writing `if decision.p > 0.7`, use `lean` or
   costs instead — that is the whole point of the library.
3. **Ask the question plainly**, as a statement about the subject. The model reads it literally.
4. **Don't ask it to count, do arithmetic, or compare dates.** Compute those in Python.
5. **Pass the narrowest subject** that contains the answer; irrelevant context hurts accuracy.
6. **Don't treat a judgment over user-controlled text as an authorisation decision.** Text in the
   subject can influence the answer, and `gut` does no taint tracking.
7. **Measure before trusting a number.** `confidence` on `classify` and `rate` is a spread
   statistic, not a probability of being right, and it does not hold up on every task.
