# Getting started

[← docs index](README.md)

## Install

```bash
pip install "gut[jev]"
export TYPESAFE_API_KEY=...
```

`gut` runs on [TypeSafe AI's Jev](https://docs.typesafe.ai), a fast, cheap model that answers with
probabilities instead of prose. With the key set it builds a backend the first time you ask
something; nothing is inferred without one, because a library that starts making billable calls on
its own is not one you can reason about.

Everything below runs offline too:

```python
import gut

gut.configure(backend=gut.FakeBackend(answers={"the customer threatens to cancel": 0.83}))

email = "If this happens again I'm cancelling my subscription."
decision = gut.likely(email, "the customer threatens to cancel")

assert decision is not None
assert bool(decision) is True
assert decision.p == 0.83
```


## The three questions you can ask

```python
from enum import Enum

from gut import classify, likely, rate

likely(ticket, "is a bug report")                        # yes or no
classify(ticket, Team)                                   # which one
rate(ticket, ["can wait", "this week", "today"])         # how much


class Team(Enum):
    BILLING  = "invoices, charges, refunds"
    PLATFORM = "outages, latency, API errors"
    OTHER    = "anything else"
```

The enum member **values** are the descriptions the model is shown and the **names** are the labels
that come back, so the enum is both your type and your prompt. `gut` warns if it has no catch-all
member, because without one every subject is forced into a category even when none fits.

What comes back behaves like the answer you wanted, and carries how it was reached:

```python
d = likely(ticket, "is a bug report")
d.p          # 0.91  — the probability behind the answer
d.outcome    # YES / NO / UNSURE
d.id         # stable id for this decision *site*, for tracking it over time
d.model      # the exact model version that answered
d.source     # "backend", "cache" or "prefetch"
```

`subject` can be a `str`, a `dict`, or a `list[str]`.
