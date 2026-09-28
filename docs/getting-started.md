# Getting started

[← docs index](README.md)

## Install

```bash
pip install "gutfeel[jev]"           # TypeSafe AI's Jev -- the model gut is built around
pip install "gutfeel[local]"         # + models that run in your process (PyTorch)
pip install gutfeel                  # core only: any OpenAI-compatible server, and FakeBackend
```

The distribution is called `gutfeel` because `gut` was already taken on PyPI. Everything else --
the import, the module, every name in these docs -- is `gut`.

## Pick a model

`gut` does not come with a model; it makes whichever one you choose answer like a function. Pick
one once, at startup. The one it is designed around is Jev:

```python
import gut

gut.configure(backend=gut.JevBackend())      # needs TYPESAFE_API_KEY
```

Jev answers typed questions natively, so `gut` hands it each question as it is and reads the
probabilities straight back. No key, or no network? A 70M-parameter NLI model runs on your own CPU:

```python
import gut

gut.configure(backend=gut.ZeroShotBackend())
```

It downloads once (about 150 MB), answers in about a tenth of a second per question, costs nothing
per call, and never sends your data anywhere. [Backends](backends.md) covers the rest: Jev through
OpenRouter, open decision models on your machine through Ollaya, a small language model, Ollama,
vLLM, OpenAI, and a `Cascade` that combines them.

Nothing is chosen for you. With no backend configured, `gut` raises and lists the options -- except
that setting `TYPESAFE_API_KEY` selects Jev, since that variable has no other use.

For tests and offline work, `FakeBackend` answers from fixtures and never touches a model:

```python
import gut

gut.configure(backend=gut.FakeBackend(answers={"the customer threatens to cancel": 0.83}))

email = "If this happens again I'm cancelling my subscription."
decision = gut.likely(email, "the customer threatens to cancel")

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
member -- `OTHER`, `NONE`, `DIGER` and their kin -- because without one every subject is forced into
a category even when none fits.

What comes back behaves like the answer you wanted, and carries how it was reached:

```python
d = likely(ticket, "is a bug report")
d.p          # 0.91  — the probability behind the answer
d.outcome    # YES / NO / UNSURE
d.model      # the exact model version that answered
d.source     # "backend", "cache" or "prefetch"
d.to_dict()  # all of it, ready for a log line
```

`subject` can be a `str`, a `dict`, or a `list` -- anything JSON can hold. A structured subject is
shown to the model as JSON.

## Writing questions a small model can answer

- **State one claim.** `"is spam"`, `"asks for a refund"`, `"the customer threatens to cancel"`. A
  bare predicate like `"is spam"` is read as a claim about the subject.
- **One idea per question.** An NLI model reads "is spam or abusive" poorly; ask two questions and
  batch them (see [Asking everything at once](batching.md)).
- **Let Python do the arithmetic.** Counting, dates and comparisons belong in code.
- **Pass the narrowest subject that contains the answer.** Irrelevant text is noise to any model,
  and the smaller the model, the more it hurts.
