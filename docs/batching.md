# Asking everything at once

[← docs index](README.md)


Billing is on input, so the subject is paid for once per request however many questions ride along.
Ten judgments in one call cost about what one costs; ten calls cost ten subjects. `@semantic` makes
that automatic:

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

It reads the function's source once, finds the judgments whose subject is a parameter and whose
question is knowable up front, and asks them all in **one request** before the body runs.

**It cannot change what your code does.** Anything the analysis can't prove statically is left
alone and asked normally. Underneath that, the prefetch is keyed by a fingerprint of the *real*
question and the *real* subject, so even a plan that guessed wrong just fails to match. Being wrong
costs a wasted request, never a wrong answer.

**It is speculative.** Questions behind branches that never run are still asked. Usually the right
trade; if a question is expensive for reasons other than tokens, keep it out.

**Coroutines work too.** The prefetch is the only call that touches the network, so it runs in a
worker thread and is awaited; the body is then answered from memory and never blocks the loop.
Three judgments about one ticket become one request, and concurrent handlers overlap instead of
queueing.

If you'd rather place the batch by hand:

```python
import gut

with gut.judge(ticket) as j:
    bug = j.likely("is a bug report")
    team = j.classify(Team)
    if bug:          # <- everything registered so far goes out in one request, here
        route(team)  # <- already answered
```

Registering sends nothing; the first time any handle is **used** — tested, matched, compared, read —
everything registered so far goes out together. So where you first read decides what got batched
with what. `repr()` never resolves, because a debugger must not cost a request, and leaving the
block resolves nothing, so a question nobody reads is never asked.
