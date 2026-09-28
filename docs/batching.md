# Asking everything at once

[← docs index](README.md)


## Many subjects, one question

```python
import gut

spam = gut.each(comments).likely("is spam")          # a list of decisions, in order
teams = gut.each(tickets).classify(Team, ask_human=True)

for comment, decision in zip(comments, spam):
    ...
```

Each decision is exactly what `gut.likely(comment, "is spam")` would have returned -- the same
posture words, the same `on_decision` hook -- with `source="batch"`. How the subjects travel is the
backend's business:

- **Jev**, and any other server, gets one request per subject, eight in flight at a time
  (`gut.each(..., concurrency=16)` to change that). Jev's SDK backs off on its own if it hits a
  rate limit.
- **A local model** runs the subjects through batched forward passes. On one laptop the NLI model
  answered 200 comments in 5.2 s through `each()` against 15.9 s in a loop; Qwen3-0.6B on the GPU
  gained little, since it is limited by compute either way.
- **A `Cascade`** sends every subject to its cheap stage and only the unsettled ones onwards, as one
  batch.

Subjects already in the cache are not asked again, and a subject that appears twice is asked once.

## One subject, many questions

The expensive part of a judgment is reading the subject, not the question. Hand a backend every
question about one subject at once and it reads the subject once: Jev bills it once per request, a
local model computes it once and answers every question from there, and a server with prefix caching
does the same on its side. Ten judgments together cost about what one costs; ten separate calls cost
ten subjects. `@semantic` makes that automatic:

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
question is knowable up front, and hands them all to the backend **together** before the body runs.

**It cannot change what your code does.** Anything the analysis can't prove statically is left
alone and asked normally. Underneath that, the prefetch is keyed by a fingerprint of the *real*
question and the *real* subject, so even a plan that guessed wrong just fails to match. Being wrong
costs a wasted request, never a wrong answer.

**It is speculative.** Questions behind branches that never run are still asked. Usually the right
trade; if a question is expensive for reasons other than the subject, keep it out.

**Coroutines work too.** The prefetch is awaited -- natively for Jev and the OpenAI-compatible
backend, from a worker thread otherwise -- and the body is then answered from memory, so it never
blocks the loop. `await gut.alikely(...)` in the body is collected just like `likely(...)`. See
[Async](async.md).

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

Posture words work inside the block as they do outside it:

```python
import gut

with gut.judge(ticket) as j:
    leaving = j.likely("the customer threatens to cancel", lean="yes", ask_human=True)
    team = j.classify(Team, stakes="high", ask_human=True)
```

With a [`Cascade`](backends.md#cascade), a batch is split by how sure each stage was: the cheap
model's settled answers stay, and only the rest travel on -- still together.
