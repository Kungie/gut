# Async

[← docs index](README.md)

Every way of asking has an awaitable twin, for code that runs on an event loop -- a FastAPI handler,
a Discord bot, an agent loop:

```python
import asyncio

import gut


async def handle(email):
    if await gut.alikely(email, "the customer threatens to cancel"):
        return "escalate"
    team = await gut.aclassify(email, Team)
    urgency = await gut.arate(email, ["can wait", "this week", "right now"])
    return team.value, urgency.nearest_level


asyncio.run(handle(email))
```

The arguments and the decisions are exactly those of `likely`, `classify` and `rate` -- the same
posture words, the same cache, the same `on_decision` hook. What changes is that nothing blocks the
loop while the model answers.

## Many subjects

```python
import asyncio

import gut


async def triage(comments, tickets):
    spam = await gut.each(comments).alikely("is spam")
    teams = await gut.each(tickets).aclassify(Team)
    return spam, teams


asyncio.run(triage(comments, tickets))
```

Or, for unrelated questions, the usual `asyncio.gather`:

```python
import asyncio

import gut


async def both(email, ticket):
    return await asyncio.gather(
        gut.alikely(email, "the customer threatens to cancel"),
        gut.aclassify(ticket, Team),
    )


asyncio.run(both(email, ticket))
```

## Several questions about one subject

`@gut.semantic` works on coroutine functions, and collects awaited judgments as well as plain ones:

```python
import asyncio

import gut


@gut.semantic
async def route(ticket):
    if await gut.alikely(ticket, "is a bug report"):      # all three were asked together,
        if await gut.alikely(ticket, "is urgent"):         # before the body ran
            return "on-call"
        return "engineering"
    if await gut.alikely(ticket, "asks for a refund"):
        return "billing"
    return "support"


asyncio.run(route(ticket))
```

With `judge()`, reading a handle cannot await -- `if bug:` is synchronous -- so resolve first:

```python
import asyncio

import gut


async def with_a_judge(ticket):
    with gut.judge(ticket) as j:
        bug = j.likely("is a bug report")
        team = j.classify(Team)
    await j.aresolve()        # one request, on the event loop
    return bug.outcome, team.value   # already answered: nothing blocks


asyncio.run(with_a_judge(ticket))
```

## What happens underneath

| backend | awaited how |
|---|---|
| `JevBackend` | natively, through the SDK's own `AsyncTypeSafeClient` |
| `OpenAICompatibleBackend` | natively, through an `httpx.AsyncClient`, `max_concurrency` requests at a time |
| `Cascade` | stage by stage, each one natively where it can be |
| `ZeroShotBackend`, `TransformersBackend` | in a worker thread: the work is on the CPU or GPU either way |
| `FakeBackend`, your own backend | in a worker thread, unless it defines `aask` |

Async does not make a model answer faster: it lets the rest of your program keep going while it
answers. Whether requests in flight together also *finish* sooner is up to the server -- Jev and
hosted APIs handle many at once; a local Ollama on a small laptop tends to take them one by one.

A backend's clients belong to the event loop they were made on. Each loop that uses a backend gets
its own, and `await backend.aclose()` (or `async with backend:`) closes them.
