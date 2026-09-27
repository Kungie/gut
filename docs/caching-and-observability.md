# Caching and observability

[← docs index](README.md)

## The cache

Answers are cached by default, keyed on the subject, the question and the model asked for, so asking
the same thing twice costs one model call:

```python
import gut

gut.configure(cache=gut.SQLiteCache(".gut_cache/answers.db"))  # survives restarts
gut.configure(cache=gut.NullCache())                           # off
gut.configure(cache=gut.MemoryCache(maxsize=10_000))           # the default, with a bigger bound
```

The key uses the model's *configured* name, because it is needed before the call. If that name is a
moving alias -- `jev-latest`, an Ollama tag you re-pull -- the cache can serve an answer from the
previous version. Pin a version if that matters. `decision.model` always reports the version that
actually answered: local models report the exact Hugging Face commit, `name@1a2b3c4d5e6f`.

A `Cascade`'s key includes every stage and both thresholds, so changing its band never serves an
answer that was settled under the old one.

## Seeing every decision

One hook, called with every decision as it is made:

```python
import json
import logging

import gut

logger = logging.getLogger("app")

gut.configure(on_decision=lambda d: logger.info("gut %s", json.dumps(d.to_dict())))
```

`to_dict()` is the whole decision as JSON-ready data:

```json
{"kind": "likely", "id": "079e4383c5e05c1f", "outcome": "yes",
 "question": "the customer threatens to cancel", "model": "Qwen/Qwen3-0.6B@c1899de289a0",
 "source": "backend", "latency_ms": 412.7, "p": 0.83, "policy": "yes above 0.5, no at or below it"}
```

`classify` adds the chosen member, its confidence and the whole distribution; `rate` adds the score
and the nearest level. `id` is stable for *this question asked from this function*, across runs and
deploys, so a log can be grouped by the decision that produced it. `source` says whether the answer
came from its own call, the cache, or a batch fetched by `@semantic` or `judge()`.

Nothing is observed unless you ask, and a hook that raises is logged and swallowed: a broken log
line must never break a decision.

## Backends

What answers a question is [its own page](backends.md).
