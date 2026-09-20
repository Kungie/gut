# Caching, logging, and backends

[← docs index](README.md)


Answers are cached by default, keyed on the subject, the question and the model asked for:

```python
import gut

gut.configure(cache=gut.SQLiteCache(".gut_cache/answers.db"))  # survives restarts
gut.configure(cache=gut.NullCache())                           # off
```

If your backend names its model by a moving alias the key does not change when the alias moves, so
the cache can serve answers from the previous version. Pin a version if that matters.
`Decision.model` always reports the version that actually answered.

Every decision can be recorded as it is made:

```python
import gut

gut.configure(sink=gut.JSONLSink("decisions.jsonl"))
```

```json
{"type":"decision","id":"079e4383c5e05c1f","kind":"noul","outcome":"yes","model":"jev-1.13.0",
 "source":"backend","p":0.83,"costs":{"cost_false_yes":2.5,"cost_false_no":2.5,"cost_human":1.0},
 "question":{"type":"noul","instructions":"the customer threatens to cancel"},
 "site":{"module":"app.inbox","function":"handle","file":"app/inbox.py","line":41}}
```

When you learn what actually happened, say so with `decision.resolve(actual=True)`. Nothing consumes
resolutions yet — there is no calibrator in this release. The point is that the data path exists
from the start, so *is this calibrated on my data?* is answerable from logs already being written.

Nothing is logged unless you ask, and a sink that raises is reported and swallowed: a broken log
must not break a decision.

## Backends

`Backend` is a protocol — `ask()` and `model_id`. `JevBackend` is the real one, `FakeBackend` is for
development, `CassetteBackend` replays recordings. `import gut` never pulls in the vendor SDK, and
nothing above this line knows which backend answered.

### Why Jev makes this worth doing

`gut` would work on an LLM. It would just be too slow and too expensive to use the way it is meant
to be used — a judgment inside an `if`, several per request, everywhere.

TypeSafe's own published figures for Jev, which are **self-reported vendor benchmarks and should be
read as such**:

| | Jev | frontier LLMs |
|---|---|---|
| end-to-end response | 70–500 ms | 3–329 s |
| speed, like for like | “40x-200x faster for the same levels of frontier intelligence for System One shaped queries” | |
| input tokens | $0.042 / MTok | $0.20 – $10 / MTok |
| output tokens | free | roughly 5x input |

Their homepage quotes `193.6x faster` and `444.6x cheaper` from workflow evaluations, with the
caveat — theirs, not ours — that those are "on the higher end of real world gains".

Two structural differences matter more than the multipliers. Jev emits all its probabilities in
parallel rather than generating tokens one at a time, which is where the latency goes; and it
returns typed answers, so there is nothing to parse and nothing to hallucinate. Billing on input
only is what makes [batching](batching.md) nearly free.

None of that is `gut`'s claim to verify, and none of it is load-bearing for the design. Jev is why
this is practical now, not what the library is.
