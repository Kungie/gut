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

Why any of this is practical at all is [its own page](why-jev.md).
