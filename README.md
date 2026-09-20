# gut

**Judgment as a programming primitive.**

Some decisions in your code aren't logic, they're judgment: *is this email a cancellation threat?*,
*which team owns this ticket?*, *how angry is this customer?* Today those get bolted on as a prompt,
a string parse, and a hard-coded `if score > 0.7`. `gut` makes them first-class: a call that returns a
typed, calibrated decision your control flow can branch on — including the branch where the answer is
*we don't know yet, ask a human*.

```python
import gut

d = gut.likely(email, "the customer threatens to cancel",
               cost_false_yes=2, cost_false_no=50, cost_human=5)

match d:
    case gut.YES:    escalate()
    case gut.NO:     auto_reply()
    case gut.UNSURE: ask_human()
```

You never wrote a threshold. You wrote what each kind of mistake *costs you* — auto-escalating a calm
customer is cheap, missing a real churn threat is expensive, a human glance is somewhere in between —
and `gut` picks the cheapest action from the probability.

> **Status: pre-alpha, under construction.** The API below is the target design. See
> [DECISIONS.md](DECISIONS.md) for design choices made along the way and the implementation order.

---

## Why not just call the model yourself?

`gut` runs on [TypeSafe AI's Jev](https://docs.typesafe.ai) — a fast, cheap "System One" model that
returns typed, calibrated answers instead of text — but the value isn't a thinner client. It's five
things that are painful to build yourself and easy to get subtly wrong:

1. **Uncertainty is part of control flow.** Every decision is `YES` / `NO` / `UNSURE`, not `True`/`False`.
2. **You write costs, not thresholds.** The runtime derives the threshold from your cost model.
3. **Batching is automatic.** Every question about the same object collapses into one model call.
4. **Semantic decisions are testable.** Example files, a pytest plugin, record/replay cassettes.
5. **Every decision is logged** with a stable site ID, so calibration can be built on real outcomes later.

`gut` stays model-agnostic: Jev is the first backend, not the only possible one.

---

## Quickstart

```bash
pip install gut          # core
pip install "gut[jev]"   # with the Jev backend
export TYPESAFE_API_KEY=...
```

With the key set, `gut` builds a Jev backend the first time you ask something. Nothing is inferred
without it — offline work is explicit:

```python
gut.configure(backend=gut.FakeBackend(answers={"is a bug report": 0.91}))
gut.configure(backend=gut.JevBackend(model="jev-1.13.0"))   # or pin a version yourself
```

Three primitives:

```python
from gut import likely, classify, rate
from enum import Enum

# Boolean judgment, backed by Jev's Noul
d = likely(ticket, "is a bug report", cost_false_yes=1, cost_false_no=10, cost_human=3)
d.p          # 0.83  — probability of yes
d.outcome    # YES / NO / UNSURE
d.id         # stable decision-site ID
d.model      # the exact model version that answered

# Categorical judgment, backed by Choice
class Team(Enum):
    BILLING = "questions about invoices, charges, refunds"
    PLATFORM = "outages, latency, API errors"
    OTHER    = "anything else"

c = classify(ticket, Team, min_confidence=0.7)
c.value          # Team.BILLING, or UNSURE below min_confidence
c.probabilities  # {Team.BILLING: 0.81, ...}

# Ordinal judgment, backed by Score
r = rate(ticket, ["calm", "annoyed", "angry", "threatening to leave"])
r.score          # 2.4 — can land between levels
```

`subject` can be a `str`, a `dict`, or a `list[str]`; it is passed through as the model's state.

---

## The cost rule

Thresholds are arbitrary. Costs are something you actually know. Given a calibrated `p = P(yes)`:

```
expected cost of saying YES        = (1 - p) · cost_false_yes
expected cost of saying NO         =      p  · cost_false_no
expected cost of asking a human    =         cost_human
```

`gut` picks the cheapest of the three. Ties prefer `UNSURE`, then `NO` — the conservative order.

With no human in the loop (`cost_human=None`), `UNSURE` is impossible and this reduces to the
familiar decision threshold:

```
YES  ⟺  p > cost_false_yes / (cost_false_yes + cost_false_no)
```

So the earlier example — `cost_false_yes=2, cost_false_no=50` — is a threshold of `2/52 ≈ 0.038`,
with a human consulted whenever `cost_human=5` beats both. You would not have guessed `0.038`.

Escape hatches exist (`threshold=`, `unsure_band=(lo, hi)`) but costs are the documented default.
Given nothing at all, `gut` uses symmetric costs and no human option: `p > 0.5`.

---

## Truthiness and `match`

`if likely(...)` works. `YES` is truthy, `NO` is falsy, and `UNSURE` is governed by an explicit policy
rather than a silent guess:

```python
import gut
gut.configure(on_unsure="raise")   # default: raises UnsureDecision
gut.configure(on_unsure="false")   # or "true", or a callback

with gut.on_unsure("false"):       # local override
    ...
```

The default is `"raise"` on purpose: an unhandled `UNSURE` silently collapsing to `False` is exactly
the bug this library exists to prevent.

The outcomes are **dotted on purpose**. A bare `case YES:` is not a value pattern in Python — it is a
capture pattern that matches anything, and the compiler rejects it outright:

```
SyntaxError: name capture 'YES' makes remaining patterns unreachable
```

So `gut` exports the outcomes as members of an `Outcome` enum, and either dotted spelling works:

```python
import gut
match d:
    case gut.YES: ...

from gut import Outcome
match d:
    case Outcome.YES: ...
```

`Decision.__eq__` compares against outcomes, so you can match the decision itself rather than reaching
for `d.outcome`.

---

## Batching

Billing is input-only, so extra questions in the same call are nearly free — but only if they're in
the *same call*. `@semantic` makes that automatic:

```python
from gut import semantic, likely

@semantic
def handle(ticket):
    if likely(ticket, "is a bug report"):
        if likely(ticket, "has reproduction steps"):   # prefetched, no second call
            ...
    elif likely(ticket, "asks for a refund"):          # prefetched
        ...
```

The decorator parses the function once with `ast`, finds the decisions whose subject is a parameter
and whose question is a literal, and issues **one backend call per subject** before the body runs.
Anything it can't prove statically falls back to a lazy single call — it never changes what your code
does, only how many round trips it takes.

It is also **speculative**: a question behind a branch that never runs is still asked. That is the
trade, and it is usually the right one — the state is paid for once per request, so five questions in
one call cost barely more than one, while five calls cost five states.

If you'd rather not have magic, ask explicitly:

```python
with gut.judge(ticket) as j:
    bug = j.likely("is a bug report")
    team = j.classify(Team)
    if bug:          # <- everything registered so far goes out in one request, here
        route(team)  # <- already answered
```

`j.likely(...)` registers a question and hands back a handle; nothing is sent. The first time any
handle is **used** — tested for truth, matched, compared, or read for a field — every question
registered up to that moment goes out in one request. Questions registered afterwards form the next
group.

So where you first read decides what got batched with what. Register everything you might need
before reading any of it. Two details that follow from the rule: `repr()` never resolves, because a
debugger must not cost a request; and leaving the block resolves nothing, so a question nobody reads
is never asked and never billed.

---

## Caching

Asking the same question about the same state twice buys nothing and is billed again, so answers are
cached by default — keyed on the state, the question *and the model asked for*.

```python
gut.configure(cache=gut.SQLiteCache(".gut_cache/answers.db"))  # survives restarts
gut.configure(cache=gut.NullCache())                           # off
```

One caveat worth knowing: if your backend names its model by a moving alias, the key does not change
when the alias moves, so the cache can keep serving answers from the previous version. Pin a version
if that matters. `Decision.model` always reports the version that actually answered, cache hit or
not, so a stale entry is at least visible in the record.

---

## Testing semantic decisions

Judgment is testable like anything else. You cannot assert an exact output, but you can assert that
a predicate agrees with you on cases you have written down, often enough to rely on.

```yaml
# predicates/cancel_threat.yaml
question: "the customer threatens to cancel"
min_accuracy: 0.9
examples:
  - text: "If this happens again I'm cancelling my subscription."
    expected: yes
  - text: "How do I cancel my subscription? I want to downgrade."
    expected: no
  - text: "Third outage this month. We're evaluating alternatives."
    expected: yes
```

```bash
pytest --gut-evals predicates/
```

```
---------------------------------- gut evals -----------------------------------
PASS  cancel_threat                      100%  (5/5) min 90%
FAIL  owning_team                         67%  (2/3) min 80%
PASS  urgency                            100%  (2/2) min 60%
```

Each file is one test, because the unit that passes or fails is the file's accuracy. A failure
prints every case that went the wrong way, with what the model actually said:

```
owning_team: accuracy 67% (2/3), below min_accuracy 80%
  question: which team should own this ticket

  expected 'OTHER', got BILLING (confidence=0.65)
    Do you have a student discount?
```

The hard cases are the point. A file of obvious examples proves nothing; the ones worth writing down
are the ones you had to think about, and the ones that went wrong in production. Choice and score
files work the same way, with `options:` or `levels:` instead of a bare question.

`min_accuracy` defaults to `1.0` — lowering it should be a decision you made, not one the library
made quietly on your behalf.

### Record once, replay forever

Evals against a live model are slow and cost money on every run. Record them once:

```python
# conftest.py
import gut
from gut import CassetteBackend, JevBackend, record_requested

inner = JevBackend(model="jev-1.13.0") if record_requested() else None
backend = CassetteBackend("cassettes/predicates.json", inner, model="jev-1.13.0")
gut.configure(backend=backend)

def pytest_sessionfinish(session, exitstatus):
    backend.save()
```

```bash
GUT_RECORD=1 pytest --gut-evals predicates/   # 4.05s, against the real model
pytest --gut-evals predicates/                # 0.07s, offline, identical
```

Commit the cassette. Entries are keyed by the state and the question, not by the request they
travelled in, so regrouping questions — adding `@semantic`, say — does not invalidate a recording.
In replay mode an unrecorded question is an error that names itself; quietly reaching for the network
would turn one forgotten re-record into a suite that passes on your laptop, fails in CI, and bills
you either way.

`FakeBackend` covers development with no API key at all, and the entire test suite of this project
passes without one.

---

## The decision log

Every decision can be recorded as it is made: the site id, the probability, the costs applied, the
exact model version that answered, where it came from, and where in your code it was made.

```python
gut.configure(sink=gut.JSONLSink("decisions.jsonl"))
```

```json
{"type":"decision","id":"079e4383c5e05c1f","kind":"noul","outcome":"yes","model":"jev-1.13.0",
 "source":"backend","p":0.83,"costs":{"cost_false_yes":2.0,"cost_false_no":50.0,"cost_human":5.0},
 "question":{"type":"noul","instructions":"the customer threatens to cancel"},
 "site":{"module":"app.inbox","function":"handle","file":"app/inbox.py","line":41}}
```

When you find out what actually happened, say so:

```python
decision.resolve(actual=True, note="customer did churn")
```

Nothing consumes resolutions yet — there is no calibration in this release. The point is that the
data path exists from the start, so the first question anyone asks of a system like this — *is it
actually calibrated on my data?* — is answerable from logs that were already being written, instead
of from an instrumentation project begun after the doubt arrives.

Nothing is recorded unless you ask. Logging is observability, never correctness: a sink that raises
is reported and swallowed, because a broken log must not break a decision.

---

## Honest limitations

- **Jev reads instructions literally** and is weak at counting, arithmetic, and date comparison. Don't
  ask it "did this arrive more than 30 days ago" — compute that in Python and let `gut` judge the rest.
- **Irrelevant state hurts accuracy.** Pass the narrowest subject that contains the answer.
- **State is untrusted input.** Text inside the subject can influence the answer. `gut` does not do
  taint tracking yet; treat a decision over user-controlled text as advisory in security contexts.
- **"Calibrated" is the model's claim, not a guarantee** for *your* data. The cost rule is only as good
  as the probabilities feeding it. `gut` logs every decision and supports `decision.resolve(actual=...)`
  precisely so you can check this later — but it does not self-calibrate yet.
- **Context limits** are 64k tokens for state plus all questions, 32k for state plus the single
  longest question. Large batches are split automatically.
- **Model versions drift.** `jev-latest` moves. Pin a version; `gut` records the exact model ID that
  answered every decision.

---

## License

Apache-2.0
