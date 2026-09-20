# gut

**Judgment as a programming primitive.**

Some decisions in your code aren't logic, they're judgment: *is this email a cancellation threat?*,
*which team owns this ticket?*, *how angry is this customer?* Today those get bolted on as a prompt,
a string parse, and a hard-coded `if score > 0.7`. `gut` makes them first-class: a call that returns
a typed decision your control flow can branch on — including the branch where the answer is *we
don't know, ask a human*.

```python
import gut

d = gut.likely(email, "the customer threatens to cancel",
               cost_false_yes=2, cost_false_no=50, cost_human=1)

match d:
    case gut.YES:    escalate()
    case gut.NO:     auto_reply()
    case gut.UNSURE: ask_human()
```

You never wrote a threshold. You wrote what each kind of mistake *costs you* — auto-escalating a calm
customer is cheap, missing a real churn threat is expensive, a human glance is somewhere in between —
and `gut` picks the cheapest action from the probability. For those numbers that works out to:
auto-reply below `0.02`, escalate above `0.5`, ask a person in between. Nobody draws that band by
hand.

> **Status: pre-1.0.** Everything documented here works and is covered by tests. Calibration from
> resolved outcomes, durable execution, and taint tracking are deliberately out of scope for now;
> see [DECISIONS.md](DECISIONS.md) for what was decided and why.

---

## Why not just call the model yourself?

`gut` runs on [TypeSafe AI's Jev](https://docs.typesafe.ai) — a fast, cheap "System One" model that
returns typed answers with probabilities instead of text — but the value isn't a thinner client.
It's five things that are painful to build yourself and easy to get subtly wrong:

1. **Uncertainty is part of control flow.** Every decision is `YES` / `NO` / `UNSURE`, not
   `True`/`False`.
2. **You write costs, not thresholds.** The runtime derives the threshold from your cost model — and
   tells you when your cost model has a branch that can never fire.
3. **Batching is automatic.** Every question about the same object collapses into one model call.
4. **Semantic decisions are testable.** Example files, a pytest plugin, record/replay cassettes.
5. **Every decision is logged** with a stable id, so calibration can be built on real outcomes later.

`gut` stays model-agnostic: Jev is the first backend, not the only possible one.

---

## Install

```bash
pip install gut          # core
pip install "gut[jev]"   # with the Jev backend
export TYPESAFE_API_KEY=...
```

With the key set, `gut` builds a Jev backend the first time you ask something. Nothing is inferred
without it — a library that starts making billable calls on its own is not one you can reason about.

## Quickstart

This runs offline, with no API key:

```python
import gut

gut.configure(backend=gut.FakeBackend(answers={"the customer threatens to cancel": 0.83}))

email = "If this happens again I'm cancelling my subscription."
decision = gut.likely(
    email,
    "the customer threatens to cancel",
    cost_false_yes=2,
    cost_false_no=50,
    cost_human=1,
)

assert decision.outcome is gut.YES
assert decision.p == 0.83
assert decision.policy.implied_threshold is None  # a human is available, so it is not one number
```

Delete the `configure` line, set `TYPESAFE_API_KEY`, and the same code asks the real model.

## The three primitives

```python
from enum import Enum

from gut import classify, likely, rate

# Boolean judgment, backed by Jev's Noul
d = likely(ticket, "is a bug report", cost_false_yes=1, cost_false_no=10, cost_human=0.5)
d.p          # 0.91  — probability of yes
d.outcome    # YES / NO / UNSURE
d.id         # stable decision-site id
d.model      # the exact model version that answered
d.source     # "backend", "cache" or "prefetch"

# Categorical judgment, backed by Choice
class Team(Enum):
    BILLING  = "questions about invoices, charges, refunds"
    PLATFORM = "outages, latency, API errors"
    OTHER    = "anything else"

c = classify(ticket, Team, min_confidence=0.7)
c.value          # Team.BILLING, or UNSURE below min_confidence
c.probabilities  # {Team.BILLING: 0.81, ...}

# Ordinal judgment, backed by Score
r = rate(ticket, ["calm", "annoyed", "angry", "threatening to leave"])
r.score          # 2.4 — can land between levels
r.nearest_level  # 2
```

The enum member **values** are the descriptions the model is shown; the **names** are the labels
that come back. So the enum is both your type and your prompt. `gut` warns if it has no catch-all
member, because without one every subject is forced into a category even when none fits.

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

So `cost_false_yes=2, cost_false_no=50` is a threshold of `2/52 ≈ 0.038`. You would not have guessed
`0.038`.

### The trap the formula hides

The expected cost of asking a person is **flat** in `p`, while the cheaper of yes and no *peaks*
where those two lines cross. Put `cost_human` above that peak and there is no probability at all
where a person is worth asking — the third branch you carefully wrote is unreachable, silently. The
ceiling is:

```
cost_false_yes · cost_false_no / (cost_false_yes + cost_false_no)
```

which for `2` and `50` is `1.92`. A `cost_human` of `5` would never fire; `1` does. `gut` warns when
you cross it, and `Policy.max_useful_cost_human` will tell you where it is. This is not theoretical:
it is exactly the mistake the demo shipped with on its first run, and the report showed it as *zero
tickets sent to a human* with no error anywhere.

Escape hatches exist (`threshold=`, `unsure_band=(lo, hi)`) but costs are the documented default.
Given nothing at all, `gut` uses symmetric costs and no human option: `p > 0.5`.

---

## Truthiness and `match`

`if likely(...)` works. `YES` is truthy, `NO` is falsy, and `UNSURE` is governed by an explicit
policy rather than a silent guess:

```python
import gut

gut.configure(on_unsure="raise")   # default: raises UnsureDecision
gut.configure(on_unsure="false")   # or "true", or a callback

with gut.on_unsure("false"):       # local override, follows async tasks, never leaks across threads
    ...
```

The default is `"raise"` on purpose. Coercing `UNSURE` to `False` is the most dangerous option
available: it is what your existing `if` already does, so the third branch would vanish down the
"no" path — the exact bug this library exists to prevent, reintroduced as a default.

The outcomes are **dotted on purpose**. A bare `case YES:` is not a value pattern in Python — it is
a capture pattern that matches anything, and the compiler rejects it outright:

```
SyntaxError: name capture 'YES' makes remaining patterns unreachable
```

So `gut` exports the outcomes as members of an `Outcome` enum, and either dotted spelling works:

```python
import gut
from gut import Outcome

match decision:
    case gut.YES: ...

match decision:
    case Outcome.YES: ...
```

`Decision.__eq__` compares against outcomes, so you can match the decision itself rather than
reaching for `d.outcome`.

---

## Batching

Billing is input-only, so the state is paid for once per request however many questions ride along.
Five questions in one call cost barely more than one; five calls cost five states. `@semantic` makes
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

The decorator reads the function's source once with `ast`, finds the decisions whose subject is a
parameter and whose question is knowable at decoration time, and issues **one request per subject**
before the body runs.

**It cannot change what your code does.** Anything the analysis can't prove statically is not
collected, and that call goes to the backend on its own exactly as it would undecorated. Underneath
that, the prefetch is keyed by a fingerprint of the *real* question and the *real* subject, so even
a plan that guessed wrong just fails to match and the ordinary path runs. Being wrong costs a wasted
request, never a wrong answer.

**It is speculative.** Questions behind branches that never run are still asked. That is the trade,
and usually the right one — but if a question is expensive for reasons other than tokens, keep it out
of a decorated function.

If you'd rather not have magic, ask explicitly:

```python
import gut

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

So where you first read decides what got batched with what. Two details follow from the rule:
`repr()` never resolves, because a debugger must not cost a request; and leaving the block resolves
nothing, so a question nobody reads is never asked and never billed.

---

## Caching

Asking the same question about the same state twice buys nothing and is billed again, so answers are
cached by default — keyed on the state, the question *and the model asked for*.

```python
import gut

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
exact model version that answered, where the answer came from, and where in your code it was made.

```python
import gut

gut.configure(sink=gut.JSONLSink("decisions.jsonl"))
```

```json
{"type":"decision","id":"079e4383c5e05c1f","kind":"noul","outcome":"yes","model":"jev-1.13.0",
 "source":"backend","p":0.83,"costs":{"cost_false_yes":2.0,"cost_false_no":50.0,"cost_human":1.0},
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

## What this looks like on real data

[`examples/support_tickets/`](examples/support_tickets/) has 101 labelled support tickets, 26 of them
deliberately ambiguous, and the same triage handler written with and without `gut`. Against the real
model:

```
resolved automatically               40  40%
sent to a human                      61  60%
wrong automatic decisions             0

requests                            101  1.0 per ticket
judgments                           505  5.0 per request

the ambiguous ones
  marked hard                        26
  sent to a human                    22  85% of them

same answers, a hard 0.7 threshold and no third branch
  churn risks missed                  8  vs 0
  total cost                        160  vs 61
  requests                          505  vs 101
```

The model is uncertain in almost exactly the places the dataset marks ambiguous. Without a third
branch that uncertainty has nowhere to go and becomes a confident guess.

---

## Honest limitations

**About the model.** Jev [reads instructions literally](https://docs.typesafe.ai) and is weak at
counting, arithmetic, and date comparison. Don't ask it "did this arrive more than 30 days ago" —
compute that in Python and let `gut` judge the rest. Irrelevant state hurts accuracy, so pass the
narrowest subject that contains the answer. Multi-hop reasoning degrades. Text inside the subject can
influence the answer, and `gut` does not do taint tracking: treat a decision over user-controlled
text as advisory in security contexts.

**"Calibrated" is the model's claim, not a guarantee** for *your* data. The cost rule is only as good
as the probabilities feeding it. `gut` logs every decision and supports `decision.resolve(actual=...)`
precisely so you can check this later — but it does not self-calibrate, and nothing in this release
reads those resolutions.

**`confidence` is not an accuracy estimate.** For `classify` and `rate` it is a statistic computed
from how peaked the answer's own distribution is. `min_confidence` is a spread filter, not a
probability of being right. Treat it as a flag for review.

**Probabilities across questions are not comparable.** The vendor's own docs warn that negated
questions need not sum to 1 and that different primitives yield non-comparable numbers. `gut` never
synthesises `P(no)` from a separately asked negated question, and neither should you.

**Decision ids are stable against edits, not against moves.** An id is built from the question plus
the module and function it is asked in, so inserting lines above a call doesn't reset its history —
but moving the call to a different function does. File and line are recorded alongside for debugging
and are deliberately not part of the identity.

**`@semantic` is speculative and synchronous.** It asks questions behind branches that never run, and
it declines to decorate `async` functions rather than quietly blocking an event loop.

**Context limits.** 64k tokens for state plus every question, 32k for state plus the single longest
question. Large batches are split automatically, using a character-count estimate rather than a real
tokeniser — a wrong estimate costs an extra request or a rejected one, never a wrong answer.

---

## Development

```bash
uv sync --all-extras
uv run pytest
uv run mypy
uv run ruff check . && uv run ruff format --check .
uv run coverage run -m pytest && uv run coverage report
```

The suite runs offline and does not need an API key. One live integration test is skipped unless
`TYPESAFE_API_KEY` is set. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0
