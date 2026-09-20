# gut

**A gut feeling that knows when to ask.**

Some decisions in code are judgment, not logic: *is this customer about to leave?*, *which team owns
this?*, *is this a bug report?* `gut` lets you write one of those as one line that reads like
English — and, unlike a real gut feeling, it can tell you when it doesn't know.

```python
import gut

if gut.likely(email, "the customer threatens to cancel"):
    escalate()
```

That is the whole API for most uses. No configuration, no prompt, no threshold.

When the decision is worth being careful about, say so in words and a third answer becomes possible:

```python
import gut

decision = gut.likely(email, "the customer threatens to cancel",
                      stakes="high", lean="yes", ask_human=True)

match decision:
    case gut.YES:    escalate()
    case gut.NO:     auto_reply()
    case gut.UNSURE: send_to_review_queue()
```

Judgments are cheap enough to ask everywhere — ten questions about the same thing cost about as much
as one — and you can measure whether to trust them before you do.

> **Status: pre-1.0.** Everything here works and is covered by tests. Fitting calibrators, durable
> execution and taint tracking are deliberately out of scope; [DECISIONS.md](DECISIONS.md) records
> what was decided and why.

---

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

---

## Knowing when it doesn't know

Three words say how careful to be. They are not thresholds — you never name a number.

```python
likely(email, "the customer threatens to cancel", lean="yes")
likely(email, "the customer threatens to cancel", ask_human=True)
likely(email, "the customer threatens to cancel", stakes="high", lean="yes", ask_human=True)
```

- **`lean`** — `"yes"`, `"no"`, or nothing. Which mistake is worse, so which way to err.
- **`ask_human`** — whether `UNSURE` is possible at all. Off by default; nothing becomes
  three-valued behind your back.
- **`stakes`** — `"low"`, `"medium"`, `"high"`. How bad an automatic mistake is next to a person
  looking instead. Only widens the range where a person is asked, so it needs `ask_human=True`;
  `gut` warns if you pass it without.

The two knobs are independent on purpose: being more careful must never quietly change which way you
err. `lean` fixes where yes overtakes no; `stakes` only widens the band around it.

| `stakes` | `lean=None` | `lean="yes"` | `lean="no"` |
|---|---|---|---|
| **low** | ask 0.40 – 0.60 | ask 0.20 – 0.40 | ask 0.60 – 0.80 |
| **medium** | ask 0.25 – 0.75 | ask 0.125 – 0.625 | ask 0.375 – 0.875 |
| **high** | ask 0.10 – 0.90 | ask 0.05 – 0.85 | ask 0.15 – 0.95 |

Below the range it answers no, above it answers yes, inside it asks. Print the table yourself, so
you never have to take this page's word for it:

```python
import gut

for preset in gut.presets():
    print(preset)
```

`classify` and `rate` take `stakes` and `ask_human` too, where they become a confidence floor. They
do not take `lean`: there is no safer side of a four-way choice.

### `if` and `match`

`if decision:` works. `YES` is truthy, `NO` is falsy, and `UNSURE` is an explicit choice rather than
a silent one:

```python
import gut

gut.configure(on_unsure="raise")   # default: raises UnsureDecision
gut.configure(on_unsure="false")   # or "true", or a callback

with gut.on_unsure("false"):       # scoped; follows async tasks, never leaks across threads
    ...
```

The default raises because coercing `UNSURE` to `False` is the most dangerous option available: it
is what your existing `if` already does, so the third branch would vanish down the "no" path — the
exact bug this library exists to prevent, reintroduced as a default.

In a `match`, the outcomes must be **dotted**. A bare `case YES:` is not a value pattern in Python,
it is a capture pattern that matches anything, and the compiler rejects it:

```
SyntaxError: name capture 'YES' makes remaining patterns unreachable
```

```python
import gut
from gut import Outcome

match decision:
    case gut.YES: ...
    case gut.NO: ...

match decision:
    case Outcome.YES: ...
```

---

## Asking everything at once

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

---

## Knowing whether to trust it

None of the above is worth anything if the probabilities are not. Write down the cases you care
about:

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
gut eval predicates/          # or: pytest --gut-evals predicates/
```

```
PASS  cancel_threat  (noul)
        accuracy   96%  (97/101)   min 85%
        Brier      0.029   0 is perfect, 0.25 is a coin flip
        cal. error 0.100
        p 0.0-0.2   n=85   said 0.07   happened 0.00
        p 0.2-0.4   n=3    said 0.28   happened 0.67
        p 0.4-0.6   n=6    said 0.52   happened 1.00
        p 0.8-1.0   n=7    said 0.94   happened 1.00
```

Accuracy says it picked the right side. **Brier score** and **calibration error** say whether the
number attached to that answer was worth anything, and the table says where it wasn't. Below 30
examples the report says the numbers are noise rather than pretending otherwise.

`--json` for CI, `--plot out.png` for a reliability diagram (`pip install "gut[plot]"`), and
`max_ece:` in a file to fail on calibration as well as accuracy. Pass and fail still hang on
`min_accuracy` alone unless you opt in.

### What it found

Five predicates, 101 labelled tickets, `jev-1.13.0`. Two of five pass, and the failures are the
interesting part:

| predicate | kind | accuracy | Brier | ECE | |
|---|---|---|---|---|---|
| refund_request | noul | 97% | 0.028 | 0.049 | pass |
| cancel_threat | noul | 96% | 0.029 | 0.100 | pass |
| owning_team | choice | 84% | 0.110 | 0.037 | fail (accuracy) |
| bug_report | noul | 83% | 0.126 | 0.156 | fail |
| urgency | score | 74% | 0.255 | 0.248 | fail |

**`urgency`'s confidence is worse than useless on this task.** Brier `0.255` is worse than answering
`0.5` to everything, and its table is close to inverted — the bucket where the model was least sure
was its most accurate. A `min_confidence` floor, which is what `stakes` maps to for `rate`, would
route away the answers most likely to be right. **Do not use `stakes` on `rate` for a task like this
without measuring first.**

**`owning_team`'s confidence holds up** — the lowest calibration error of the five despite the
lowest accuracy. Same mechanism, sound on one task, unsound on another. That is the argument for
measuring rather than assuming, and it is why `gut eval` exists.

**The yes/no probabilities rank well but are not centred.** `cancel_threat` is under-confident in
the middle, `bug_report` over-confident at the top. Good enough to act on, not good enough to read
literally — which means a preset band drawn at `0.25–0.75` does not sit where you would assume on
either distribution.

### Fixing it

A measurement you can't act on is just bad news. `gut calibrate` fits a correction per predicate
from the same files, and reports what it bought **out of fold** — each point corrected by a
calibrator that never saw it — so the improvement is not self-graded:

```bash
gut calibrate predicates/ --cassette tape.json --model jev-1.13.0 --out calibration.json
```

```
  cancel_threat      brier 0.029 -> 0.006   ece 0.100 -> 0.004   (out-of-fold)
                     isotonic, 30 breakpoints
  urgency            brier 0.255 -> 0.197   ece 0.248 -> 0.061   (out-of-fold)
                     isotonic, 4 breakpoints
  bug_report         brier 0.126 -> 0.095   ece 0.156 -> 0.082   (out-of-fold)
                     in-sample ece would have read 0.007; that gap is the overfit
  owning_team        brier 0.110 -> 0.124   ece 0.037 -> 0.058   (out-of-fold)
                     DROPPED: this makes calibration worse out of fold (0.037 -> 0.058).

4 correction(s) written to calibration.json
dropped for making things worse: owning_team
```

```python
import gut
from gut import CalibrationSet

gut.configure(calibration=CalibrationSet.load("calibration.json"))
```

Four things that file will not let you get wrong.

**A correction that loses is not shipped.** `owning_team` was already the best-calibrated of the
five; fitting on a hundred examples added noise and nothing else. Shipping that is strictly worse
than shipping nothing, so it is dropped unless you pass `--keep-all`.

**The in-sample number is a lie and the tool says so.** `bug_report` reads `0.007` in sample against
`0.082` out of fold. That gap is isotonic fitting noise, and it is printed rather than averaged
away.

**A correction cannot fix a ranking problem.** `urgency`'s calibration error falls from `0.248` to
`0.061` and its accuracy does not move, because the fit flattens a signal that does not rank. That
is the right outcome — the number becomes honestly uninformative instead of confidently wrong — and
it is why isotonic is the default: given an inverted relationship, the best non-decreasing fit is a
constant, which says "I have no information" rather than dressing noise up as a probability.

**Corrections are per question, and per model.** A fit for "is this a churn threat" says nothing
about "is this a bug report". A question with no entry is left alone, and using a correction against
a model it was not fitted on warns once.

Decisions keep both numbers: `decision.p` is what the rule acted on, `decision.raw_p` is what the
model said. The cache stores the raw answer, so refitting invalidates nothing.

### Record once, replay forever

```bash
GUT_RECORD=1 gut eval predicates/ --cassette tape.json --model jev-1.13.0
gut eval predicates/ --cassette tape.json --model jev-1.13.0     # 0.63s, offline
```

Entries are keyed by the subject and the question, not by the request they travelled in, so
regrouping questions — adding `@semantic`, say — does not invalidate a recording. In replay an
unrecorded question is an error that names itself; quietly reaching for the network would turn one
forgotten re-record into a suite that passes on your laptop, fails in CI, and bills you either way.

> ⚠️ **A cassette holds whatever you asked about, verbatim.** Deliberate: a reviewer should be able
> to read a diff and see what changed about the model's behaviour. But recording against real
> customer tickets and committing the file commits customer text to your repository. Record against
> fixtures you are happy to publish, or keep the cassette out of version control.

`FakeBackend` covers development with no key at all, and this project's own suite passes without one.

---

## What it looks like on real data

[`examples/support_tickets/`](examples/support_tickets/) has 101 labelled tickets and the same
handler written with and without `gut`. Every row below asks the same five questions about the same
tickets; only what happens to the answers changes.

```
posture                     auto   human  false yes    missed    cost    team
--------------------------------------------------------------------------
no arguments                100%      0%          0         4      80     84%
stakes=low    lean=yes       97%      3%          0         0       3     79%
stakes=medium lean=yes       80%     20%          0         0      20     76%
stakes=high   lean=yes       40%     60%          0         0      61     67%
stakes=low    lean=no       100%      0%          0         8     160     79%

a hard 0.7 threshold, no third branch
threshold=0.7               100%      0%          -         8     160       -
```

**`stakes="low", lean="yes"` costs 3. The hard threshold costs 160.** Same model, same
probabilities, 97% of the queue still automatic, and the difference is three tickets out of a
hundred going to a person instead of eight churn risks going unanswered.

**More caution is not better.** High stakes also misses nothing and costs twenty times more, because
it sends 60% of the queue to a human. The dial has a minimum and it is not at either end.

**Leaning the wrong way is worse than having no opinion** — `lean="no"` reproduces the hard
threshold exactly.

---

## Exact costs, when a mistake has a price tag

The postures are shorthand for a cost model. If you know the actual numbers, give them instead:

```python
import gut

gut.likely(email, "the customer threatens to cancel",
           cost_false_yes=2,     # a CSM spends twenty minutes on a calm customer
           cost_false_no=50,     # we lose the account
           cost_human=1)         # someone reads the ticket and decides
```

Given a probability `p`:

```
expected cost of saying YES        = (1 - p) · cost_false_yes
expected cost of saying NO         =      p  · cost_false_no
expected cost of asking a human    =         cost_human
```

`gut` takes the cheapest. Ties prefer `UNSURE`, then `NO`. With no human in the loop this reduces to

```
YES  ⟺  p > cost_false_yes / (cost_false_yes + cost_false_no)
```

which for `2` and `50` is `0.038`. Nobody guesses `0.038`.

`threshold=` and `unsure_band=(lo, hi)` are there if you already know the number you want. Mixing
exact costs with posture words in one call is an error — one of them would have to win silently.

### The trap in that formula

The expected cost of asking a person is **flat** in `p`, while the cheaper of yes and no *peaks*
where those two lines cross. Put `cost_human` above that peak and there is no probability at all
where a person is worth asking: the third branch you carefully wrote is unreachable, silently. The
ceiling is

```
cost_false_yes · cost_false_no / (cost_false_yes + cost_false_no)
```

`1.92` for `2` and `50`. A `cost_human` of `5` never fires; `1` does. `gut` warns when you cross it,
and `Policy.max_useful_cost_human` tells you where it is. This is not theoretical — it is the
mistake the demo shipped with, and the report showed it as *zero tickets sent to a human* with no
error anywhere.

The postures cannot do this to you. They are defined as bands and the costs derived, and a band is
reachable whenever its ends are in the right order.

```python
import gut

print(gut.likely.__name__)                             # the rule is inspectable
print(gut.presets()[0].policy.describe())              # "no below 0.4, ask a person from 0.4 ..."
```

---

## Caching, logging, backends

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

`Backend` is a protocol — `ask()` and `model_id`. `JevBackend` is the real one, `FakeBackend` is for
development, `CassetteBackend` replays recordings. `import gut` never pulls in the vendor SDK.

---

## Honest limitations

**About the model.** Jev reads instructions literally and is weak at counting, arithmetic and date
comparison — compute those in Python and let `gut` judge the rest. Irrelevant state hurts accuracy,
so pass the narrowest subject that contains the answer. Multi-hop reasoning degrades.

**Text in the subject can influence the answer**, and `gut` does not do taint tracking. Treat a
decision over user-controlled text as advisory in security contexts.

**"Calibrated" is a claim, not a guarantee, and the measurements above show where it fails.** The
cost rule and the postures are only as good as the probabilities feeding them. Measure yours with
`gut eval` before trusting a preset band to sit where this page says it does, and fix what you can
with `gut calibrate`. A correction is fitted on your data, for your model — nothing ships
pre-calibrated, because nothing could be.

**`confidence` is not an accuracy estimate.** For `classify` and `rate` it is a statistic over how
peaked the answer's own distribution is. It held up for `classify` on our data and inverted for
`rate`. Measure it.

**Costs do not transfer across backends.** A cost model is a claim about *these* probabilities. A
different model, or an unpinned version that moves, can be sharper or flatter in the middle and put
the implied boundaries somewhere else on its distribution.

**Probabilities across questions are not comparable.** The vendor's docs warn that negated questions
need not sum to 1 and that different primitives yield non-comparable numbers. `gut` never
synthesises `P(no)` from a separately asked negated question, and neither should you.

**Decision ids survive edits, not moves.** An id is the question plus the module and function it is
asked in, so inserting lines above a call doesn't reset its history — moving it to another function
does.

**`@semantic` is speculative.** It asks questions behind branches that never run. It handles
`async` functions by running the prefetch in a worker thread -- correct and non-blocking, but a
backend that spoke `async` natively would not need the thread. `judge()` is still synchronous.

**Context limits.** 64k tokens for subject plus every question, 32k for subject plus the longest
one. Large batches split automatically, using a character-count estimate rather than a real
tokeniser — a wrong estimate costs an extra request, never a wrong answer.

---

## Development

```bash
uv sync --all-extras
uv run pytest
uv run mypy
uv run ruff check . && uv run ruff format --check .
uv run coverage run -m pytest && uv run coverage report
```

The suite runs offline and needs no API key. One live integration test is skipped unless
`TYPESAFE_API_KEY` is set. See [CONTRIBUTING.md](CONTRIBUTING.md) and [DECISIONS.md](DECISIONS.md).

## License

Apache-2.0
