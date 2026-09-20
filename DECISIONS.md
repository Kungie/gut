# Design decisions

A running log of choices made while building `gut`, especially the ambiguous ones. The rule we follow:
when a design question is genuinely ambiguous, pick the option that **never silently changes user code
behavior**, write it down here, and move on.

---

## D1 — Name: `gut`

**Date:** 2026-09-20

From *gut feeling* — the everyday phrase for exactly the kind of fast, confident-but-fallible
judgment this library makes programmable. Three letters, available on PyPI and TestPyPI, not a Python
keyword, no stdlib collision, and it reads well at the call site: `gut.likely(...)`, `gut.YES`.

The name appears in exactly one place for packaging metadata (`pyproject.toml`) and one package
directory (`src/gut/`), so a later rename stays cheap.

## D2 — License: Apache-2.0

**Date:** 2026-09-20

The handoff says "open-source Python package" without naming a license. Apache-2.0 is permissive, so
it imposes nothing on adopters, and it adds two things worth having for a library meant to sit in
other people's production decision paths: an explicit patent grant, and an explicit contribution
clause that sets the terms for inbound patches without a separate CLA.

The repository carries the full licence text in `LICENSE` and an attribution `NOTICE`, as section
4(d) provides for.

## D3 — Packaging: `uv_build`, `src/` layout, Python 3.10+

**Date:** 2026-09-20

- `src/` layout so tests import the installed package, not the working directory.
- Python 3.10 floor because `match` statements are core to the public API, and because `typesafe-sdk`
  itself requires `>=3.10`.
- `typesafe-sdk` is an **optional extra** (`gut[jev]`), not a hard dependency. The core decision rule,
  `Decision` types, cache, `@semantic` batching and the entire test suite must work with no API key
  and no vendor SDK installed.
- Core runtime dependencies stay at `httpx` and `pyyaml`.

---

## Jev / `typesafe-sdk` inspection — findings

**Date:** 2026-09-20 · SDK `typesafe-sdk==0.7.0` · model `jev-1.13.0` · docs `docs.typesafe.ai`

The handoff's description of Jev predates this repo, so every claim was checked against the installed
SDK source and the live docs before any backend design. **The handoff was accurate on all the facts it
stated.** What follows is the corrections and the things it did not mention.

### Corrections

| Handoff | Reality |
|---|---|
| import `typesafe` (implied) | The import name is **`typesafe_sdk`**. `pip install typesafe-sdk` was right. |
| `Noul(instructions)`, nothing else | `Noul` also takes **`criteria={"true": ..., "false": ...}`**, optional descriptions of each outcome. Both fields are optional. |
| `Choice(criteria={key: description})` | Same, but a description may be **`None`** — the label is then interpreted by its name alone. |
| `Score` takes 2–10 levels | Docs confirm "at least two levels and up to 10", **but the SDK's wire schema only enforces `min_length=1`**. A 1-level rubric passes the SDK and fails at the API. |
| `Score` returns `.score` / `.probabilities` / `.confidence` | Also returns **`.legend`**, level number → description. And the public `ScoreAnswer` **coerces `legend` and `probabilities` keys to `int`**, while `ChoiceAnswer.probabilities` stays keyed by `str`. Easy to get wrong. |
| "`JevBackend` … retries with backoff on 429, honors `retry-after`" | **The SDK already does all of this.** `RetryPolicy`: `max_retries=2`, backoff `0.5s → 5s` doubling, `backoff_jitter=0.25`, retried statuses `{408, 429, 5xx}`, `respect_retry_after=True` (honors both `retry-after` **and** `retry-after-ms`), and a 30s total retry budget per call. |

### Not mentioned in the handoff

- `model` is a **required** field on the wire request. The SDK fills it from the `model=` argument,
  then `TYPESAFE_DEFAULT_MODEL`, then the constant `"jev-latest"`.
- Aliases: `jev-latest` and `jev-preview`, both currently `jev-1.13.0`. `response.model` reports the
  resolved versioned ID — the handoff was right that this must be recorded on every decision.
- Env vars: `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`, `TYPESAFE_LOG_LEVEL`.
  `DEFAULT_TIMEOUT = 10.0` seconds per HTTP operation.
- `SystemOneResponse` offers typed views `.nouls` / `.choices` / `.scores` alongside `.answers`, plus
  `.model` and `.usage` (`input_tokens` / `output_tokens`, both `int | None` on the public type).
- Unknown answer types are **dropped with a warning** rather than raising — the SDK is forward-compatible.
- `questions` must be nonempty (`min_length=1`).
- There is an `AsyncTypeSafeClient`, and `client.models.list()` for discovering available versions.
- The SDK depends on **`httpx2`**, not `httpx`.
- Pricing: `$0.042` per million input tokens, output free. Confirms that padding a batch with extra
  questions is nearly free — the economic premise behind `@semantic`.
- Limits confirmed: 64k tokens per request, 32k for `state` plus the longest single question,
  250k tokens/s and 1,200 requests/min.

### The one that changes our design

**`confidence` is not a calibrated accuracy estimate.** The docs define it as "a statistic computed
from the probability distribution the answer already gives you" — i.e. how peaked the distribution is —
and advise "start with conservative thresholds, test with your own data, and adjust as you observe
results." Noul carries no confidence at all: "(Noul answers don't carry one.)"

Consequences for `gut`:

- `min_confidence` on `classify` / `rate` is a **spread filter**, not a probability of being right.
  The README and docstrings must say so rather than implying a guarantee.
- The cost rule for `likely` runs on `noul` directly, which *is* a probability of yes. Keep the two
  ideas separate in the API and never quietly convert one into the other.
- Docs warn that "negated questions don't necessarily sum to 1, and different primitives yield
  non-comparable results". So `gut` must **never** synthesize `P(no)` as `1 - noul` from a separately
  asked negated question, and must not compare a `Score` probability against a `Noul` probability.

## D4 — `JevBackend` configures the SDK's retry policy rather than reimplementing it

**Date:** 2026-09-20

Given the finding above, wrapping the SDK in our own retry loop would stack two backoffs and double
the effective delay on a 429. `JevBackend` instead exposes the knobs we care about (max retries,
timeout, pinned model) and maps them onto `RetryPolicy`. The `Backend` protocol stays free of retry
concerns so that a future backend without built-in retries can add its own.

## D5 — Client-side validation of question shapes

**Date:** 2026-09-20

`gut` validates before calling: 2–10 score levels, 2–255 choice options, nonempty questions. The SDK
lets a 1-level rubric through to fail server-side; catching it locally turns a round trip and an opaque
422 into an immediate, readable error. Validation lives next to the question builders, not in the
backend, so `FakeBackend` enforces the same rules as `JevBackend`.

## D6 — No mypy override for `typesafe_sdk`

**Date:** 2026-09-20

The SDK ships `py.typed` and is fully annotated; a probe module using `system_one`, `Noul`, `Choice`
and `Score` passes `mypy --strict` with no `ignore_missing_imports`. The override drafted in the
initial scaffold was removed so that real type errors against the SDK surface instead of being silenced.

## D7 — Outcomes are `Outcome` enum members, matched via a dotted name

**Date:** 2026-09-20

The handoff's example writes `case YES:` / `case NO:` / `case UNSURE:`. That is not valid Python: a
bare name in a `case` is a *capture* pattern, not a value pattern, so it matches everything and the
compiler rejects the rest. Verified on 3.10.21:

```
SyntaxError: name capture 'YES' makes remaining patterns unreachable
```

Only a dotted name is a value pattern. So the outcomes are members of an `Outcome` enum, exported at
package level as `gut.YES` / `gut.NO` / `gut.UNSURE`, and the documented spellings are `case gut.YES:`
or `case Outcome.YES:`. Both were verified to match, against the `Decision` object and against a bare
outcome, with `Decision.__eq__` comparing on the outcome.

`YES` / `NO` / `UNSURE` are still importable by bare name — they are useful in ordinary comparisons
(`if d is gut.YES`) — but the README documents the dotted form for `match`, and the test suite asserts
that the dotted form works rather than silently testing the capture-pattern spelling that always passes.

## D8 — Coverage runs as `coverage run -m pytest`, not through pytest-cov

**Date:** 2026-09-20

`gut` registers its own pytest plugin through the `pytest11` entry point, so installing the package
makes `--gut-evals` available everywhere. A consequence: pytest imports `gut.pytest_plugin` — and
therefore `gut/__init__.py` and everything it re-exports — while loading plugins, which happens
*before* pytest-cov starts measuring. The whole package then reads as unexecuted. Measured directly:

| | reported coverage |
|---|---|
| `pytest --cov` | 71% (`_errors.py` 0%, `__init__.py` 0%) |
| `coverage run -m pytest` | 100% |

Nothing about the tests differed; only when measurement began. Starting coverage as the process
entry point puts it ahead of plugin loading, so `coverage run -m pytest` is the project's coverage
command, `pytest-cov` is not a dependency, and CI runs the two steps separately. `fail_under = 95`
guards the number.

This is worth knowing beyond coverage: **the entry point means `import gut` happens in every pytest
run of every project that installs it.** Keeping `gut/__init__.py` cheap to import is a real
constraint, not a nicety.

## D9 — `on_unsure` defaults to `"raise"`

**Date:** 2026-09-20

`bool(decision)` has no honest answer for UNSURE, and the three candidate defaults are not equally
safe. Coercing to `False` is the most dangerous option available: it is what a developer's existing
`if` already does, so an UNSURE decision would flow silently down the "no" branch — precisely the
bug `gut` exists to prevent, reintroduced as a default. Coercing to `True` is the same failure
pointed at the more expensive branch.

So the default raises `UnsureDecision`, and the error message names the three ways out: handle the
branch with `match`, coerce process-wide with `configure(on_unsure=...)`, or coerce locally with the
`on_unsure()` context manager. Loud at development time, and never silent in production.

The scoped override lives in a `ContextVar` rather than a module global, so it follows `async` tasks
and does not leak across threads; both are covered by tests.

## D10 — `Decision` compares equal to its `Outcome`

**Date:** 2026-09-20

`match d: case gut.YES:` requires `d == gut.YES` to be true, because a value pattern is an equality
test. So `Decision.__eq__` returns `True` when compared against the matching `Outcome`, and
`__hash__` hashes the outcome so that the equal-implies-equal-hash contract holds.

The cost is that equality is **not transitive across decisions**: two different decisions can both
equal `gut.YES` without equalling each other. Decisions of the same kind still compare field by
field, so the surprise is confined to comparisons against outcomes, which is exactly where it is
wanted. The alternative — requiring `match d.outcome:` — keeps equality clean but makes the common
path noisier, and the handoff's API is explicit about matching the decision itself.

Comparing a decision against anything else returns `NotImplemented` rather than `False`, so Python's
reflected-comparison fallback still works. A test asserts `Outcome.YES == decision` in that
direction specifically.

## D11 — Decision-site ids use module and function, not file and line

**Date:** 2026-09-20

The handoff specifies the site id as a hash of the question spec plus the `file:line` of the call
site. Following that literally has a failure mode worth avoiding: **adding a line anywhere above the
call changes its id.** Since the id is what ties today's decisions to next month's resolved outcomes,
an unrelated edit would silently orphan a decision's entire calibration history, with nothing in the
data to indicate why the series stopped.

So the id is built from the question fingerprint plus `module:function`, which survives ordinary
editing. The file and line are still captured on every `CallSite` and recorded alongside the
decision, so debugging keeps the precise location; they are simply not part of the identity.

Two known limits, both preferred to the churn above:

- Two same-named functions in one module asking the *same* question share an id. That needs a
  genuine collision of place and question, and the answer is arguably the same decision anyway.
- `co_qualname` would distinguish methods on different classes, but it only exists on Python 3.11+,
  and an id that changes with the interpreter version is worse than one that merges rare duplicates.

The walk skips however many `gut` frames sit between the call and `caller_site()`, so the site does
not move when the library's internal call depth changes between releases.

## D12 — The cache key names the model asked for, not the model that answered

**Date:** 2026-09-20

The key is a hash of the state, the question spec and the model — as the handoff specifies — but
there is an ordering problem the specification glosses over: **the resolved model version is only
known after the call, and the key is needed before it.** So the key uses the model the backend was
*configured* to ask, which may be a moving alias such as `jev-latest`, while the entry stores the
resolved version that actually answered.

Two consequences, both deliberate:

- `Decision.model` on a cache hit reports the **stored** version, not the alias. Reporting the alias
  would make a recorded decision unfalsifiable later — you could no longer tell which model produced
  it.
- Caching under an alias can serve an answer from an older version after a release, because the key
  did not change when the alias moved. The honest fix is the one the vendor docs already recommend:
  pin a version. Callers who pin get exact keys; callers who use an alias get a documented trade,
  not a silent one.

`Backend` therefore exposes `model_id` as a read-only property, known before any call. Declaring it
read-only rather than as a plain attribute matters: a protocol attribute is implicitly settable, and
mypy rejects an implementation that exposes it as a property — which `FakeBackend` does.

An in-memory bounded LRU is the default, `SQLiteCache` persists across restarts, and `NullCache`
turns caching off.

## D13 — A backend is inferred only when the environment names an API key

**Date:** 2026-09-20

The quickstart reads `export TYPESAFE_API_KEY=...` and then calls `likely(...)` with no setup, so
`gut` has to produce a backend from nothing. Building one implicitly is a real side effect — it
starts billable calls — so the trigger has to be something nobody sets by accident.

`TYPESAFE_API_KEY` being present is exactly that: an explicit statement of intent, and the same
signal the vendor SDK already uses. With it set, `current_backend()` builds a `JevBackend` once and
keeps it. Without it, `gut` raises and names both ways forward — `FakeBackend` for offline work, or
configuring `JevBackend` explicitly. A whitespace-only value does not count.

`JevBackend` itself is resolved through a module `__getattr__` on both `gut` and `gut._backends`, so
`import gut` never pulls in `typesafe-sdk`. That matters more than it looks: the `pytest11` entry
point means `import gut` runs in every pytest session of every project that installs it (D8), and
none of those should drag in a vendor SDK nobody asked for.

## D14 — `@semantic` is conservative by construction, and speculative by design

**Date:** 2026-09-20

Two properties are worth stating plainly, because they are the ones a user will be surprised by.

**It cannot change what your code does.** Anything the analysis cannot prove statically is not
collected, and that call goes to the backend on its own exactly as it would undecorated. There is a
second layer under that: the prefetch scope is keyed by a fingerprint of the *real* question and the
*real* subject, computed at call time. So even a plan that guessed wrong cannot substitute an answer
— it simply fails to match, and the normal path runs. Being wrong costs a wasted request, never a
wrong answer.

The conservative exclusions worth naming:

- A parameter that is **assigned, deleted, or declared `global`/`nonlocal` anywhere** in the function
  is excluded entirely. After `ticket = ticket.strip()`, prefetching against the original value would
  answer about the wrong state. Tracking the rebinding through the control flow would be cleverer and
  occasionally wrong; excluding it is dull and always right.
- Calls inside **nested functions, lambdas and class bodies** are left alone: they may run with
  different bindings, or not at all.
- A call passing **`backend=`, `*args` or `**kwargs`** is left alone, since the question or the
  backend may not be the one the prefetch would use.

**It is speculative.** Questions behind branches that never execute are still asked. That is the
whole trade: billing is on input tokens, the state is paid for once per request, so five questions in
one call cost barely more than one, while five calls cost five states. If a question is expensive for
reasons other than tokens, keep it out of a decorated function.

Async functions are returned unchanged. The fetch is blocking, and quietly blocking an event loop is
worse than not batching.

## D15 — `judge()` hands back a handle typed as the decision it will become

**Date:** 2026-09-20

`j.likely(...)` returns a `Lazy` proxy but is annotated as returning `Decision`. The alternative —
annotating it honestly as `Lazy[Decision]` and making callers write `handle.decision.p` — would make
the common path worse for a distinction that does not matter at the call site: the handle is truthy,
matchable, comparable and readable exactly like the decision, because every one of those operations
resolves it first.

The cost is contained and worth naming. `isinstance(handle, Decision)` is `False`, and the two
members that belong to the handle rather than the decision — `.decision` and `.pending` — are
reachable at runtime but invisible to a type checker until you narrow with
`isinstance(handle, Lazy)`. Nothing in the documented API needs them.

`repr()` is the one operation that deliberately does **not** resolve. A handle printed in a debugger
or a log line must not issue a billable request as a side effect, so it reports whether it is pending
instead.

Leaving the `with` block resolves nothing either. Forcing resolution on exit would bill for questions
nobody read, which is the opposite of what an explicit API should do. The block only stops further
registration; existing handles still resolve afterwards.

## D16 — The decision log is off by default, and can never break a decision

**Date:** 2026-09-20

Two rules, both about a library writing to someone else's disk.

**Off by default.** The default sink writes nothing, and `recording()` short-circuits record
construction entirely so an unconfigured install pays nothing per decision. A library that starts
writing files nobody asked for is a library people configure around.

**A broken sink is swallowed.** Emission is wrapped, failures are logged at warning level, and the
decision proceeds. Logging is observability, not correctness — a full disk must not turn a working
classifier into an outage.

The record carries the canonical question and the resolved model version, not just the id, so a log
line stays interpretable without the code that produced it and without guessing which model answered.
`site` records file and line for a human chasing it down, while the id is built from module and
function (D11) — the log keeps both, and only one of them is identity.

`resolve(actual=...)` exists now although nothing reads it. Calibration is explicitly out of scope
for this release, but it is impossible to build retroactively: a decision that was never recorded
cannot be checked against an outcome. Shipping the data path first is the difference between
answering "is this calibrated on my data?" from existing logs and answering it six months late.

## D17 — Cassettes key per question, and a miss is an error

**Date:** 2026-09-20

Two choices about record/replay, both about what a fixture is for.

**Entries are keyed by state and question, not by the request they travelled in.** The obvious
implementation keys a whole batch, which means adding `@semantic` — or changing which questions get
grouped — silently invalidates every recording. Keying per question makes a cassette survive changes
to how the calls are batched, which is exactly the refactor most likely to happen after the
recording exists.

**Replay never falls back to the network.** An unrecorded question raises and names itself. Quietly
making the call instead would mean a suite that passes on the author's laptop, fails in CI where
there is no key, and bills the account in between — a failure mode that is hard to notice and easy
to blame on something else.

Cassettes store the state, the question and the answer in full rather than hashes. They are test
fixtures meant to be committed and reviewed: a diff should show *what the model's behaviour changed
to*, which a file of hashes cannot.

## D18 — `min_accuracy` defaults to 1.0

**Date:** 2026-09-20

Judgment tasks rarely justify demanding every example, and the handoff's own illustration uses `0.9`.
The default is `1.0` anyway, because the two failure modes are not symmetric: a bar that is too high
fails loudly on the first run and gets lowered deliberately, while a bar that is too low silently
accepts a predicate that was already wrong about a case you wrote down. Lowering it should be
something you decided.

## D19 — `gut` warns when the human branch is unreachable

**Date:** 2026-09-20

Found while building the demo, and worth recording because the example it invalidates is the
handoff's own: `cost_false_yes=2, cost_false_no=50, cost_human=5` **can never return UNSURE.**

The expected cost of asking a person is flat in `p`, while the cheaper of yes and no peaks where
those two lines cross, at `cost_false_yes · cost_false_no / (cost_false_yes + cost_false_no)` —
`1.92` for those numbers. A `cost_human` of `5` sits above the peak, so at every probability either
yes or no is cheaper, and the third branch the developer carefully wrote is dead code. Nothing
errors. The `UNSURE` case simply never runs, and a queue that was supposed to route hard cases to a
person routes none.

`policy()` now warns, naming the ceiling. `Policy.unsure_reachable` and
`Policy.max_useful_cost_human` expose the same fact for anyone who wants to check it themselves.

The boundary is reachable, not unreachable: exactly at the peak all three options tie, and the tie
rule prefers `UNSURE`. A test pins that, because the off-by-one here is the difference between a
warning that is right and one that cries wolf on a working configuration.

The demo's own costs were wrong in exactly this way on the first run: 101 tickets, zero sent to a
human. That is the report doing its job, and the reason the demo is in the repository rather than in
a README.

## D20 — Posture presets are defined as bands, with the costs derived

**Date:** 2026-09-20

The middle layer — `stakes`, `lean`, `ask_human` — maps onto the cost rule rather than sitting
beside it, so there is still exactly one thing that decides anything. The question was which end to
define.

Defining **bands** and deriving the costs, rather than tabulating costs and hoping the bands come
out sensible, buys two properties that matter:

**The trap in D19 becomes impossible here.** A cost policy asks a person exactly for `p` in
`[cost_human/cost_false_no, 1 - cost_human/cost_false_yes]`, which is a real interval precisely when
`lo < hi`. A preset *is* a band with `lo < hi`, so every preset is reachable by construction. It is
not guarded against; it cannot occur.

**The two knobs stay independent.** `lean` fixes the point where yes overtakes no (`0.25` / `0.5` /
`0.75`) and `stakes` only widens the band around it. Being more careful must never quietly change
which way you err, so with `a = lo` and `b = 1 - hi`, the threshold is `a/(a+b)` — fixed by `lean` —
and `a + b` is the width knob, fixed by `stakes`.

| stakes | `lean=None` | `lean="yes"` | `lean="no"` |
|---|---|---|---|
| low | 0.400 – 0.600 | 0.200 – 0.400 | 0.600 – 0.800 |
| medium | 0.250 – 0.750 | 0.125 – 0.625 | 0.375 – 0.875 |
| high | 0.100 – 0.900 | 0.050 – 0.850 | 0.150 – 0.950 |

`lean=None` lands exactly on the targets the brief named. The derived costs are `cost_human = 1`,
`cost_false_yes = 1/(1 - hi)`, `cost_false_no = 1/lo`.

Without `ask_human`, the costs come from the threshold alone (`cost_false_yes = t`,
`cost_false_no = 1 - t`), so `stakes` is genuinely inert rather than merely inconsequential — the
resulting `Policy` compares equal whatever `stakes` said. A test asserts that, because "has no
effect" in a warning should be literally true.

`gut.presets()` and `Policy.describe()` print the bands. Costs are what you configure; boundaries
are what you can check against a model's behaviour, and the gap between the two is where a
misunderstanding lives.

## D21 — `lean` does not apply to `classify` and `rate`

**Date:** 2026-09-20

`classify` and `rate` answer *which* and *how much*; neither has a probability of yes for the cost
rule to work on, and neither has a direction to err in — there is no "safer side" of a four-way
choice. So the posture maps onto `min_confidence` instead: `stakes` becomes a confidence floor
(`low` 0.50, `medium` 0.65, `high` 0.80) and `ask_human` decides whether there is a floor at all.
`lean` is not accepted, rather than accepted and ignored.

That floor is a **spread filter, not a probability of being right** — the same caveat that applies
to `min_confidence` everywhere else. `stakes="high"` on a `classify` means "only act on a peaked
answer", not "only act when 80% likely to be correct".

Mixing a posture with `min_confidence` is an error, on the same reasoning as mixing it with costs:
one of them would have to win silently.

## D22 — Thirty examples before a calibration number means anything

**Date:** 2026-09-20

`gut eval` reports Brier score and expected calibration error on whatever you give it, and both are
meaningless on ten examples: one flipped label moves ECE by a tenth. Below `MIN_EXAMPLES = 30` the
report says so in the output and `Calibration.reliable` is `False`, but the numbers are still
printed — hiding them would just move the guessing somewhere else.

Thirty is a convention, not a derivation. With the default five buckets it is roughly the point
where a bucket can hold enough examples for its observed frequency to be more than one or two
tickets. Larger would be defensible; the important part is that the threshold is stated and visible
in the output rather than left to the reader.

Pass and fail still hang on `min_accuracy` alone. A file written before anyone measured calibration
must not start failing because the measurement now exists, so `max_ece` is an opt-in field.

## D23 — What the first real calibration run found

**Date:** 2026-09-20

Five predicates built from the 101-ticket dataset, recorded against `jev-1.13.0`, replayed offline
in 0.63s. This is the measurement the whole cost rule rests on, so it is recorded here rather than
summarised away.

| predicate | kind | accuracy | Brier | ECE | |
|---|---|---|---|---|---|
| refund_request | noul | 97% | 0.028 | 0.049 | pass |
| cancel_threat | noul | 96% | 0.029 | 0.100 | pass |
| owning_team | choice | 84% | 0.110 | 0.037 | fail (accuracy) |
| bug_report | noul | 83% | 0.126 | 0.156 | fail |
| urgency | score | 74% | 0.255 | 0.248 | fail |

Three findings, in order of how much they matter.

**`rate` confidence is not usable as a gate on this task.** A Brier score of `0.255` is worse than
answering `0.5` to everything, and the reliability table is close to inverted:

```
p 0.2-0.4   n=18   said 0.31   happened 0.89
p 0.4-0.6   n=24   said 0.51   happened 0.79
p 0.6-0.8   n=28   said 0.70   happened 0.46
p 0.8-1.0   n=31   said 0.90   happened 0.87
```

The bucket where the model was *least* sure was its most accurate, and the `0.6-0.8` bucket was its
worst. A `min_confidence` floor — which is what `stakes` maps to for `rate` (D21) — would therefore
route away the answers most likely to be right. **On this task, `stakes` on `rate` is worse than
useless.** The mechanism is not broken; the assumption that a score's confidence tracks correctness
does not hold here. It is documented as something to measure, not something to trust, and this is
the measurement.

**`classify` confidence holds up.** `owning_team` has the lowest ECE of the five (`0.037`) despite
only 84% accuracy: the model knows when it is guessing. So the same `stakes` → `min_confidence`
mapping is sound for `classify` on this data and unsound for `rate`, which is exactly why the
answer has to be measured per task rather than assumed per primitive.

**Noul probabilities are good but not centred.** `cancel_threat` is under-confident in the middle
(said `0.28`, happened `0.67`; said `0.52`, happened `1.00`) while `bug_report` is over-confident at
the top (said `0.92`, happened `0.75`). Both rank well; the numbers are stretched. For the cost
rule this means a preset band drawn at `0.25–0.75` does not sit where you would expect on either
distribution, and the honest fix is to measure and adjust rather than to trust the defaults.

Fitting calibrators to correct any of this is deliberately out of scope. Measuring it is the
prerequisite, and it now exists.

## D24 — Isotonic by default, because of how it fails

**Date:** 2026-09-20

Two calibrators ship. **Platt** is a two-parameter logistic fit in log-odds space: it can stretch or
shift a curve but never bend it, and it works on very little data. **Isotonic** is the best
non-decreasing fit by pool-adjacent-violators, assuming nothing about the shape.

Isotonic is the default because of its failure mode rather than its fit. Given a relationship that
is actually *inverted* — which is exactly what D23 found in `urgency` — the best non-decreasing fit
is a **constant**. A signal that does not rank therefore calibrates to "I have no information"
instead of to a confident lie. Fitted on the real `urgency` data it collapses three of its four
buckets onto a single value.

That property is worth more than a better curve, because the failure it guards against is the one
that does real damage: a number that looks like a probability, behaves like noise, and gets fed into
a cost rule.

Neither invents information. A calibrator fixes a number that ranks well and is wrongly scaled; it
cannot fix a number that does not rank, and a test pins that accuracy is untouched by construction.

## D25 — Corrections are keyed per question, and per model

**Date:** 2026-09-20

A correction fitted on "is this a churn threat" says nothing about "is this a bug report". The two
questions have different base rates, different difficulty, and different failure shapes; applying
one to the other would be worse than applying nothing. So a `CalibrationSet` maps question
fingerprint → calibrator, and a question with no entry gets the identity.

The model is recorded on each entry and checked at use. A mismatch warns once per question rather
than refusing: refusing would break a deployment over a version bump, and a correction fitted on a
near neighbour is usually better than none — but it is a claim about one model's distribution, so
silence would be wrong too.

Corrections are applied **on the way out of the cache**, never on the way in. What is stored is what
the model said, so refitting a calibrator does not invalidate a single cached answer. The same
applies to cassettes, and to `Decision.raw_p`, which keeps the uncorrected number alongside the one
the rule used.

## D26 — What fitting produced, and what got thrown away

**Date:** 2026-09-20

Fitted on the same five predicates as D23, reported out-of-fold over five folds so the improvement
is not self-graded.

| predicate | Brier | ECE | |
|---|---|---|---|
| cancel_threat | 0.029 → 0.006 | 0.100 → **0.004** | shipped |
| urgency | 0.255 → 0.197 | 0.248 → **0.061** | shipped |
| bug_report | 0.126 → 0.095 | 0.156 → **0.082** | shipped |
| refund_request | 0.028 → 0.037 | 0.049 → **0.036** | shipped |
| owning_team | 0.110 → 0.124 | 0.037 → 0.058 | **dropped** |

**`owning_team` got worse, so it is not shipped.** It was already the best-calibrated of the five;
fitting on 101 examples added noise and nothing else. `gut calibrate` drops any correction that
loses out of fold and says so, with `--keep-all` to override. Shipping a correction that makes
calibration worse is strictly worse than shipping nothing, and the in-sample number would have
hidden it.

**The in-sample numbers are not to be believed, and the tool prints the gap.** For `bug_report`,
in-sample ECE reads `0.007` against `0.082` out of fold — a tenfold difference, which is isotonic
fitting noise on 101 points. Evaluating with the corrections applied on the same data shows
`cancel_threat` at 100% accuracy and an ECE of exactly `0.000`, which is the signature of measuring
a fit on its own training set, not a result.

**Calibration cannot fix `urgency`'s real problem.** Its ECE falls from `0.248` to `0.061` and its
accuracy does not move at all, because the correction flattens a signal that does not rank. That is
the right outcome: the number is now honestly uninformative rather than confidently wrong, and a
`min_confidence` gate built on it will abstain rather than choosing badly.

**Calibration optimises calibration, not accuracy.** `refund_request` improves on ECE and loses a
point of accuracy. They are different objectives and a fit will trade one for the other.

## D27 — `@semantic` batches coroutines by moving the prefetch off the loop

**Date:** 2026-09-20

`@semantic` used to return coroutine functions unchanged, on the reasoning that a blocking fetch
inside an event loop is worse than no batching. That was the wrong comparison. An `async` handler
calling `likely()` was *already* blocking — once per judgment — so declining to decorate it made
things strictly worse, not safer.

The prefetch is the only call that touches the network. Everything after it is answered from the
scope, in memory. So a decorated coroutine now runs that single call through `asyncio.to_thread`
and awaits it, and the body never blocks at all. Measured on a backend with a 200 ms delay: three
judgments in one request, the loop taking nineteen turns while it ran, and three concurrent tickets
finishing in 209 ms where sequential would be 600.

The scope is entered in the coroutine's own context after the await, so it follows that task and no
other. A sibling task awaiting concurrently sees none of it, which a test pins by interleaving two
handlers with different subjects across an `await`.

The event-loop claim is tested without timing. The backend parks inside `ask` until a coroutine on
the loop releases it; if the loop were blocked, that coroutine could never run and the wait times
out. A stopwatch would have been flaky in CI and would have proved less.

A backend that speaks `async` natively would avoid the thread entirely and is the better end state.
It is a much larger change -- an `AsyncBackend` protocol, an async path through cache, batching and
logging -- and is noted below rather than started. `judge()` is still synchronous for the same
reason.

## D28 — An agent skill, tested like code

**Date:** 2026-09-20

Much of the code that will use `gut` is going to be written by coding agents and reviewed by people.
An agent reading a 500-line README to write three lines is the wrong shape, so `skills/gut/SKILL.md`
is the short version: the correct spellings, the traps, and seven rules of thumb. `llms.txt` at the
root is the index, following the convention this project's own research depended on when reading
the vendor's docs.

Both are **tested against the code**, harder than prose usually is, because their reader will not
notice a mistake and argue about it:

- every Python example compiles, and the offline setup snippet is executed;
- every API name the skill mentions must still be in `gut.__all__`;
- the values it says are accepted — `stakes="low"|"medium"|"high"` — are checked against
  `STAKES_CERTAINTY` and `LEAN_THRESHOLD`, so adding a level without documenting it fails;
- its YAML predicate example is loaded through the real parser;
- the `SyntaxError` it quotes is produced by compiling the bad spelling;
- the ceiling formula it states is compared against `max_useful_cost_human`;
- every file `llms.txt` links to must exist, and every CLI subcommand must be mentioned.

`_cli.COMMANDS` exists so that last check does not reach into argparse internals. Documentation that
can drift silently is worse than none, and this is the documentation most likely to be acted on
without a second look.

## D29 — `stakes` on `rate` warns, pointing at what we measured

**Date:** 2026-09-20

D23 measured `rate`'s confidence running close to backwards on a real task: the bucket where the
model was least sure was its most accurate. `stakes` and `ask_human` map onto a `min_confidence`
floor for `rate` (D21), so on a task shaped like that one, the floor would route away exactly the
ratings worth keeping.

A warning rather than an error, and rather than removing the feature. The mechanism is sound —
`classify` confidence, measured the same way, was the best calibrated of the five predicates — and
the caller's task may not be the one we measured. What is not defensible is saying nothing while
handing someone a gate built on a number we have watched behave badly. The warning names the
finding and points at `gut eval`.

An explicit `min_confidence=` on `rate` is left alone: a number the caller chose is a decision, not
a default worth second-guessing.

## D30 — The README is a pitch; the manual lives in `docs/`

**Date:** 2026-09-20

The README had grown to 574 accurate lines, and after the first screen it was `match` syntax, cache
keys and cassette warnings. All of that earns its place somewhere — just not in the file that has
thirty seconds to explain what problem this solves.

So the README is now the pitch: the problem, why the existing three answers hurt, one line of code,
what it is good for, the third branch, one measured result, and links. Seven pages under `docs/`
carry everything else, unchanged in substance.

The guarantee that documentation runs got **stronger** rather than weaker in the move. Every Python
block in the README and in every `docs/` page is now *executed*, not merely compiled, against a
`FakeBackend` in a temporary directory, with a prelude supplying the names an illustrative snippet
expects its reader to have (`email`, `ticket`, `Team`). A page runs top to bottom in one namespace,
the way it is read, and the backend is re-seeded between blocks so a snippet that reconfigures `gut`
cannot break the next one. The sandbox sets a dummy API key so `JevBackend(...)` constructs, and an
unroutable base URL so that if a snippet ever tried to *call* it, the test would fail instead of
reaching the real API.

Two checks exist to stop the README growing back: it must stay under 120 lines, and naming
`cost_false_yes`, `on_unsure`, `GUT_RECORD` or `SQLiteCache` in it is a failure that names the page
each belongs to.

## D31 — The three benchmark datasets, and what may be committed

**Date:** 2026-09-20

Every number in this repository was measured against 101 tickets written for it. These three are
not. Each was verified — link, licence, size, label set — before use, and pinned.

| dataset | source | licence | may we redistribute? |
|---|---|---|---|
| **CLINC150** | `clinc/oos-eval` at `828f8093`, `data/data_full.json`, SHA-256 `36923c37…` | CC BY 3.0 per the HuggingFace dataset card; **the upstream repository declares none** | yes, with attribution |
| **NLBSE'24 issues** | `nlbse2024/issue-report-classification` at `2927bc67`, `data/issues_{train,test}.csv`, SHA-256 `18dc42a3…` / `4f7d8619…` | **none declared**, and the text is third-party GitHub content | **no** |
| **SMS Spam Collection** | UCI dataset 228, SHA-256 `1587ea43…` | CC BY 4.0 (UCI is authoritative; the HuggingFace mirror says "unknown") | yes, with attribution |

Nothing is downloaded without a SHA-256 check. A benchmark whose inputs can change underneath it is
not a benchmark, and "the upstream file moved" should be a loud failure rather than a quiet shift in
the results.

**Cassettes follow the licence.** CLINC and SMS recordings are committed, so those numbers reproduce
offline from a clean clone. The NLBSE cassette contains issue text under no licence and is therefore
**not** committed; `docs/benchmarks.md` says how to regenerate it, and that is the honest cost of
using it.

The NLBSE'24 competition data was used rather than NLBSE'23: 2023 ships 1.4 M issues as external
tarballs, while 2024 has 3,000 balanced issues in-repo. Three thousand real issues is plenty for a
500-example test sample, and the smaller download is the difference between a benchmark someone runs
and one they read about.

CLINC's 150 intent descriptions are the label names with underscores removed — mechanical, not
hand-written. Writing 150 descriptions by hand would be tuning the prompt against the labels, which
is the exact thing the dev/test split exists to prevent.

Banking77, TweetEval and GoEmotions were skipped. The first three cover a binary judgment, a
multiway classification and out-of-scope detection, which is what the claims need; GoEmotions would
have re-tested D23's finding about `rate` confidence and is the most interesting of the three to
add next.

## D32 — The method, and why each part of it is there

**Date:** 2026-09-20

Numbers from a model you are also tuning against are worthless. The protocol is ordinary and the
discipline is the point.

- **One split, fixed seed `20260920`,** stratified by label. Dev ≈ 200, test ≈ 500. Dev is drawn
  first and removed, so they cannot overlap. The sampling code is `stratified_split`, and a test
  asserts the split is deterministic, that the two halves are disjoint, and that proportions hold.
- **CLINC's out-of-scope share is set deliberately** to 100 of 550 test examples (18.2%), matching
  the canonical CLINC test split, rather than the 5% its natural share in the corpus would give.
  The departure is a `quota` argument, and it is reported rather than hidden.
- **Question wording is written against dev and then frozen.** `python -m benchmarks --dev-only`
  exists so that judging the wording cannot accidentally show a test result. The wording lives in
  `benchmarks/_datasets.py` as constants for the same reason.
- **Calibrators are fitted on dev, applied to test.** Never fitted on what they are scored on.
- **The model is pinned** to `jev-1.13.0` and recorded in every result.
- **The budget is checked before spending.** The estimate is printed and a run that would exceed
  five dollars refuses to start. The real figure is around six cents.

Each example is asked **exactly once**. A posture changes how an answer is acted on, never what was
asked — a property of the design that the test suite already asserts — so the whole posture sweep is
computed afterwards from those probabilities. Ten postures therefore cost what one costs, and more
importantly every posture is scored on *identical* model answers, which is the only thing that makes
comparing them meaningful.

## D33 — The question wording, and the one time it was revised

**Date:** 2026-09-20

Wording was written against the dev samples and frozen before any test set was scored. It was
revised **once**, on dev evidence, and this is the record of it.

The first attempt at the two NLBSE questions asked what the issue *was about*: "this issue reports
that something is broken or behaving incorrectly". On dev that agreed with the label 63.2% of the
time, and the failures said why. GitHub issue templates put the category in the body — a feature
request whose form reads `Type: <b>Bug</b>`, a question that pastes a stack trace. What an issue
*contains* and what it is *for* are different things, and the question was asking about the wrong
one.

Rewritten around the author's purpose — "the author opened this issue to report a defect: they are
saying the software does something wrong and should be fixed, rather than asking for a new feature
or for help" — and the input truncated from 4,000 to 1,500 characters, since Jev's documented
weakness is that irrelevant state acts as a distractor and a GitHub issue is mostly boilerplate.

| dev | before | after |
|---|---|---|
| nlbse-bug agreement | 63.2% | **81.6%** |
| nlbse-bug Brier | 0.294 | **0.142** |
| nlbse-kind agreement | 70.6% | **72.6%** |
| nlbse-kind Brier | 0.217 | **0.187** |

CLINC (72.6% overall, 92% in-scope) and SMS (99.5%) were left alone. No wording was touched after a
test set was scored.

## D34 — CLINC's out-of-scope set overlaps its in-scope intents

**Date:** 2026-09-20

Worth recording because it caps what any system can score, and reporting an out-of-scope number
without it would be misleading.

Of the four out-of-scope dev queries answered confidently, two are near-duplicates of in-scope
intents. `"give me the weather forecast for today"` is labelled out-of-scope, while `"give me the
7 day forecast"` and `"what is the weather going to be like today"` are labelled `weather`.
Likewise `"how many calories does jumping up and down burn"` against the `calories` intent.

Nothing is wrong with the model's answer there, and nothing is wrong with `gut`. The dataset's
out-of-scope set was collected separately from its in-scope one and they touch. So the measured
out-of-scope recall is a floor, not a ceiling, and the two-or-so percent it costs should be read as
label noise rather than as a failure to abstain.

## D35 — A correction that collapses the range disables `ask_human`, and now says so

**Date:** 2026-09-20

Found on the SMS benchmark, and it is the most useful thing the benchmarks turned up about `gut`
itself.

SMS spam is nearly separable, so the isotonic fit on 200 dev examples learned a step. Applied to
the 500 test messages it mapped **every one of them to exactly 0.0 or 1.0** — 433 and 67. Average
calibration improved (ECE 0.038 → 0.016), and the third branch vanished: no posture band can
contain a probability that is exactly 0 or exactly 1, so `ask_human=True` became a no-op at every
`stakes` and the automatic error rate got slightly *worse* (1.4% → 1.6%).

That is the objectives diverging. A calibrator minimises average error over the whole distribution,
which rewards confidence wherever the model is usually right. `gut`'s value is concentrated in the
cases where it is not, and those are exactly the ones a collapsing fit throws away.

`gut calibrate` now checks how many examples land inside the `stakes="medium"` band after
correction and warns when the answer is none. A warning rather than a refusal: the correction is
genuinely better calibrated, and a caller who is not using `ask_human` loses nothing by it. What is
not defensible is disabling someone's third branch silently.

The general lesson, and it belongs in the docs rather than only here: **calibrate for the decision
you are making, not for the average.** If abstention matters, check that the corrected
probabilities still reach the middle.

## D36 — No `tone()`, and the benchmark that settles it

**Date:** 2026-09-20

Asked for directly: *should there be a `tone(msg, "sarcastic")`?*

The API answer is no, and it is the same answer as for `sentiment()`, `spam()` and `intent()`:
`tone(msg, "sarcastic")` is `likely(msg, "…the author is being sarcastic…")` with a smaller
vocabulary. Tone is a subject, not a shape of question. `gut` has three shapes — is it true, which
one, how much — and a fourth entry point that collapses to the first buys nothing and costs the
library a boundary it currently keeps.

But the question underneath it was real, and not answerable from the armchair: **can the model
read tone at all?** Jev's documented weakness is literal reading, and sarcasm is exactly what a
literal reader misses. So TweetEval's irony set was added as a fifth benchmark
([docs/benchmarks.md](docs/benchmarks.md)), and it turned out to be the clearest demonstration of
the mechanism on the page:

- 28.4% error with no arguments, against a 48% base rate. The model is genuinely bad at this.
- ECE **0.036** — the second-best calibration of the five, better than the CLINC router that is
  three times more accurate.
- `stakes="high"` reaches **6.8% error at 15% coverage**: a 4× reduction, on the task where the
  model comes closest to guessing. (Not the highest raw error on the page — `nlbse-kind` is 30.7%
  — but that is a three-way choice where chance errs 67%; here chance errs 50%.)

Being frequently wrong and being overconfident are independent properties, and `gut` only requires
the second to be false. `nlbse-kind` is the control: similar accuracy, ECE 0.174, and the same
posture only reaches 18.8%.

Three wordings were tried on dev before freezing, and they landed within one point of each other
(66.0–67.0%). Naming the failure mode in the prompt — "dry understatement, fake enthusiasm" —
bought 0.5 points. The ceiling belongs to the model, not to the prompt, which is worth recording
because the opposite is usually assumed.

The cassette is not committed: the tweets are third-party content under no declared licence, the
same call as NLBSE in [D31](#d31).

What the docs get instead of a function: the irony section, and the point that this question is
only usable at `stakes="high"`. A `tone()` returning a bare label would have hidden precisely that.

## D37 — Running one benchmark deleted the other four

**Date:** 2026-09-20

`save()` wrote the results of the current run and nothing else, so `python -m benchmarks irony`
replaced a five-entry `results.json` with a one-entry one. It already had a guard for the related
problem — an offline replay measures ~0 ms, so recorded latencies were protected from being
overwritten with zeros — and that guard was defeated by this one: the entries were not overwritten,
they were dropped, and the next full run found nothing to restore from. Four live latency
measurements were lost and had to be re-measured.

`save()` now merges into the file and keeps entries it did not rerun, and `--latency` says so out
loud when a benchmark has no scored run to merge into rather than skipping it silently. Three
tests in `tests/test_benchmarks.py` cover it, named after the failures rather than the functions.

Worth the entry because of the shape: a guard that protects a value against being *changed* does
nothing about it being *deleted*, and the deletion was invisible — the file was still valid JSON
and the run still printed a table.

## Next steps, noted and not started

- **A second backend.** Every benchmark number comes from one model. The strongest evidence that
  `gut`'s value is in `gut` rather than in Jev would be repeating the risk-coverage and
  cross-dataset measurements against an LLM exposing token log-probabilities, or a small local
  model reading option probabilities. This is the most valuable thing left undone.
- A native `async` backend, so batched judgments need no worker thread, and an `async` `judge()`.
- Fitting calibrators from resolved production decisions (`resolve()`) rather than only from eval
  files, which is where the data actually accumulates.
- More datasets: GoEmotions would re-test D23's finding about `rate` confidence on real data,
  which is the one primitive the current five do not exercise.
- A written specification separate from the docs.
- Agent skills, so a coding agent can use `gut` without reading the whole README.
- `async` support in `@semantic`, which today declines rather than blocking an event loop.

---

## Implementation order

Built in this order, keeping the suite green at every step. All of it is done.

| | | |
|---|---|---|
| 1 | decision rule, property-tested | `_rule.py` |
| 2 | `Decision` types, truthiness, `match` semantics | `_decision.py`, `_outcomes.py`, `_config.py` |
| 3 | question specs and `FakeBackend` | `_questions.py`, `_backends/` |
| 4 | `likely` / `classify` / `rate` | `_api.py`, `_site.py` |
| 5 | cache | `_cache.py`, `_serde.py` |
| 6 | `JevBackend` | `_backends/jev.py` |
| 7 | `@semantic` and `judge()` | `_semantic.py`, `_judge.py`, `_scope.py`, `_batching.py` |
| 8 | decision log | `_log.py` |
| 9 | example files, pytest plugin, cassettes | `_evals.py`, `pytest_plugin.py`, `_cassette.py` |
| 10 | demo | `examples/support_tickets/` |
| 11 | README | `README.md`, `tests/test_readme.py` |

## Acceptance criteria

Each one, and where it is checked.

| criterion | evidence |
|---|---|
| `pip install -e .` works; the quickstart runs with `FakeBackend` and with a real key | verified in a clean 3.12 venv, core-only and with `[jev]`; `tests/test_readme.py` executes the quickstart block from the README itself |
| `match` on YES / NO / UNSURE works; `if likely(...)` follows the configured unsure policy | `tests/test_decision.py`, plus a test that compiles the bare-name spelling and asserts the `SyntaxError` |
| the cost rule matches the formulas, covered by property-based tests | `tests/test_rule.py`: expected-cost optimality, the threshold reduction, and monotonicity in `p`, all under hypothesis |
| a decorated handler with 5 questions on one ticket makes exactly 1 backend call | `tests/test_semantic.py`, and the demo against the live model: 101 requests carrying 505 judgments |
| `pytest --gut-evals` runs the YAML files and fails below `min_accuracy` | `tests/test_pytest_plugin.py`, run through pytest's own `pytester` |
| replay mode runs the whole suite offline and deterministically | `tests/test_cassette.py`; measured on the demo predicates at 4.05s recording, 0.07s replaying |
| `report.py` prints the demo metrics | `examples/support_tickets/report.py`, output in that directory's README |

Deliberately out of scope, and not designed around: calibration from resolved outcomes, durable
execution, taint tracking, a linter for unhandled UNSURE, anything hosted, and a TypeScript port.
The decision log exists so the first of those can be built on real data rather than retrofitted.
