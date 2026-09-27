# Design decisions

A running log of choices made while building `gut`, especially the ambiguous ones. The rule we follow:
when a design question is genuinely ambiguous, pick the option that **never silently changes user code
behavior**, write it down here, and move on.

`gut` changed direction on 2026-09-27 (D38): it is a coding primitive for small models, not a study
of one model. Decisions that belonged to the old direction -- benchmarks, calibration, evaluation
files, record/replay -- are listed under [Retired](#retired) with one line each; their full text is in
the git history. Numbers are never reused, so a `D`-reference in the code always means one thing.

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
- Core runtime dependencies stay at `httpx`, used by the OpenAI-compatible backend and imported
  only when that backend is. `pyyaml` left with the evaluation files (D38). PyTorch and
  `transformers` are the optional `local` extra (D43).

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

**Date:** 2026-09-20 · **Reason superseded 2026-09-27**

`gut` used to register a pytest plugin through the `pytest11` entry point, which imported the package
before pytest-cov began measuring and made coverage read 71% instead of 100%. The plugin went with the
evaluation files (D38), so the original reason is gone. `coverage run -m pytest` stays the documented
command because it is correct either way and CI already uses it. `fail_under = 95` guards the number.

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
call changes its id.** Since the id is what groups a decision's log lines across runs and deploys,
an unrelated edit would silently split its history in two, with nothing in the data to indicate why
the series stopped.

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

`OPENAI_API_KEY` is deliberately **not** treated the same way (D39). Plenty of machines have it set
for other reasons, so finding it is not a statement that `gut` should spend it.

Every real backend is resolved through a module `__getattr__` on both `gut` and `gut._backends`, so
`import gut` pulls in no HTTP client, no vendor SDK and no PyTorch. A test runs the import in a fresh
interpreter and checks.

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
whole trade: reading the subject is the expensive part, and every backend reads it once per batch --
Jev bills on input and reads the state once per request, `TransformersBackend` computes the subject's
key-value cache once (D43), a server with prefix caching does the same on its side. Five questions in
one call cost barely more than one, while five calls cost five subjects. If a question is expensive for
reasons other than the subject, keep it out of a decorated function.

Coroutine functions are batched too; see D27.

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

**Date:** 2026-09-20 · **Superseded by D42**

The two rules survive -- nothing is observed unless asked, and observing a decision can never break
it -- but the sink protocol, the JSONL file and `resolve()` do not. They existed to feed calibration.

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

The first demo's own costs were wrong in exactly this way: 101 tickets, zero sent to a human, and no
error anywhere.

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
It is a much larger change -- an `AsyncBackend` protocol, an async path through cache and batching --
and is noted below rather than started. `judge()` is still synchronous for the same reason.

## D28 — An agent skill, tested like code

**Date:** 2026-09-20

Much of the code that will use `gut` is going to be written by coding agents and reviewed by people.
An agent reading a 500-line README to write three lines is the wrong shape, so `skills/gut/SKILL.md`
is the short version: the correct spellings, the traps, and the rules of thumb. `llms.txt` at the
root is the index, following the convention this project's own research depended on when reading
the vendor's docs.

Both are **tested against the code**, harder than prose usually is, because their reader will not
notice a mistake and argue about it:

- every Python example compiles, and the offline setup snippet is executed;
- every API name the skill mentions must still be in `gut.__all__`;
- the values it says are accepted — `stakes="low"|"medium"|"high"` — are checked against
  `STAKES_CERTAINTY` and `LEAN_THRESHOLD`, so adding a level without documenting it fails;
- the backends it recommends are the ones `gut` exports;
- the `SyntaxError` it quotes is produced by compiling the bad spelling;
- the ceiling formula it states is compared against `max_useful_cost_human`;
- every file `llms.txt` links to must exist.

Documentation that can drift silently is worse than none, and this is the documentation most likely
to be acted on without a second look.

## D30 — The README is a pitch; the manual lives in `docs/`

**Date:** 2026-09-20

The README had grown to 574 accurate lines, and after the first screen it was `match` syntax, cache
keys and cassette warnings. All of that earns its place somewhere — just not in the file that has
thirty seconds to explain what problem this solves.

So the README is the pitch: the problem, why the existing answers hurt, one line of code, what it is
good for, the third branch, the backends, and links. The pages under `docs/` carry everything else.

The guarantee that documentation runs got **stronger** rather than weaker in the move. Every Python
block in the README and in every `docs/` page is now *executed*, not merely compiled, against a
`FakeBackend` in a temporary directory, with a prelude supplying the names an illustrative snippet
expects its reader to have (`email`, `ticket`, `Team`). A page runs top to bottom in one namespace,
the way it is read, and the backend is re-seeded between blocks so a snippet that reconfigures `gut`
cannot break the next one. The sandbox sets a dummy API key so `JevBackend(...)` constructs, and an
unroutable base URL so that if a snippet ever tried to *call* it, the test would fail instead of
reaching the real API.

Two checks exist to stop the README growing back: it must stay under 120 lines, and naming
`cost_false_yes`, `on_unsure` or `SQLiteCache` in it is a failure that names the page each belongs
to.

## D38 — `gut` is a coding primitive for small models, not a study of one

**Date:** 2026-09-27

Until now every number in this repository measured one model. Five public benchmarks, a calibration
toolkit, evaluation files and a record/replay layer all answered the question *how good is Jev at
this?* That is a real question, and it is the model vendor's to answer. It is not what a library is
for.

What `gut` is for is the gap between two ways of writing code. Classic code cannot answer "is this a
bug report?"; a frontier LLM can, but it is seconds and cents per call, returns prose to be parsed,
and is absurd overkill for a yes/no question. Between those sits a whole class of small, fast, cheap
models -- NLI encoders, sub-billion-parameter language models, Jev -- that are good enough for
exactly these questions. What is missing is the primitive that makes any of them one line of code.
That is `gut`.

So, removed: `benchmarks/`, `gut eval`, `gut calibrate`, the calibrators, the pytest plugin,
evaluation files, cassettes, the decision log's resolutions, and the demo whose purpose was to be
measured. Added: backends for the models this is actually about (D39, D43), a cascade between them
(D41), and a hook for seeing what was decided (D42). What stays is everything that makes the call
site good: the three question shapes, `YES / NO / UNSURE`, the posture words and the cost rule
under them, batching, caching.

**The test suite stays.** It tests `gut`, not a model: that a cost of 2 against 50 puts the
threshold at 0.038, that five judgments become one request, that the batched answer from a shared
key-value cache equals the unbatched one. That is the library's job.

## D39 — Text models answer by label probability, and API keys stay where they belong

**Date:** 2026-09-27

Most small models are text models. They do not answer typed questions; they predict a next token.
`gut` turns each question into a prompt whose first answer token is a label -- `Yes`/`No`, a letter
per option, a digit per level -- and reads the probability of each label straight from the model's
next-token distribution. Nothing is generated and nothing is parsed. The same function turns those
probabilities into the same `NoulAnswer` / `ChoiceAnswer` / `ScoreAnswer` Jev returns, so nothing
above the backend knows the difference.

Decisions inside that, each chosen to fail loudly rather than quietly:

- **Only the labels count.** Probability on other tokens is dropped and the rest renormalised. If no
  label appears among the likeliest tokens at all, that is a `BackendError` naming the tokens that
  did -- usually a thinking model spending its first token on `<think>` -- never a silent `0.5`.
- **A label a server did not list gets the most it could have had.** An OpenAI-compatible server
  reports its top 20 tokens. A missing label gets `min(least listed probability, even share of the
  unlisted mass)` -- an upper bound, so the answer looks *less* certain, the safe direction.
- **Only single-token spellings count.** In Qwen's vocabulary `" 0"` is a bare space followed by
  `0`; counting the space for the label `0` would count every digit's leading space too. Found by
  running a real tokenizer, not by reading about one.
- **Letters cap choices at 26** on these backends, with a `QuestionError` pointing at the backends
  that have no cap.
- **The subject comes first and the question last**, so every question about one subject shares a
  prefix.

`OpenAICompatibleBackend` speaks the chat completions protocol over `httpx` with `max_tokens=1` and
`logprobs`, which covers OpenAI's non-reasoning models and anything served by Ollama (0.12.11+),
vLLM or llama.cpp. **`OPENAI_API_KEY` is only sent to the server it belongs to** -- the
OpenAI default, or `OPENAI_BASE_URL` if set. Point the backend at any other `base_url` and the
environment's key stays home unless you pass `api_key=` yourself. Sending a secret to whatever URL a
config line names is how keys leak.

## D40 — Read every yes/no question both ways round

**Date:** 2026-09-27

The first run against a real 0.5B model was sobering. Asked *"Judge this about the subject: is spam.
Answer Yes or No"*, Qwen2.5-0.5B called a meeting request spam at 0.69. Offered the same question as
*"No or Yes"*, it said 0.04. A small model leans hard towards whichever label it is offered first --
the probabilities were mostly measuring the prompt.

Two changes, both chosen on twenty obvious sentence/claim pairs written to separate prompt artefacts
from model quality (not a benchmark: the question was which *prompt* lets a model's own judgment
through):

- **Balanced readings.** A yes/no question is asked with its labels in both orders and the two
  readings averaged; a choice is asked with its options in both orders and averaged *by option*, so
  B-in-one-order and the same option under another letter count together. A rating's scale has a
  real order and is asked once. On `TransformersBackend` the second reading costs a few dozen tokens
  against the shared prefix; on `OpenAICompatibleBackend` it is a second request, and `balanced=False`
  turns it off for a model known not to need it.
- **A claim about "the text".** The question is framed as *"Claim: the text is spam. Is the claim
  true?"*, with a bare predicate given a subject. *"The subject is spam"* scored worse: a small model
  reads it as a claim about an email's subject line.

| Qwen3-0.6B, 20 pairs | mean p when true | mean p when false | right side of 0.5 | ranking (AUC) |
|---|---|---|---|---|
| "Judge this about the subject", one order | 0.95 | 0.78 | 0.55 | 0.95 |
| "Judge this about the subject", both orders | 0.75 | 0.53 | 0.80 | 0.81 |
| "Claim: the text …", one order | 0.95 | 0.42 | 0.80 | 0.97 |
| "Claim: the text …", both orders | 0.81 | 0.30 | 0.95 | 0.93 |

Read it for what each change does. Both framings *rank* the pairs well in one order. What differs is
where the probabilities sit: asked to "judge", the model gave false claims 0.78 on average, so
almost everything came out yes. The claim framing halves that lean; the balanced reading removes
most of the rest, at a small cost in ranking. Where the probabilities sit is what `gut` decides on --
a threshold, a band, a cost -- so that is the column that matters here.

That table is about the prompt, measured on one model with twenty sentences. It says nothing about
how good Qwen3 is, and is not meant to.

## D41 — `Cascade`: the cheap model first, and only the unsure answers go further

**Date:** 2026-09-27

Most judgments a program makes are easy, and a 70M-parameter NLI model gets the easy ones right in
milliseconds for free. `Cascade(a, b, ...)` asks each backend in order and keeps an answer as soon as
it is decisive -- a probability outside `unsure_band`, or a confidence at or above `min_confidence`
-- escalating only the rest. It is the same idea as `ask_human`, one level down: UNSURE means *ask
someone better*, and that someone can be a bigger model before it is a person.

It is an ordinary `Backend`, so it composes with everything: `@semantic` hands it a whole batch and
only the unsettled questions travel on; the cache stores what it returns; and because a response can
now carry **per-answer models** (`BackendResponse.models`), `Decision.model` names the model that
actually answered each question rather than the cascade.

Choices worth naming:

- **The band is the cascade's own**, separate from the caller's posture. The final answer still
  goes through `stakes` / `ask_human`, so what the last model is unsure of can still reach a person.
- **Every stage and both thresholds are in `model_id`**, which is part of the cache key: changing the
  band changes which answer comes back, so it must not be served from an entry made under the old one.
- **A stage that fails is treated like one that was unsure** -- its questions move on. So a cascade
  doubles as a fallback chain: a local model that cannot take a 40-option choice hands it to one that
  can. The last stage's errors are raised; there is nobody left to ask.
- **`answered_by`** counts answers per model. It is the one number that shows what the cascade saves.

## D42 — One `on_decision` hook instead of sinks

**Date:** 2026-09-27

The decision log (D16) had a `Sink` protocol, three implementations, a record type, a resolution
type and `resolve()`, because it was the data path for calibration. Without calibration, what is left
is observability, and the primitive for that is a callback:

```python
gut.configure(on_decision=lambda d: logger.info("gut", extra=d.to_dict()))
```

`Decision.to_dict()` is the ready-made record: kind, id, outcome, question, model, source, latency,
and the kind-specific fields -- `p` and the policy in words for `likely`, the member and distribution
for `classify`, the score and nearest level for `rate`. Writing it to a JSONL file, a metrics counter
or a trace span is one line in the caller's code, where the choice belongs. The two rules of D16
still hold: nothing happens unless asked, and a hook that raises is logged and swallowed.

## D43 — Local models: one pass over the subject, float32 on a CPU, thinking off

**Date:** 2026-09-27

`TransformersBackend` and `ZeroShotBackend` run in-process through the optional `local` extra.

**One pass over the subject.** A batch's prompts are tokenized in full and their longest common
prefix found on token ids -- not on strings, so tokenizer boundaries cannot cause a mismatch. The
prefix runs once; its key-value cache is repeated per question and every remainder runs against it in
**one** right-padded forward pass. No attention mask is needed: under causal attention a real token
never sees the padding after it, and every row's positions continue from the same prefix. A test
against a real model checks the batched answers equal the unbatched ones to 1e-4. Measured on an
8 GB M-series laptop, five questions about one ticket went from 13.4 s to 2.6 s on the CPU and from
1.4 s to 0.9 s on the GPU.

**float32 on a CPU.** `transformers` loads Qwen3 in bfloat16 by default, which on a laptop CPU took
3.4 s per question; float32 took 1.8 s. The backend picks float32 for a CPU unless told otherwise.

**Thinking off.** The answer is read from the first token, so a model that opens with `<think>` has
nothing to read. `enable_thinking=False` is passed to every chat template; templates that do not
know the switch ignore it.

**The default model is Qwen3-0.6B**, over Qwen2.5-0.5B, because the latter could not be prompted
into seeing obvious spam in any wording tried (D40). Neither is gated, and both are Apache-2.0.

`ZeroShotBackend` needs no prompt at all: an NLI model scores whether the subject *entails* a
hypothesis. A yes/no question is one hypothesis (a bare predicate becomes "This text is ..."); a
choice or a rating is one hypothesis per option, with entailment log-odds competing through one
softmax. It reads "A or B" claims poorly -- ask two questions -- and that is documented rather than
worked around.

The `torch`-touching lines are excluded from the coverage floor, since CI cannot download models.
Everything they feed is pure and tested; `tests/test_local_models.py` runs them end to end wherever
the extra is installed.

## D44 — A catch-all can be called `DIGER`

**Date:** 2026-09-27

`classify` warns when an enum has no "none of these" member, because a forced choice among options
that do not fit is the most common way a classifier quietly goes wrong. The check knew six English
names, so `class Kategori(Enum): ... DIGER = "başka"` got the warning while doing the right thing.
Names are now compared after upper-casing and stripping accents -- `Diğer`, `DİĞER` and `DIGER` are
one name -- against a list covering English, Turkish, German, Dutch, French, Spanish, Portuguese and
Italian.

## D45 — `judge()` takes the posture words

**Date:** 2026-09-27

`gut.likely(...)` took `stakes` / `lean` / `ask_human`; `j.likely(...)` inside a `judge()` block took
only raw costs, so batching a question meant giving up the words for how careful to be. The
methods now go through the same resolution as the module functions, including the error for mixing
words with numbers.

## D46 — Published as `gutfeel`, imported as `gut`

**Date:** 2026-09-28

The first upload was refused: `gut` on PyPI is an existing project with no files, owned by someone
else. D1 recorded the name as available; it had only checked that no release existed, which is not
the same thing.

The distribution is `gutfeel` -- after the *gut feeling* D1 took the name from -- and nothing else
changes: the import, the module directory, every API name and every page of documentation still say
`gut`, the way `scikit-learn` installs `sklearn`. Install commands and the missing-extra hints say
`gutfeel[...]`, because those are the only places a user types the distribution name. Asking for
`gut` under PEP 541 remains possible later; a rename of the distribution would then touch only
`pyproject.toml` and those commands.

## D47 — Releases publish themselves, with no token anywhere

**Date:** 2026-09-28

0.1.0 went to PyPI by hand, with a token pasted into a terminal. From 0.2.0 on, pushing a `v*` tag
runs `release.yml`: it refuses a tag that does not match `pyproject.toml`, runs the same checks as
CI, builds, and publishes through PyPI's trusted publishing, so no secret exists to leak, rotate or
paste. A test pins `gut.__version__` to the `pyproject.toml` version, since the two are written by
hand and the tag check reads only one of them.

The README's links are absolute GitHub URLs. PyPI renders the README on its own, so a relative link
to `docs/backends.md` was a dead link on the page most new users see first; a test now refuses a
relative link there and still checks every absolute one lands on a file that exists.

## Retired

Retired on 2026-09-27 with the change of direction (D38). Kept here so the numbers stay meaningful;
the full entries are in the git history.

| | what it decided | why it went |
|---|---|---|
| D17 | Cassettes key per question, and a replay miss is an error | record/replay existed for the benchmarks; `FakeBackend` covers offline development |
| D18 | `min_accuracy` defaults to 1.0 in evaluation files | evaluation files removed |
| D22 | Thirty examples before a calibration number means anything | calibration removed |
| D23 | What the first real calibration run found | a measurement of one model, not a property of `gut` |
| D24 | Isotonic calibration by default | calibration removed |
| D25 | Corrections keyed per question and per model | calibration removed |
| D26 | What fitting produced | calibration removed |
| D29 | `stakes` on `rate` warns, citing D23 | the warning cited a measurement of one model and a command that no longer exists |
| D31–D37 | The benchmark datasets, method, wording, and results | benchmarks removed: they measured Jev, which is Jev's job |

## Next steps, noted and not started

- A native `async` backend protocol, so batched judgments need no worker thread, and an `async`
  `judge()`. The OpenAI-compatible backend is the obvious first implementation.
- Letting a local backend batch *across* subjects -- many tickets at once -- not only across the
  questions about one.
- A GGUF backend through `llama-cpp-python`, for machines without PyTorch.
- A written specification separate from the docs.
