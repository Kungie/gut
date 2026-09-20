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

---

## Implementation order

Per the handoff, kept in this order so the suite stays green at every step:

1. decision rule (pure, property-tested)
2. `Decision` types, truthiness, `match` semantics
3. `FakeBackend`
4. `likely` / `classify` / `rate`
5. cache
6. `JevBackend`
7. `@semantic` + `judge()`
8. decision log
9. testing tools (example files, pytest plugin, record/replay)
10. demo (`examples/support_tickets/`)
11. README pass
