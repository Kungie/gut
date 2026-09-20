# Design decisions

A running log of choices made while building `gut`, especially the ambiguous ones. The rule we follow:
when a design question is genuinely ambiguous, pick the option that **never silently changes user code
behavior**, write it down here, and move on.

---

## D1 — Name: `gut` (was `maybe`)

**Date:** 2026-09-20

`maybe` was the working name in the handoff, but it is taken on PyPI by an abandoned 2016 CLI tool
(`p-e-w/maybe`, last release 2016-03-27, "See what a program does before deciding whether you really
want it to happen"). Taking it over is not an option and shipping under a colliding import name would
break anyone who has the old package installed.

Also checked and taken: `reckon`, `deem`, `hunch`, `verdict`, `judgment`, `judgement`, `discern`,
`probably`, `unsure`, `hedge`, `squint`, `inkling`, `appraise`, `mull`, `tilt`, `kinda`, `prob`,
`guts`, `pymaybe`.

**Chosen:** `gut`, from *gut feeling* — the everyday word for exactly the kind of fast, confident-but-
fallible judgment this library makes programmable. Free on PyPI and TestPyPI, not a Python keyword,
no stdlib collision, four letters, and it reads well at the call site (`gut.likely(...)`).

The name is referenced in exactly one place for packaging metadata (`pyproject.toml`) and one package
directory (`src/gut/`), so a later rename stays cheap.

## D2 — License: MIT

**Date:** 2026-09-20

The handoff says "open-source Python package" without naming a license. MIT is the conventional,
lowest-friction default for a small Python library and imposes nothing on adopters. Apache-2.0 would
add an explicit patent grant; if that becomes desirable before 1.0 it is a one-file change while the
contributor set is still one person.

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

## Facts to verify before coding the backend

The handoff's description of Jev predates this repo and **must not be trusted blind**. Verified so far:

- [x] `typesafe-sdk` exists on PyPI — version `0.7.0`, `requires-python >=3.10`,
      repo `github.com/typesafe-ai/typesafe-sdk-python`, docs `docs.typesafe.ai/sdk/python/`.
- [ ] `TypeSafeClient().system_one(state=..., questions={...})` signature and `.answers[id]` shape
- [ ] `Choice` / `Score` / `Noul` constructor arguments and response attributes
- [ ] Whether `state` genuinely accepts `str | dict | list[str]`
- [ ] Context limits (64k combined / 32k single question) and rate limits (250k tok/s, 1200 req/min)
- [ ] How the response reports the versioned model ID, and how to pin a version
- [ ] Error/retry semantics: 429 body, `retry-after` header

Differences found during inspection get written up here before any backend code is committed.

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
