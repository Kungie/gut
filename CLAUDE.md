# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`gut` is a Python library (PyPI name `gutfeel`, import name `gut`) that turns judgment calls into one
line — `gut.likely` / `gut.classify` / `gut.rate` — answered by a small model and returned as
YES / NO / UNSURE. The model is configuration (`gut.configure(backend=...)`), never named at the call
site. It also ships a CLI (`gut filter`, `gut map`) and an MCP server (`gutfeel-mcp`).

## Commands

Everything runs through `uv`; this is a uv workspace (root `gutfeel` + `packages/gutfeel-mcp`).

```bash
uv sync --extra jev                       # not --all-extras: `local` is PyTorch, deliberately not needed
uv run pytest                             # whole suite; needs no model and no API key
uv run pytest tests/test_rule.py          # one file
uv run pytest tests/test_rule.py::test_name   # one test
uv run coverage run -m pytest && uv run coverage report   # CI gate: fail_under = 95, branch coverage
uv run mypy                               # strict; covers src, tests, examples, scripts, the MCP package
uv run ruff check . && uv run ruff format --check .
uv run python scripts/build_site.py       # rebuild site/docs/*.html after touching docs/*.md
```

Opt-in tests, skipped by default:

```bash
uv sync --extra jev --extra local && uv run pytest -m local tests/test_local_models.py   # downloads ~1.7 GB of models
GUT_LIVE_BASE_URL=http://localhost:11434/v1 GUT_LIVE_MODEL=qwen2.5:1.5b uv run pytest -m live
```

CI runs lint + mypy once and the suite on Python 3.10–3.13. Code must stay 3.10-compatible
(`target-version = "py310"`), and `match` statements are used throughout.

## Architecture

The public API is whatever `src/gut/__init__.py` re-exports; every other module is `_private`.

**One question's path** (`_api.py`): `likely/classify/rate` build a backend-agnostic spec
(`_questions.py`: `NoulSpec` / `ChoiceSpec` / `ScoreSpec`, limits validated here before anything is
sent) → `_stored()` looks for a free answer, first in the prefetch scope (`_scope.py`), then the cache
(`_cache.py`, keyed by state + question + `backend.model_id`) → otherwise `backend.ask(state, {"q": spec})`
→ the raw answer becomes a `Decision` / `ChoiceDecision` / `ScoreDecision` (`_decision.py`) and is
handed to the `on_decision` hook. Each primitive has an async twin (`alikely`, …).

**Probability → outcome is one pure module.** `_rule.py` (`Policy.decide`) is the only place that
decides what to do with a probability: it picks the cheapest of say-yes / say-no / ask-a-human by
expected cost. `_posture.py` is a word layer on top (`stakes`, `lean`, `ask_human`) that maps to
those costs; presets are defined as bands with costs derived from them. A call may use words *or*
numbers, never both (`PolicyError`). `classify`/`rate` have no NO — only a confidence floor that
yields UNSURE.

**UNSURE is never silently coerced.** `bool(decision)` on UNSURE raises unless `on_unsure` is
configured (`_config.py`); the scoped override lives in a `ContextVar`.

**Backends** (`_backends/`) are the only code that knows how a question gets answered. The protocol
(`base.py`) is two members: `model_id` and `ask(state, questions) -> BackendResponse`. Optional
members a backend may add, discovered by `_many.py`: `ask_many` (batch across subjects), `aask` /
`aask_many` (native async). Anything without them is fanned out over a thread pool / worker thread.

- `jev.py` — TypeSafe's Jev via `typesafe-sdk`; answers typed questions natively. Also
  `JevBackend.openrouter()`, `.ollaya()`, `.clm()`. Does not implement retries: it configures the SDK's own.
- `openai.py` — any OpenAI-compatible chat server; requests one token with `logprobs`.
- `decisions.py` — OpenAI's Decisions API (`POST /v1/decisions`); typed answers like Jev's, over
  plain `httpx`. A `refusal` answer raises `BackendError`.
- `local.py` — `TransformersBackend` (causal LM, shared KV cache per subject) and `ZeroShotBackend` (NLI).
- `_labels.py` — shared by `openai.py` and `local.py`: turns a spec into a prompt whose first answer
  token is a label, and label probabilities back into a typed answer. Answers are read from the
  distribution, never generated or parsed, and each question is asked in both label orders and averaged.
- `cascade.py` — a backend made of backends; escalates only what a cheaper stage was unsure of.
- `fake.py` — `FakeBackend`, deterministic fixtures, strict by default, counts calls.

The heavyweight backends are lazy: `LAZY` in `_backends/__init__.py` plus module `__getattr__` in
both `__init__.py` files. `import gut` must not import `httpx`, `torch`, `transformers` or
`typesafe_sdk` — `tests/test_smoke.py` checks this in a subprocess.

**Batching: three front ends over one mechanism.** `@gut.semantic` (`_semantic.py`, static analysis of
the decorated function's source at decoration time), `gut.judge()` (`_judge.py`, lazy handles that
resolve everything registered so far on first read) and `gut.each()` (`_each.py`, one question over
many subjects) all ask up front, park answers in the prefetch scope, and let ordinary `likely(...)`
calls find them there. The scope is keyed by a fingerprint of the real state + question, so a wrong
plan simply misses and the normal path runs — batching can never change an answer. `_batching.py`
splits batches at the context limit; `_usage.py` counts calls/cost and enforces `max_cost`.

**Shipped programs** pick their backend from the environment via `_env.py` (`TYPESAFE_API_KEY`,
`GUT_BACKEND`, `GUT_MODEL`, `GUT_BASE_URL`), not from code: `_cli.py` (`gut filter` / `gut map`) and
`_mcp.py` (tools `likely`, `classify`, `rate`, `each`). `packages/gutfeel-mcp` contains no server
logic — it is only the `gutfeel-mcp` command plus dependencies so `uvx gutfeel-mcp` works alone.

## Things that are tested and will fail if they drift

- **Docs are executed.** `tests/test_docs.py` runs every Python block in `README.md` and `docs/*.md`
  top to bottom per page against a `FakeBackend` stand-in. Python must be in a labelled fence. The
  README must stay under 120 lines and may not mention `cost_false_yes`, `on_unsure` or `SQLiteCache`;
  it must show every backend. `docs/backends.md` needs a ``## `Name` `` section for every exported
  `*Backend` and `Cascade`. No page may mention benchmarks or calibration.
- **`site/docs/*.html` is generated** from `docs/*.md` by `scripts/build_site.py` (stdlib only) and
  committed; `tests/test_website.py` fails while it is stale. `site/index.html` is hand-written.
  `site/` is served live at gutpy.dev from `main` within minutes of a push.
- **`skills/gut/SKILL.md` and `llms.txt`** are checked by `tests/test_skill.py`: the skill's examples
  compile, every API name and accepted value it mentions must exist, and every file `llms.txt` links
  to must be there.
- **The version is repeated** in places that tests require to agree: `pyproject.toml` (version and the
  `mcp` extra's pin), `packages/gutfeel-mcp/pyproject.toml` (its version and its `gutfeel==` pin),
  `server.json`, `src/gut/__init__.py`, the README's PyPI badge URL, and the built site (rebuild it).
  Releasing is a `vX.Y.Z` tag push; `release.yml` refuses a tag that does not match.

Adding a backend therefore touches: the module, `LAZY` / `__all__` / the `TYPE_CHECKING` imports in
both `__init__.py` files, `_env.py` (`BACKENDS`, `NO_MODEL`), `docs/backends.md`, the README's backend
list, the stand-ins in `tests/test_docs.py`, then a site rebuild.

## Ground rules (from CONTRIBUTING.md)

- The suite must pass with no model and no API key. Model access goes behind `FakeBackend`, a fake
  HTTP transport (`httpx.MockTransport`), or the `live` / `local` markers.
- Test `gut`, not a model: whether it asked the right thing and read the answer correctly, never how
  accurate some model is.
- Never silently change user code behaviour. What cannot be proven statically (batching especially)
  falls back to the slow-but-correct path.
- A backend that cannot answer raises `BackendError`, never a made-up probability.
- Ambiguous design calls get their reasoning in the docstring of the code that makes them — modules
  here open with long docstrings explaining why; keep that up when changing behaviour.
- Small commits whose messages say why, not what.
- Code that needs `torch` is `# pragma: no cover`; everything feeding it (prompts, label ids, logit
  arithmetic) is pure and covered.
