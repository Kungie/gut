# gut

[![PyPI](https://img.shields.io/pypi/v/gutfeel)](https://pypi.org/project/gutfeel/) [![CI](https://github.com/Kungie/gut/actions/workflows/ci.yml/badge.svg)](https://github.com/Kungie/gut/actions/workflows/ci.yml)

**Judgment calls as one line of Python — built for Jev, and running on any small model.**

[Try it in your browser →](https://kungie.github.io/gut/) The site runs gut's local model in the page: no key, no server.

Your code keeps running into questions that aren't logic: *Is this comment spam? Which team owns
this ticket? How urgent is it? Is the agent's task done?* Until now there were three answers:

- **Regex and keyword rules** — free and instant, and wrong the moment someone phrases it differently.
- **A frontier LLM** — understands anything, at seconds and cents a call, with prose to parse.
- **Train a classifier** — cheap to run, once you have the labelled data, the pipeline and the week.

There is a fourth: **a small model made for exactly these questions.** TypeSafe AI's
[Jev](https://docs.typesafe.ai) answers typed questions directly — a probability for yes, a
distribution over options, a score on a scale — with nothing to generate or parse, billed on input
only. `gut` is built around it, and makes it a line of code:

```python
import gut

gut.configure(backend=gut.JevBackend())   # or just set TYPESAFE_API_KEY

if gut.likely(comment, "is spam"):
    hide(comment)
```

No prompt, no parsing, no threshold — and no model named at the call site.

## Three questions

```python
gut.likely(ticket, "is a bug report")                      # yes / no
gut.classify(ticket, Team)                                 # which one — an Enum
gut.rate(ticket, ["can wait", "this week", "right now"])   # how much
```

## It knows when it doesn't know

A regex never hesitates, and neither does an LLM. `gut` can:

```python
match gut.likely(email, "the customer threatens to cancel", ask_human=True):
    case gut.YES:    escalate(email)
    case gut.NO:     auto_reply(email)
    case gut.UNSURE: send_to_a_person(email)
```

Say how careful to be in words — `lean="yes"`, `stakes="high"` — and `gut` works out the thresholds.

## A thousand subjects, one line

```python
spam = gut.each(comments).likely("is spam")     # one decision per comment, in order
teams = gut.each(tickets).classify(Team)
```

Jev gets concurrent requests, a local model batched passes, and nothing already cached is asked
twice. `@gut.semantic` does the same for several questions about one subject.

## Jev first, any model

Jev is the model `gut` is designed around. It is not the only one: the model is configuration, and
the same line runs unchanged on any of these.

```python
gut.configure(backend=gut.JevBackend())                            # TypeSafe AI's Jev
gut.configure(backend=gut.ZeroShotBackend())                       # NLI model, on your CPU
gut.configure(backend=gut.TransformersBackend("Qwen/Qwen3-0.6B"))  # small LLM, on your machine
gut.configure(backend=gut.OpenAICompatibleBackend(                 # Ollama, vLLM, llama.cpp
    "qwen2.5:1.5b", base_url="http://localhost:11434/v1"))
gut.configure(backend=gut.OpenAICompatibleBackend("gpt-4.1-nano")) # OpenAI
```

Or several at once. `Cascade` asks the cheapest model first and passes on only what it is unsure of:

```python
gut.configure(backend=gut.Cascade(
    gut.ZeroShotBackend(),   # free and local: settles the obvious
    gut.JevBackend(),        # sees only what the first could not
))
```

Every answer is a model's own probabilities, never parsed from text, and `decision.model` names the
model that gave it. Your own model can be a backend too: [here is how](https://github.com/Kungie/gut/blob/main/docs/backends.md).

## Install

```bash
pip install "gutfeel[jev]"           # + JevBackend
pip install "gutfeel[local]"         # + ZeroShotBackend and TransformersBackend (PyTorch)
pip install gutfeel                  # core: any OpenAI-compatible server; FakeBackend for tests
```

The package on PyPI is `gutfeel` (`gut` was taken); the import is plain `import gut`.
No model at hand? `gut.FakeBackend(answers={"is spam": 0.97})` answers from fixtures, for tests.

## Docs

| | |
|---|---|
| [Getting started](https://github.com/Kungie/gut/blob/main/docs/getting-started.md) | Install, pick a backend, and the three questions. |
| [Backends](https://github.com/Kungie/gut/blob/main/docs/backends.md) | Jev, every other model `gut` runs on, `Cascade`, and writing your own. |
| [Knowing when it doesn't know](https://github.com/Kungie/gut/blob/main/docs/knowing-when-it-doesnt-know.md) | `lean`, `ask_human`, `stakes`, and what `if` and `match` do with `UNSURE`. |
| [Asking everything at once](https://github.com/Kungie/gut/blob/main/docs/batching.md) | `each()`, `@semantic` and `judge()`: many subjects, or many questions, together. |
| [Async](https://github.com/Kungie/gut/blob/main/docs/async.md) | `await gut.alikely(...)` and friends: nothing blocks the event loop. |
| [Exact costs](https://github.com/Kungie/gut/blob/main/docs/exact-costs.md) | The cost model under the posture words. |
| [Caching and observability](https://github.com/Kungie/gut/blob/main/docs/caching-and-observability.md) | The cache, and seeing every decision as it is made. |
| [Honest limitations](https://github.com/Kungie/gut/blob/main/docs/limitations.md) | What small models get wrong, and what `gut` does not do. |

[`examples/`](https://github.com/Kungie/gut/tree/main/examples/) runs the same code on every backend. Coding agents: read [`SKILL.md`](https://github.com/Kungie/gut/blob/main/skills/gut/SKILL.md).

## Status and license

Pre-1.0, Apache-2.0. Every code block in these docs runs in the test suite ·
[contributing](https://github.com/Kungie/gut/blob/main/CONTRIBUTING.md)
