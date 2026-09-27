# gut

**Judgment calls as one line of Python, on models small enough to put inside an `if`.**

Your code keeps running into questions that aren't logic: *Is this comment spam? Which team owns
this ticket? How urgent is it? Is the agent's task done?* Until now there were three answers:

- **Regex and keyword rules** — free and instant, and wrong the moment someone phrases it differently.
- **A frontier LLM** — understands anything, at seconds and cents a call, with prose to parse.
- **Train a classifier** — cheap to run, once you have the labelled data, the pipeline and the week.

There is a fourth: **small models**. A 70M-parameter NLI model answers "is this spam?" in about a
tenth of a second on a laptop CPU; a 0.6B language model, in under a second on a laptop GPU.
Hosted models built for exactly this, like Jev, need no hardware at all. They are good enough for
these questions — but each speaks its own API, and none of them hands you a decision.

`gut` is the primitive that does:

```python
import gut

if gut.likely(comment, "is spam"):
    hide(comment)
```

No prompt, no parsing, no threshold — and no model named in your code.

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

## Any model

The model is configuration, not code. Change it and nothing else changes:

```python
gut.configure(backend=gut.ZeroShotBackend())                       # NLI model, on your CPU
gut.configure(backend=gut.TransformersBackend("Qwen/Qwen3-0.6B"))  # small LLM, on your machine
gut.configure(backend=gut.OpenAICompatibleBackend(                 # Ollama, vLLM, llama.cpp
    "qwen2.5:1.5b", base_url="http://localhost:11434/v1"))
gut.configure(backend=gut.OpenAICompatibleBackend("gpt-4.1-nano")) # OpenAI
gut.configure(backend=gut.JevBackend())                            # TypeSafe AI's Jev
```

Or several at once. `Cascade` asks the cheapest model first and passes on only what it is unsure of:

```python
gut.configure(backend=gut.Cascade(
    gut.ZeroShotBackend(),                        # free and local: settles the obvious
    gut.OpenAICompatibleBackend("gpt-4.1-nano"),  # sees only what the first could not
))
```

Answers are read from each model's own probabilities, never parsed from text, and `decision.model`
names the model that gave one. Your own model can be a backend too: [here is how](docs/backends.md).

## Several questions, one pass

```python
@gut.semantic
def handle(ticket):
    if gut.likely(ticket, "is a bug report"):
        ...
    elif gut.likely(ticket, "asks for a refund"):   # already answered
        ...
```

Every judgment about `ticket` goes to the model together: one request to a hosted model, one pass
over the ticket for a local one.

## Install

```bash
pip install gutfeel                  # any OpenAI-compatible server; FakeBackend for tests
pip install "gutfeel[local]"         # + ZeroShotBackend and TransformersBackend (PyTorch)
pip install "gutfeel[jev]"           # + JevBackend
```

The package on PyPI is `gutfeel` (`gut` was taken); the import is plain `import gut`.
No model at hand? `gut.FakeBackend(answers={"is spam": 0.97})` answers from fixtures, for tests.

## Docs

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, pick a backend, and the three questions. |
| [Backends](docs/backends.md) | Every model `gut` can run on, `Cascade`, and writing your own. |
| [Knowing when it doesn't know](docs/knowing-when-it-doesnt-know.md) | `lean`, `ask_human`, `stakes`, and what `if` and `match` do with `UNSURE`. |
| [Asking everything at once](docs/batching.md) | `@semantic` and `judge()`: every judgment about one subject, together. |
| [Exact costs](docs/exact-costs.md) | The cost model under the posture words. |
| [Caching and observability](docs/caching-and-observability.md) | The cache, and seeing every decision as it is made. |
| [Honest limitations](docs/limitations.md) | What small models get wrong, and what `gut` does not do. |

[`examples/`](examples/) runs the same code on every backend. Coding agents: read [`SKILL.md`](skills/gut/SKILL.md).

## Status

Pre-1.0. Every code block in these docs runs in the test suite; every decision is in [DECISIONS.md](DECISIONS.md).

## License

Apache-2.0 · [contributing](CONTRIBUTING.md)
