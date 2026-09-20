# gut

**A gut feeling that knows when to ask.**

Your code keeps running into questions that aren't logic: *Is this customer about to leave? Which
team should handle this ticket? Is this comment spam?* Until now there were three ways to answer
them, and all of them hurt:

- **Keyword rules and regexes** — fast and free, but brittle. They miss anything phrased differently.
- **Call an LLM** — smart, but seconds per call and a real bill at volume. And you still parse text.
- **Train your own classifier** — accurate and cheap to run, but you need labelled data and time.

`gut` gives your code a gut feeling instead. It runs on [Jev](https://docs.typesafe.ai), a new kind
of model that answers with probabilities instead of text — [up to hundreds of times faster and
cheaper](docs/why-jev.md) than an LLM, with no training.

```python
import gut

if gut.likely(email, "the customer threatens to cancel"):
    escalate(email)
```

One line. No prompt, no parsing, no threshold.

## What it's good for

```python
gut.likely(comment, "is spam or self-promotion")                  # moderation
gut.classify(ticket, Team)                                        # routing
gut.rate(message, ["can wait", "this week", "right now"])         # urgency
gut.likely(review, "mentions a safety problem with the product")  # monitoring
gut.likely(agent_state, "the task is finished")                   # agent loops
```

## And it knows when it doesn't know

A keyword rule never hesitates, and neither does an LLM. `gut` can:

```python
match gut.likely(email, "the customer threatens to cancel", ask_human=True):
    case gut.YES:    escalate(email)
    case gut.NO:     auto_reply(email)
    case gut.UNSURE: send_to_a_person(email)
```

Clear cases get handled automatically. Only the unclear ones reach a person.

## Install

```bash
pip install "gut[jev]"
export TYPESAFE_API_KEY=...
```

No key yet? `gut.configure(backend=gut.FakeBackend(...))` runs everything above offline — see
[getting started](docs/getting-started.md).

## Does it work?

On [CLINC150](docs/benchmarks.md), a public benchmark of 150 user intents plus deliberately
out-of-scope requests: a classifier **forced to pick an intent is wrong 26% of the time.** The same
model, allowed to answer "none of these" and to ask a person when it is unsure, is wrong **7.6%** —
while still handling **76%** of the traffic on its own.

```python
gut.classify(query, Intent, ask_human=True, stakes="medium")
```

It declines 89 of the 100 out-of-scope requests. On SMS spam a good-faith keyword filter errs 7.0%
and the same one-line judgment errs 1.4%. All of it cost 5.7 cents.

These are well-known public datasets and may be in the model's training data, so read the numbers
as optimistic — and measure your own task with [`gut eval`](docs/trusting-it.md). [The full
results](docs/benchmarks.md), including where `gut` loses to a trained model and where calibration
made things worse.

## Docs

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, run it offline, and the three questions you can ask. |
| [Knowing when it doesn't know](docs/knowing-when-it-doesnt-know.md) | How careful to be, in words rather than thresholds. |
| [Asking everything at once](docs/batching.md) | Ten judgments about one subject, in one request. |
| [Knowing whether to trust it](docs/trusting-it.md) | Test your judgments, measure calibration, and fix it. |
| [Exact costs](docs/exact-costs.md) | The cost model underneath, for when a mistake has a price tag. |
| [Caching, logging, and backends](docs/caching-and-logging.md) | The cache, the decision log, and the backend protocol. |
| [Why Jev](docs/why-jev.md) | What makes a judgment cheap enough to put inside an `if`. |
| [Benchmarks](docs/benchmarks.md) | What it does on three public datasets, and what it does not. |
| [Honest limitations](docs/limitations.md) | What it is bad at, and what its numbers do not mean. |

Writing `gut` code with a coding agent? Point it at [`skills/gut/SKILL.md`](skills/gut/SKILL.md).

## Status

Pre-1.0. Everything documented works and is covered by tests. Not done yet, deliberately: a native
`async` backend, a real public dataset to measure against, and fitting calibrators from resolved
production outcomes rather than eval files. Every design decision and its reasoning is in
[DECISIONS.md](DECISIONS.md).

## License

Apache-2.0 · [contributing](CONTRIBUTING.md)
