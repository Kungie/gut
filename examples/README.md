# Examples

The same code on any model. Each example takes `--backend`, and none of them names a model anywhere
else:

```bash
pip install "gutfeel[local]"

python examples/triage.py                     # a 70M-parameter NLI model, on this machine
python examples/triage.py --backend qwen      # Qwen3-0.6B, on this machine
python examples/triage.py --backend ollama    # OLLAMA_MODEL (default qwen2.5:1.5b) via Ollama
python examples/triage.py --backend openai    # gpt-4.1-nano; needs OPENAI_API_KEY
python examples/triage.py --backend jev       # TypeSafe's Jev; needs TYPESAFE_API_KEY
python examples/triage.py --backend fake      # no model: arbitrary but stable answers
```

| | |
|---|---|
| [`triage.py`](triage.py) | Support tickets: which queue, how soon, and whether a person should look first. Three judgments per ticket, batched by `@gut.semantic`. |
| [`moderation.py`](moderation.py) | Comments, through a `Cascade`: the local NLI model settles what it can, a bigger model sees the rest. |
| [`custom_backend.py`](custom_backend.py) | Your old keyword rules as a backend, in a page, and as the free first stage of a cascade. |

And three for fun, each a real use of a different part of `gut`:

| | |
|---|---|
| [`commit_roast.py`](commit_roast.py) | Sorts and roasts the commits of whatever repository you run it in: `gut.each()` for what kind of change each one is, plain Python for counting words. |
| [`meeting_or_email.py`](meeting_or_email.py) | Could this meeting have been an email? One concrete question, and `UNSURE` for the invites nobody can read. |
| [`recipe_or_memoir.py`](recipe_or_memoir.py) | Finds the recipe pages that open with a life story, through a `Cascade`: the NLI model settles six of eight judgments, a small LLM the rest. |

## What you will see

The models are small, and it shows -- which is what `UNSURE` and `Cascade` are for. One run of
`triage.py` on a laptop, for instance: the NLI model routed each ticket sensibly, and missed that
*"If it happens again we're moving to another vendor"* is a customer threatening to leave.
Qwen3-0.6B caught that one, but was unsure about the other four: it flagged each as a possible
churn risk, and sent three to the front desk rather than guess a team. In `moderation.py`, the NLI
model settled nine of the ten judgments on its own and passed one on.

Your numbers will differ with the model, the machine and the version. Try your own questions on
your own data before relying on any of them.

`tests/test_examples.py` runs all three with a `FakeBackend`, so they cannot quietly stop working.
