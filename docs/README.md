# gut documentation

[← back to the pitch](../README.md)

Read in this order if you are new. Jump straight to the page you need if you are not.

| | |
|---|---|
| [Getting started](getting-started.md) | Install, pick a model, and the three questions you can ask. |
| [Backends](backends.md) | Every model `gut` can run on -- local NLI, local LLMs, Ollama, vLLM, OpenAI, Jev -- `Cascade`, and writing your own. |
| [Knowing when it doesn't know](knowing-when-it-doesnt-know.md) | `lean`, `ask_human`, `stakes` -- how careful to be, in words. What `if` and `match` do with `UNSURE`. |
| [Asking everything at once](batching.md) | `@semantic` and `judge()`: every judgment about one subject, together. |
| [Async](async.md) | `alikely`, `aclassify`, `arate`, and the async side of `each()`, `@semantic` and `judge()`. |
| [Exact costs](exact-costs.md) | The cost model underneath the posture words, its formula, and the trap in it. |
| [Caching and observability](caching-and-observability.md) | The cache, and seeing every decision as it is made. |
| [Honest limitations](limitations.md) | What small models get wrong, and what `gut` does not do. |

Also worth knowing about:

- [`examples/`](../examples/) -- the same handlers running on any backend you pick from the command
  line, including a keyword rule turned into the first stage of a cascade.
- [`skills/gut/SKILL.md`](../skills/gut/SKILL.md) -- the compact version, written for a coding agent.
  Point your agent at this rather than at the docs.
- [`llms.txt`](../llms.txt) -- the machine-readable index.

Every code block on these pages is executed by the test suite, so none of it can drift from the
library.
