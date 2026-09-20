# gut documentation

[← back to the pitch](../README.md)

Read in this order if you are new. Jump straight to the page you need if you are not.

| | |
|---|---|
| [Getting started](getting-started.md) | Install, run it offline with no API key, and the three questions you can ask. |
| [Knowing when it doesn't know](knowing-when-it-doesnt-know.md) | `lean`, `ask_human`, `stakes` — how careful to be, in words. What `if` and `match` do with `UNSURE`. |
| [Asking everything at once](batching.md) | `@semantic` and `judge()`: every judgment about one subject in a single request. |
| [Knowing whether to trust it](trusting-it.md) | Example files, `gut eval`, calibration, and `gut calibrate`. What we measured, including where it failed. |
| [Exact costs](exact-costs.md) | The cost model underneath the posture words, its formula, and the trap in it. |
| [Caching, logging, and backends](caching-and-logging.md) | The cache, the decision log, and the backend protocol. |
| [Why Jev](why-jev.md) | What makes a judgment cheap enough to put inside an `if`, and the vendor's own numbers. |
| [Honest limitations](limitations.md) | What the model is bad at, what `gut` does not do, and what its numbers do not mean. |

Also worth knowing about:

- [`skills/gut/SKILL.md`](../skills/gut/SKILL.md) — the compact version, written for a coding agent.
  Point your agent at this rather than at the docs.
- [`llms.txt`](../llms.txt) — the machine-readable index.
- [`DECISIONS.md`](../DECISIONS.md) — every design decision and why, including the measurements that
  changed the API.
- [`examples/support_tickets/`](../examples/support_tickets/) — 101 labelled tickets, the same
  handler written with and without `gut`, and a sweep of every risk posture over the lot.

Every code block on these pages is executed by the test suite, so none of it can drift from the
library.
