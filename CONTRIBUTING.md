# Contributing

```bash
uv sync --all-extras
uv run pytest
uv run coverage run -m pytest && uv run coverage report
uv run mypy
uv run ruff check . && uv run ruff format --check .
```

Ground rules:

- **The suite must pass with no `TYPESAFE_API_KEY`.** Anything that talks to Jev goes behind
  `FakeBackend`, a cassette, or the `live` marker.
- **Never silently change user code behavior.** When `gut` cannot prove something statically
  (batching, in particular), it falls back to the slow-but-correct path rather than guessing.
- Ambiguous design calls get a short entry in [DECISIONS.md](DECISIONS.md).
- Small, well-described commits.
