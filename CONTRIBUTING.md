# Contributing

```bash
uv sync --extra jev
uv run pytest
uv run coverage run -m pytest && uv run coverage report
uv run mypy
uv run ruff check . && uv run ruff format --check .
```

The local backends have their own end-to-end tests, which download two small models (about 1.7 GB):

```bash
uv sync --extra jev --extra local
uv run pytest -m local tests/test_local_models.py
```

Ground rules:

- **The suite must pass with no model and no API key.** Anything that talks to a model goes behind
  `FakeBackend`, a fake HTTP transport, or the `live` / `local` markers.
- **Test `gut`, not a model.** How accurate some model is on some dataset is its maker's question.
  What belongs here is whether `gut` asked it the right thing and read its answer correctly (D38).
- **Never silently change user code behavior.** When `gut` cannot prove something statically
  (batching, in particular), it falls back to the slow-but-correct path rather than guessing.
- **A backend that cannot answer raises `BackendError`.** Never a made-up probability.
- Ambiguous design calls get a short entry in [DECISIONS.md](DECISIONS.md).
- Small commits whose messages say why, not what.
