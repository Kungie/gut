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

## The website

`site/` is published to <https://kungie.github.io/gut/> on every push to `main` that touches it.
`site/index.html` is written by hand; the documentation pages under `site/docs/` are built from
`docs/*.md`, which stay the only source. After changing a page in `docs/`, rebuild:

```bash
uv run python scripts/build_site.py
```

A test fails while the built pages are behind the markdown, so they cannot drift apart.

## Releasing

Bump the version in both `pyproject.toml` and `src/gut/__init__.py` (a test checks they agree),
rebuild the site so it shows the new version, commit, then tag and push:

```bash
git tag -a v0.2.0 -m "gutfeel 0.2.0" && git push origin main --tags
```

`.github/workflows/release.yml` checks the tag matches the version, runs the suite, builds, publishes
`gutfeel` to PyPI and creates a GitHub release. There is no token: PyPI trusts the workflow itself
(trusted publishing), configured once under the project's *Publishing* settings on pypi.org with
owner `Kungie`, repository `gut`, workflow `release.yml` and environment `pypi`.

## Ground rules

- **The suite must pass with no model and no API key.** Anything that talks to a model goes behind
  `FakeBackend`, a fake HTTP transport, or the `live` / `local` markers.
- **Test `gut`, not a model.** How accurate some model is on some dataset is its maker's question.
  What belongs here is whether `gut` asked it the right thing and read its answer correctly.
- **Never silently change user code behavior.** When `gut` cannot prove something statically
  (batching, in particular), it falls back to the slow-but-correct path rather than guessing.
- **A backend that cannot answer raises `BackendError`.** Never a made-up probability.
- Ambiguous design calls get their reasoning in the docstring of the code that makes them.
- Small commits whose messages say why, not what.
