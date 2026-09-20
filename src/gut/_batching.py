"""Turning many questions about one subject into one request.

Billing is on input tokens, so the expensive part of a request is the state, and it is paid for once
however many questions ride along. Asking five questions in one call costs barely more than asking
one; asking them in five calls costs five times the state. That asymmetry is the entire reason
`@semantic` and `judge()` exist.

The one thing that has to be respected is the context limit. A batch that would exceed it is split,
which means paying for the state again in each part -- still far better than one call per question.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from gut._backends.base import Backend
from gut._cache import CacheEntry, cache_key
from gut._config import current_cache
from gut._errors import BackendError
from gut._questions import QuestionSpec, State, canonical_json

CONTEXT_LIMIT_TOKENS = 64_000
"""Documented ceiling for state plus every question in one request."""

SINGLE_QUESTION_LIMIT_TOKENS = 32_000
"""Documented ceiling for state plus the single longest question."""

SAFETY_MARGIN = 0.9
"""Fraction of the limit actually used, since the estimate below is not a real tokeniser."""

CHARS_PER_TOKEN = 4
"""The usual rule of thumb. Deliberately crude: see `estimate_tokens`."""


def estimate_tokens(value: object) -> int:
    """Roughly how many tokens `value` will occupy.

    This is a character count divided by four, not a tokeniser. `gut` does not ship one and will not
    guess which one the backend uses. The estimate only ever decides *where to split a batch*, so
    being wrong costs an extra request or a rejected one, never a wrong answer -- and the safety
    margin absorbs ordinary error.
    """
    return max(1, len(canonical_json(value)) // CHARS_PER_TOKEN)


def _limit(backend: Backend, name: str, default: int) -> int:
    """Read a backend's own limit if it declares one, otherwise use Jev's documented ceiling."""
    value = getattr(backend, name, default)
    return int(value) if isinstance(value, int) and value > 0 else default


def plan_batches(
    state: State,
    specs: Mapping[str, QuestionSpec],
    backend: Backend,
) -> list[dict[str, QuestionSpec]]:
    """Split `specs` into batches that each fit alongside `state`.

    Questions are packed greedily in the order given. A single question too large to fit with the
    state is still emitted on its own: the backend is entitled to reject it, and silently dropping a
    question the caller asked for would be worse than an error they can read.
    """
    if not specs:
        return []

    budget = int(_limit(backend, "context_limit_tokens", CONTEXT_LIMIT_TOKENS) * SAFETY_MARGIN)
    state_tokens = estimate_tokens(state)

    batches: list[dict[str, QuestionSpec]] = []
    current: dict[str, QuestionSpec] = {}
    used = state_tokens
    for name, spec in specs.items():
        cost = estimate_tokens(spec.canonical())
        if current and used + cost > budget:
            batches.append(current)
            current = {}
            used = state_tokens
        current[name] = spec
        used += cost
    batches.append(current)
    return batches


def fetch(
    state: State,
    specs: Sequence[QuestionSpec],
    backend: Backend,
) -> dict[QuestionSpec, CacheEntry]:
    """Answer every spec about `state`, using the cache and as few calls as possible.

    Returns what it managed to answer. A spec missing from the result was not answered and the
    caller should fall back to asking it on its own, rather than a missing answer becoming a
    missing decision.
    """
    cache = current_cache()
    resolved: dict[QuestionSpec, CacheEntry] = {}
    outstanding: dict[str, QuestionSpec] = {}
    queued: set[QuestionSpec] = set()

    for index, spec in enumerate(specs):
        # Deduplicate against both halves: the same question twice in one batch is the same
        # question paid for twice.
        if spec in resolved or spec in queued:
            continue
        hit = cache.get(cache_key(state, spec, backend.model_id))
        if hit is not None:
            resolved[spec] = hit
        else:
            outstanding[f"q{index}"] = spec
            queued.add(spec)

    for batch in plan_batches(state, outstanding, backend):
        response = backend.ask(state, batch)
        for name, spec in batch.items():
            if name not in response.answers:
                raise BackendError(f"Backend answered without {name!r} for a batched question.")
            entry = CacheEntry(answer=response.answers[name], model=response.model)
            resolved[spec] = entry
            cache.set(cache_key(state, spec, backend.model_id), entry)

    return resolved
