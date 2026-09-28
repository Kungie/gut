"""Choosing a backend from environment variables, for the programs gut ships: the command line
and the MCP server. A library user configures a backend in code instead.

| variable | meaning |
|---|---|
| `TYPESAFE_API_KEY` | use TypeSafe's Jev -- the default whenever it is set |
| `GUT_BACKEND` | one of `gut._env.BACKENDS`: `jev`, `openrouter`, `ollaya`, `ollama`, ... |
| `GUT_MODEL` | the model to ask, for backends that take one |
| `GUT_BASE_URL` | a server's URL, for `ollaya`, `openai` and `ollama` |
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from typing import Final

from gut._backends import Backend, FakeBackend, deterministic_rule
from gut._errors import ConfigurationError

BACKENDS: Final = (
    "jev",
    "openrouter",
    "ollaya",
    "openai",
    "ollama",
    "zeroshot",
    "transformers",
    "fake",
)
OLLAMA_URL: Final = "http://localhost:11434/v1"

NO_MODEL: Final = (
    "No model is configured. Set one of these in the environment:\n"
    '  TYPESAFE_API_KEY="..."                        TypeSafe\'s Jev\n'
    '  GUT_BACKEND="openrouter"                      Jev through OpenRouter (OPENROUTER_API_KEY)\n'
    '  GUT_BACKEND="ollaya", GUT_MODEL="winnow:e4b"  an open decision model, on Ollaya\n'
    '  GUT_BACKEND="ollama", GUT_MODEL="qwen3:0.6b"  a local Ollama server\n'
    '  GUT_BACKEND="openai", GUT_MODEL="..."         OpenAI, or any server via GUT_BASE_URL\n'
    '  GUT_BACKEND="zeroshot"                        a local NLI model, with gutfeel[local]\n'
    "See https://gutpy.dev/docs/backends.html"
)


def backend_from_env(env: Mapping[str, str]) -> Backend | None:
    """The backend named by `GUT_BACKEND`, or `None` to let gut choose (Jev, given a key).

    Raises:
        ConfigurationError: An unknown backend, or one that needs a model and got none.
        BackendError: The backend could not be built -- a missing extra, or a missing key.
    """
    name = env.get("GUT_BACKEND", "").strip().lower()
    model = env.get("GUT_MODEL", "").strip() or None
    base_url = env.get("GUT_BASE_URL", "").strip() or None
    if not name:
        return None
    if name not in BACKENDS:
        raise ConfigurationError(
            f"GUT_BACKEND={name!r} is not a backend gut knows. Use one of: {', '.join(BACKENDS)}."
        )
    if name == "fake":
        return FakeBackend(rule=deterministic_rule)
    if name in ("jev", "openrouter", "ollaya"):
        from gut._backends.jev import JevBackend

        if name == "ollaya":
            if model is None:
                raise ConfigurationError(
                    "GUT_BACKEND=ollaya needs GUT_MODEL, the model to ask, such as winnow:e4b."
                )
            return JevBackend.ollaya(model, host=base_url or env.get("OLLAYA_HOST"))
        if name == "openrouter":
            return JevBackend.openrouter(model=model, api_key=env.get("OPENROUTER_API_KEY"))
        return JevBackend(model=model, base_url=base_url)
    if name in ("openai", "ollama"):
        if model is None:
            raise ConfigurationError(f"GUT_BACKEND={name} needs GUT_MODEL, the model to ask.")
        from gut._backends.openai import OpenAICompatibleBackend

        if name == "ollama":
            base_url = base_url or OLLAMA_URL
        return OpenAICompatibleBackend(model, base_url=base_url)
    from gut._backends.local import (  # pragma: no cover - loads a model
        TransformersBackend,
        ZeroShotBackend,
    )

    if name == "zeroshot":  # pragma: no cover
        return ZeroShotBackend(model) if model else ZeroShotBackend()
    return TransformersBackend(model) if model else TransformersBackend()  # pragma: no cover


class FromEnvironment:
    """The backend the environment asks for, built once, on first use."""

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._env = env
        self._lock = threading.Lock()
        self._built = False
        self._backend: Backend | None = None

    def get(self) -> Backend | None:
        with self._lock:
            if not self._built:
                self._backend = backend_from_env(os.environ if self._env is None else self._env)
                self._built = True
            return self._backend
