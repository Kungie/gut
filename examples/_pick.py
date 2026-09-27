"""Choose the model an example runs on. The examples themselves never name one -- that is the point.

Every example takes `--backend`:

    python examples/triage.py --backend nli      # a 70M-parameter NLI model, on this machine
    python examples/triage.py --backend qwen     # Qwen3-0.6B, on this machine
    python examples/triage.py --backend ollama   # whatever OLLAMA_MODEL names, via Ollama
    python examples/triage.py --backend openai   # gpt-4.1-nano
    python examples/triage.py --backend jev      # TypeSafe's Jev
    python examples/triage.py --backend fake     # no model: arbitrary but stable answers
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import gut

BACKENDS: dict[str, tuple[str, Callable[[], gut.Backend]]] = {
    "nli": ("a 70M-parameter NLI model, on this machine", lambda: gut.ZeroShotBackend()),
    "qwen": ("Qwen3-0.6B, on this machine", lambda: gut.TransformersBackend()),
    "ollama": (
        "a model served by a local Ollama",
        lambda: gut.OpenAICompatibleBackend(
            os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b"), base_url="http://localhost:11434/v1"
        ),
    ),
    "openai": ("gpt-4.1-nano", lambda: gut.OpenAICompatibleBackend("gpt-4.1-nano")),
    "jev": ("TypeSafe's Jev", lambda: gut.JevBackend()),
    "fake": ("no model at all", lambda: gut.FakeBackend(rule=gut.deterministic_rule)),
}


def backend_from_argv(description: str, *, default: str = "nli") -> gut.Backend:
    """Build the backend named by `--backend`, and say which one it is."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=default,
        help=", ".join(f"{name}: {label}" for name, (label, _) in BACKENDS.items()),
    )
    label, build = BACKENDS[parser.parse_args().backend]
    print(f"Answering with {label}.\n")
    return build()
