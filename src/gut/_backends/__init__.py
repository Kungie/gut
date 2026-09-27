"""Backends: the only part of `gut` that knows how a question actually gets answered.

`FakeBackend` and `Cascade` are plain Python. The others wrap something heavier -- an HTTP client,
a vendor SDK, PyTorch -- and are imported on first use, so `import gut` stays instant and never
fails for want of an optional extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gut._backends.base import (
    Answer,
    Backend,
    BackendResponse,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from gut._backends.cascade import Cascade
from gut._backends.fake import FakeBackend, RecordedCall, deterministic_rule

if TYPE_CHECKING:
    from gut._backends.jev import JevBackend as JevBackend
    from gut._backends.local import TransformersBackend as TransformersBackend
    from gut._backends.local import ZeroShotBackend as ZeroShotBackend
    from gut._backends.openai import OpenAICompatibleBackend as OpenAICompatibleBackend

LAZY: dict[str, str] = {
    "JevBackend": "gut._backends.jev",
    "OpenAICompatibleBackend": "gut._backends.openai",
    "TransformersBackend": "gut._backends.local",
    "ZeroShotBackend": "gut._backends.local",
}
"""Backends resolved on first attribute access, and the module each one lives in."""

__all__ = [
    "Answer",
    "Backend",
    "BackendResponse",
    "Cascade",
    "ChoiceAnswer",
    "FakeBackend",
    "JevBackend",
    "NoulAnswer",
    "OpenAICompatibleBackend",
    "RecordedCall",
    "ScoreAnswer",
    "TransformersBackend",
    "ZeroShotBackend",
    "deterministic_rule",
]


def __getattr__(name: str) -> object:
    """Import a heavyweight backend the first time someone asks for it."""
    if name in LAZY:
        import importlib

        return getattr(importlib.import_module(LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
