"""Backends: the only part of `gut` that knows how a question actually gets answered."""

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
from gut._backends.fake import FakeBackend, RecordedCall, deterministic_rule

if TYPE_CHECKING:
    from gut._backends.jev import JevBackend as JevBackend

__all__ = [
    "Answer",
    "Backend",
    "BackendResponse",
    "ChoiceAnswer",
    "FakeBackend",
    "JevBackend",
    "NoulAnswer",
    "RecordedCall",
    "ScoreAnswer",
    "deterministic_rule",
]


def __getattr__(name: str) -> object:
    """Resolve `JevBackend` on first use, so `typesafe-sdk` stays an optional extra."""
    if name == "JevBackend":
        from gut._backends.jev import JevBackend

        return JevBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
