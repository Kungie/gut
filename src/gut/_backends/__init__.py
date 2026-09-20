"""Backends: the only part of `gut` that knows how a question actually gets answered."""

from __future__ import annotations

from gut._backends.base import (
    Answer,
    Backend,
    BackendResponse,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from gut._backends.fake import FakeBackend, RecordedCall, deterministic_rule

__all__ = [
    "Answer",
    "Backend",
    "BackendResponse",
    "ChoiceAnswer",
    "FakeBackend",
    "NoulAnswer",
    "RecordedCall",
    "ScoreAnswer",
    "deterministic_rule",
]
