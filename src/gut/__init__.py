"""gut -- judgment as a programming primitive.

Some decisions in code are not logic but judgment: *is this a cancellation threat?*, *which team
owns this ticket?*, *how angry is this customer?* `gut` makes those first-class, as decisions that
can come back YES, NO, or UNSURE, and that you configure by what each kind of mistake costs you
rather than by a threshold you guessed.

The public API is re-exported here. Everything else is private and may move between releases.
"""

from __future__ import annotations

from gut._api import classify, likely, rate
from gut._backends import (
    Answer,
    Backend,
    BackendResponse,
    ChoiceAnswer,
    FakeBackend,
    NoulAnswer,
    ScoreAnswer,
    deterministic_rule,
)
from gut._cache import Cache, CacheEntry, MemoryCache, NullCache, SQLiteCache
from gut._config import configure, on_unsure
from gut._decision import BaseDecision, ChoiceDecision, Decision, ScoreDecision
from gut._errors import (
    BackendError,
    ConfigurationError,
    GutError,
    PolicyError,
    QuestionError,
    UnsureDecision,
)
from gut._outcomes import NO, UNSURE, YES, Outcome
from gut._questions import ChoiceSpec, NoulSpec, ScoreSpec
from gut._rule import DEFAULT_POLICY, Policy, policy

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_POLICY",
    "NO",
    "UNSURE",
    "YES",
    "Answer",
    "Backend",
    "BackendError",
    "BackendResponse",
    "BaseDecision",
    "Cache",
    "CacheEntry",
    "ChoiceAnswer",
    "ChoiceDecision",
    "ChoiceSpec",
    "ConfigurationError",
    "Decision",
    "FakeBackend",
    "GutError",
    "MemoryCache",
    "NoulAnswer",
    "NoulSpec",
    "NullCache",
    "Outcome",
    "Policy",
    "PolicyError",
    "QuestionError",
    "SQLiteCache",
    "ScoreAnswer",
    "ScoreDecision",
    "ScoreSpec",
    "UnsureDecision",
    "__version__",
    "classify",
    "configure",
    "deterministic_rule",
    "likely",
    "on_unsure",
    "policy",
    "rate",
]
