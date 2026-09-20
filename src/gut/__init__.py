"""gut -- judgment as a programming primitive.

Some decisions in code are not logic but judgment: *is this a cancellation threat?*, *which team
owns this ticket?*, *how angry is this customer?* `gut` makes those first-class, as decisions that
can come back YES, NO, or UNSURE, and that you configure by what each kind of mistake costs you
rather than by a threshold you guessed.

The public API is re-exported here. Everything else is private and may move between releases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from gut._cassette import Cassette, CassetteBackend, record_requested
from gut._config import configure, on_unsure
from gut._decision import BaseDecision, ChoiceDecision, Decision, ScoreDecision
from gut._errors import (
    BackendError,
    CassetteMissError,
    ConfigurationError,
    EvalError,
    GutError,
    JudgeClosedError,
    PolicyError,
    QuestionError,
    UnsureDecision,
)
from gut._evals import EvalResult, EvalSuite, load_suite, run_suite
from gut._judge import Judge, Lazy, judge
from gut._log import (
    DecisionRecord,
    JSONLSink,
    MemorySink,
    NullSink,
    ResolutionRecord,
    Sink,
)
from gut._outcomes import NO, UNSURE, YES, Outcome
from gut._questions import ChoiceSpec, NoulSpec, ScoreSpec
from gut._rule import DEFAULT_POLICY, Policy, policy
from gut._semantic import Plan, PlannedQuestion, semantic

if TYPE_CHECKING:
    from gut._backends.jev import JevBackend as JevBackend

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
    "Cassette",
    "CassetteBackend",
    "CassetteMissError",
    "ChoiceAnswer",
    "ChoiceDecision",
    "ChoiceSpec",
    "ConfigurationError",
    "Decision",
    "DecisionRecord",
    "EvalError",
    "EvalResult",
    "EvalSuite",
    "FakeBackend",
    "GutError",
    "JSONLSink",
    "JevBackend",
    "Judge",
    "JudgeClosedError",
    "Lazy",
    "MemoryCache",
    "MemorySink",
    "NoulAnswer",
    "NoulSpec",
    "NullCache",
    "NullSink",
    "Outcome",
    "Plan",
    "PlannedQuestion",
    "Policy",
    "PolicyError",
    "QuestionError",
    "ResolutionRecord",
    "SQLiteCache",
    "ScoreAnswer",
    "ScoreDecision",
    "ScoreSpec",
    "Sink",
    "UnsureDecision",
    "__version__",
    "classify",
    "configure",
    "deterministic_rule",
    "judge",
    "likely",
    "load_suite",
    "on_unsure",
    "policy",
    "rate",
    "record_requested",
    "run_suite",
    "semantic",
]


def __getattr__(name: str) -> object:
    """Resolve `JevBackend` on first use.

    `typesafe-sdk` is an optional extra, and the pytest11 entry point means `import gut` runs in
    every pytest session of every project that installs it (D8). Neither should drag in a vendor
    SDK nobody asked for.
    """
    if name == "JevBackend":
        from gut._backends.jev import JevBackend

        return JevBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
