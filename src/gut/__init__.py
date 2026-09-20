"""gut -- a gut feeling that knows when to ask.

Your code keeps running into questions that are not logic: *is this customer about to leave?*,
*which team should handle this?*, *is this comment spam?* `gut` answers one of those in a line,
fast and cheap enough to put inside an `if`, and -- unlike a keyword rule or an LLM -- it can say
that it does not know.

```python
if gut.likely(email, "the customer threatens to cancel"):
    escalate(email)
```

Say how careful to be in words, and a third answer becomes possible:

```python
match gut.likely(email, "the customer threatens to cancel", lean="yes", ask_human=True):
    case gut.YES:    escalate(email)
    case gut.NO:     auto_reply(email)
    case gut.UNSURE: send_to_a_person(email)
```

The public API is re-exported here; everything else is private and may move between releases.
Documentation lives in `docs/`, and `skills/gut/SKILL.md` is the short version for coding agents.
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
from gut._calibrators import (
    CalibrationSet,
    Calibrator,
    Entry,
    Identity,
    Isotonic,
    Platt,
)
from gut._cassette import Cassette, CassetteBackend, record_requested
from gut._config import configure, on_unsure
from gut._decision import BaseDecision, ChoiceDecision, Decision, ScoreDecision
from gut._errors import (
    BackendError,
    CalibrationError,
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
from gut._posture import Lean, Preset, Stakes, presets
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
    "CalibrationError",
    "CalibrationSet",
    "Calibrator",
    "Cassette",
    "CassetteBackend",
    "CassetteMissError",
    "ChoiceAnswer",
    "ChoiceDecision",
    "ChoiceSpec",
    "ConfigurationError",
    "Decision",
    "DecisionRecord",
    "Entry",
    "EvalError",
    "EvalResult",
    "EvalSuite",
    "FakeBackend",
    "GutError",
    "Identity",
    "Isotonic",
    "JSONLSink",
    "JevBackend",
    "Judge",
    "JudgeClosedError",
    "Lazy",
    "Lean",
    "MemoryCache",
    "MemorySink",
    "NoulAnswer",
    "NoulSpec",
    "NullCache",
    "NullSink",
    "Outcome",
    "Plan",
    "PlannedQuestion",
    "Platt",
    "Policy",
    "PolicyError",
    "Preset",
    "QuestionError",
    "ResolutionRecord",
    "SQLiteCache",
    "ScoreAnswer",
    "ScoreDecision",
    "ScoreSpec",
    "Sink",
    "Stakes",
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
    "presets",
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
