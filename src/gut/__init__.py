"""gut -- judgment calls as one line of Python, on small, fast, cheap models.

Code keeps running into questions that are not logic: *is this comment spam?*, *which team owns this
ticket?*, *how urgent is it?* A regex is too brittle for them and a frontier LLM is too slow and too
expensive to put inside an `if`. Small models made for these questions are fast and cheap enough --
above all TypeSafe AI's Jev, which `gut` is built around, but also a local NLI encoder or a 0.6B
language model. `gut` is the primitive that makes any of them a single line, and lets that line
answer that it does not know:

```python
if gut.likely(comment, "is spam or self-promotion"):
    hide(comment)

match gut.likely(email, "the customer threatens to cancel", ask_human=True):
    case gut.YES:    escalate(email)
    case gut.NO:     auto_reply(email)
    case gut.UNSURE: send_to_a_person(email)
```

The model is configuration, not code: `gut.configure(backend=...)` with any `Backend`.

The public API is re-exported here; everything else is private and may move between releases.
Documentation lives in `docs/`, and `skills/gut/SKILL.md` is the short version for coding agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gut._api import classify, likely, rate
from gut._backends import (
    LAZY,
    Answer,
    Backend,
    BackendResponse,
    Cascade,
    ChoiceAnswer,
    FakeBackend,
    NoulAnswer,
    ScoreAnswer,
    deterministic_rule,
)
from gut._cache import Cache, CacheEntry, MemoryCache, NullCache, SQLiteCache
from gut._config import configure, on_unsure
from gut._decision import BaseDecision, ChoiceDecision, Decision, ScoreDecision
from gut._each import Each, each
from gut._errors import (
    BackendError,
    ConfigurationError,
    GutError,
    JudgeClosedError,
    PolicyError,
    QuestionError,
    UnsureDecision,
)
from gut._judge import Judge, Lazy, judge
from gut._outcomes import NO, UNSURE, YES, Outcome
from gut._posture import Lean, Preset, Stakes, presets
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State
from gut._rule import DEFAULT_POLICY, Policy, policy
from gut._semantic import Plan, PlannedQuestion, semantic

if TYPE_CHECKING:
    from gut._backends.jev import JevBackend as JevBackend
    from gut._backends.local import TransformersBackend as TransformersBackend
    from gut._backends.local import ZeroShotBackend as ZeroShotBackend
    from gut._backends.openai import OpenAICompatibleBackend as OpenAICompatibleBackend

__version__ = "0.3.0"

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
    "Cascade",
    "ChoiceAnswer",
    "ChoiceDecision",
    "ChoiceSpec",
    "ConfigurationError",
    "Decision",
    "Each",
    "FakeBackend",
    "GutError",
    "JevBackend",
    "Judge",
    "JudgeClosedError",
    "Lazy",
    "Lean",
    "MemoryCache",
    "NoulAnswer",
    "NoulSpec",
    "NullCache",
    "OpenAICompatibleBackend",
    "Outcome",
    "Plan",
    "PlannedQuestion",
    "Policy",
    "PolicyError",
    "Preset",
    "QuestionError",
    "QuestionSpec",
    "SQLiteCache",
    "ScoreAnswer",
    "ScoreDecision",
    "ScoreSpec",
    "Stakes",
    "State",
    "TransformersBackend",
    "UnsureDecision",
    "ZeroShotBackend",
    "__version__",
    "classify",
    "configure",
    "deterministic_rule",
    "each",
    "judge",
    "likely",
    "on_unsure",
    "policy",
    "presets",
    "rate",
    "semantic",
]


def __getattr__(name: str) -> object:
    """Resolve the heavyweight backends on first use.

    Each one wraps something `import gut` should not pay for -- an HTTP client, a vendor SDK,
    PyTorch -- and two of them depend on optional extras that may not be installed at all.
    """
    if name in LAZY:
        import importlib

        return getattr(importlib.import_module(LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
