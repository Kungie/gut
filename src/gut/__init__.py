"""gut -- judgment as a programming primitive.

Some decisions in code are not logic but judgment: *is this a cancellation threat?*, *which team
owns this ticket?*, *how angry is this customer?* `gut` makes those first-class, as decisions that
can come back YES, NO, or UNSURE, and that you configure by what each kind of mistake costs you
rather than by a threshold you guessed.

The public API is re-exported here. Everything else is private and may move between releases.
"""

from __future__ import annotations

from gut._config import configure, on_unsure
from gut._decision import BaseDecision, ChoiceDecision, Decision, ScoreDecision
from gut._errors import ConfigurationError, GutError, PolicyError, UnsureDecision
from gut._outcomes import NO, UNSURE, YES, Outcome
from gut._rule import DEFAULT_POLICY, Policy, policy

__version__ = "0.0.1"

__all__ = [
    "DEFAULT_POLICY",
    "NO",
    "UNSURE",
    "YES",
    "BaseDecision",
    "ChoiceDecision",
    "ConfigurationError",
    "Decision",
    "GutError",
    "Outcome",
    "Policy",
    "PolicyError",
    "ScoreDecision",
    "UnsureDecision",
    "__version__",
    "configure",
    "on_unsure",
    "policy",
]
