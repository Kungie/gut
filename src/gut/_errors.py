"""Exception hierarchy.

Everything `gut` raises derives from `GutError`, so `except GutError` is enough to catch
anything the library originates. Where an error is also a misuse of a standard protocol the
specific class inherits from the matching builtin too, so existing `except ValueError` handlers
keep working.
"""

from __future__ import annotations


class GutError(Exception):
    """Base class for every error raised by `gut`."""


class PolicyError(GutError, ValueError):
    """A decision policy is contradictory, incomplete, or out of range.

    Raised at policy-construction time wherever possible, so the failure points at the call site
    that configured the costs rather than at the decision that later used them.
    """


class ConfigurationError(GutError, ValueError):
    """`gut` was configured with a value it does not recognise."""


class UnsureDecision(GutError):  # noqa: N818 - named by the public API, not the convention.
    """An UNSURE decision was used as a bool under the default `on_unsure="raise"` policy.

    This is the error the library exists to produce. Silently reading UNSURE as `False` is exactly
    the bug `gut` is meant to prevent, so the default is to stop and make the caller choose: handle
    the third branch with `match`, or opt into a coercion with `configure(on_unsure=...)` or the
    `on_unsure()` context manager.
    """

    def __init__(self, decision: object) -> None:
        self.decision = decision
        """The decision that could not be coerced."""
        super().__init__(
            f"Decision {getattr(decision, 'id', '?')!r} is UNSURE, and bool() has no honest "
            f"answer for that. Handle it explicitly (match on gut.UNSURE), or choose a coercion "
            f"with gut.configure(on_unsure=...) or the gut.on_unsure(...) context manager."
        )
