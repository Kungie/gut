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
