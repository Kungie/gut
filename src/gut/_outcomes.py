"""The three outcomes a decision can have.

`Outcome` is an `enum.Enum` rather than a set of module-level sentinels because only a *dotted*
name is a value pattern in a `match` statement. A bare ``case YES:`` is a capture pattern: it binds
anything and makes the following clauses unreachable, which the compiler rejects outright. See D7
in DECISIONS.md.
"""

from __future__ import annotations

import enum
from typing import Final


class Outcome(enum.Enum):
    """What a decision resolved to.

    Ordered `NO < UNSURE < YES` by `rank`, which is the order they occupy as the probability of
    yes rises. Enum members are never compared by `<` directly; `rank` exists so that callers and
    tests can talk about that ordering explicitly.
    """

    NO = "no"
    UNSURE = "unsure"
    YES = "yes"

    @property
    def rank(self) -> int:
        """Position in the `NO` → `UNSURE` → `YES` progression, as the probability of yes rises."""
        return _RANKS[self]

    def __str__(self) -> str:
        return self.value


_RANKS: Final[dict[Outcome, int]] = {Outcome.NO: 0, Outcome.UNSURE: 1, Outcome.YES: 2}

YES: Final = Outcome.YES
NO: Final = Outcome.NO
UNSURE: Final = Outcome.UNSURE
