"""Risk posture in words: `stakes`, `lean`, `ask_human`.

Exact costs are the honest way to configure a decision, and they are also the thing most people will
not do. The numbers are unfamiliar, the units are arbitrary, and the ratio that matters is not
obvious from either figure alone. So there is a middle layer: three words that say how you want the
decision to behave, which map onto costs so that the cost rule stays the only thing that decides
anything.

- **`lean`** says which mistake is worse, and fixes the probability at which yes overtakes no:
  `"yes"` → `0.25`, `None` → `0.5`, `"no"` → `0.75`.
- **`stakes`** says how bad any automatic mistake is next to a person looking at it, and only ever
  *widens* the band around that point. It has no effect without `ask_human=True`, and `gut` warns
  if you pass it anyway.
- **`ask_human`** decides whether `UNSURE` is a possible outcome at all. Off by default.

The two knobs are deliberately independent: changing how careful you are must not quietly change
which way you err.

**Every preset band is reachable by construction.** A cost policy sends a decision to a person
exactly for `p` in `[cost_human/cost_false_no, 1 - cost_human/cost_false_yes]`, which is a real
interval precisely when `lo < hi`. Presets are defined *as* bands and the costs derived from them,
so the unreachable-human trap (D19) cannot happen at this layer — it is not guarded against, it is
structurally impossible. See D20.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from gut._errors import PolicyError
from gut._rule import Policy

Stakes: TypeAlias = Literal["low", "medium", "high"]
"""How costly an automatic mistake is, compared to a person deciding instead."""

Lean: TypeAlias = Literal["yes", "no"]
"""Which way to err when in doubt: which of the two mistakes is worse."""

LEAN_THRESHOLD: Final[dict[Lean | None, float]] = {None: 0.5, "yes": 0.25, "no": 0.75}
"""Where yes overtakes no. Set by `lean` alone, never by `stakes`."""

STAKES_CERTAINTY: Final[dict[Stakes, float]] = {"low": 0.8, "medium": 0.5, "high": 0.2}
"""How much certainty the caller is willing to act on. Smaller means a wider band."""

DEFAULT_STAKES: Final[Stakes] = "medium"

HUMAN_COST: Final = 1.0
"""The unit the derived costs are expressed in. Only the ratios matter."""


def band_for(stakes: Stakes, lean: Lean | None) -> tuple[float, float]:
    """The probabilities at which this posture asks a person.

    Derived rather than tabulated, so the two knobs stay independent: `lean` fixes the point the
    band straddles and `stakes` fixes how far it reaches either side of it.
    """
    threshold = LEAN_THRESHOLD[lean]
    certainty = STAKES_CERTAINTY[stakes]
    low = certainty * threshold
    high_side = certainty - low
    return (low, 1.0 - high_side)


def _costs_for(band: tuple[float, float]) -> tuple[float, float]:
    """The `(cost_false_yes, cost_false_no)` whose UNSURE band is exactly `band`."""
    low, high = band
    return (HUMAN_COST / (1.0 - high), HUMAN_COST / low)


def policy_for(
    *, stakes: Stakes | None = None, lean: Lean | None = None, ask_human: bool = False
) -> Policy:
    """Turn a posture into the cost policy that implements it.

    Raises:
        PolicyError: `stakes` or `lean` is not a recognised value.

    Warns:
        UserWarning: `stakes` was given without `ask_human=True`, where it has no effect.
    """
    if lean is not None and lean not in LEAN_THRESHOLD:
        allowed = ", ".join(repr(name) for name in ("yes", "no"))
        raise PolicyError(f"lean must be {allowed} or None, got {lean!r}.")
    if stakes is not None and stakes not in STAKES_CERTAINTY:
        allowed = ", ".join(repr(name) for name in STAKES_CERTAINTY)
        raise PolicyError(f"stakes must be one of {allowed}, got {stakes!r}.")

    if stakes is not None and not ask_human:
        warnings.warn(
            f"stakes={stakes!r} only widens the range where a person is asked, and "
            f"ask_human is False, so it has no effect. Pass ask_human=True, or drop stakes and "
            f"use lean alone to choose which way to err.",
            UserWarning,
            stacklevel=4,
        )

    if not ask_human:
        # Without a human only the ratio matters, and the threshold is `lean` alone. Deriving the
        # costs from the threshold rather than from the band keeps `stakes` genuinely inert here,
        # so the warning above is literally true and not merely true of the outcome.
        threshold = LEAN_THRESHOLD[lean]
        return Policy(cost_false_yes=threshold, cost_false_no=1.0 - threshold)

    cost_false_yes, cost_false_no = _costs_for(band_for(stakes or DEFAULT_STAKES, lean))
    return Policy(cost_false_yes=cost_false_yes, cost_false_no=cost_false_no, cost_human=HUMAN_COST)


STAKES_CONFIDENCE: Final[dict[Stakes, float]] = {"low": 0.50, "medium": 0.65, "high": 0.80}
"""Confidence floor a `classify` or `rate` answer must clear before it is acted on.

`classify` and `rate` have no probability of yes to run the cost rule against -- they answer *which*
or *how much*, and the only thing to gate on is how peaked the distribution is. So the posture maps
onto `min_confidence` instead, and `lean` has no meaning here at all: there is no direction to err
in when there are four options. See D21.
"""


def min_confidence_for(stakes: Stakes | None, ask_human: bool) -> float | None:
    """The confidence floor a posture implies for `classify` and `rate`.

    Raises:
        PolicyError: `stakes` is not a recognised value.

    Warns:
        UserWarning: `stakes` was given without `ask_human=True`.
    """
    if stakes is not None and stakes not in STAKES_CONFIDENCE:
        allowed = ", ".join(repr(name) for name in STAKES_CONFIDENCE)
        raise PolicyError(f"stakes must be one of {allowed}, got {stakes!r}.")
    if stakes is not None and not ask_human:
        warnings.warn(
            f"stakes={stakes!r} sets the confidence below which the answer goes to a person, and "
            f"ask_human is False, so it has no effect. Pass ask_human=True.",
            UserWarning,
            stacklevel=4,
        )
    return STAKES_CONFIDENCE[stakes or DEFAULT_STAKES] if ask_human else None


@dataclass(frozen=True, slots=True)
class Preset:
    """One posture, and what it actually does."""

    stakes: Stakes
    lean: Lean | None
    ask_human: bool
    policy: Policy

    @property
    def unsure_band(self) -> tuple[float, float] | None:
        """Where this posture asks a person, or `None` when it never does."""
        return self.policy.implied_unsure_band

    @property
    def threshold(self) -> float:
        """Where yes overtakes no."""
        return LEAN_THRESHOLD[self.lean]

    def __str__(self) -> str:
        lean = "none" if self.lean is None else self.lean
        return (
            f"stakes={self.stakes:<6} lean={lean:<4} ask_human={self.ask_human!s:<5} "
            f"{self.policy.describe()}"
        )


def presets(*, ask_human: bool = True) -> tuple[Preset, ...]:
    """Every posture and the probabilities it works out to.

    Printing this is how you check that `stakes="high", lean="yes"` means what you assumed:

    ```python
    for preset in gut.presets():
        print(preset)
    ```
    """
    return tuple(
        Preset(
            stakes=stakes,
            lean=lean,
            ask_human=ask_human,
            policy=policy_for(stakes=stakes, lean=lean, ask_human=ask_human)
            if ask_human
            else policy_for(lean=lean),
        )
        for stakes in STAKES_CERTAINTY
        for lean in (None, "yes", "no")
    )
