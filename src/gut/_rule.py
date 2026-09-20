"""The cost rule: turning a probability into an action.

This module is pure. It imports nothing outside the standard library and `gut`'s own outcome and
error types, it performs no I/O, and it holds no state. Everything in `gut` that decides *what to
do* with a probability goes through `Policy.decide`.

The idea it exists to serve: a threshold is an arbitrary number, but the cost of each kind of
mistake is something a developer actually knows. Given a calibrated ``p = P(yes)``:

    expected cost of saying YES     = (1 - p) * cost_false_yes
    expected cost of saying NO      =      p  * cost_false_no
    expected cost of asking a human =         cost_human

`gut` takes the cheapest. With no human available that reduces to the familiar

    YES  <=>  p > cost_false_yes / (cost_false_yes + cost_false_no)

which for ``cost_false_yes=2, cost_false_no=50`` is ``0.038`` -- a number nobody guesses correctly.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Final

from gut._errors import PolicyError
from gut._outcomes import Outcome

DEFAULT_COST: Final = 1.0
"""Cost assumed for a mistake in either direction when the caller specifies no costs at all."""


def _set(policy: Policy, field: str, value: object) -> None:
    """Write a validated, float-normalised value onto a frozen `Policy` during `__post_init__`.

    Normalising means `Policy(cost_false_yes=2)` and `Policy(cost_false_yes=2.0)` compare equal and
    hash alike, which the decision cache later relies on for stable keys.
    """
    object.__setattr__(policy, field, value)


def _check_cost(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PolicyError(f"{name} must be a number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise PolicyError(f"{name} must be finite, got {value!r}.")
    if value < 0:
        raise PolicyError(f"{name} must be non-negative, got {value!r}.")
    return value


def _check_probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PolicyError(f"{name} must be a number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise PolicyError(f"{name} must be between 0 and 1 inclusive, got {value!r}.")
    return value


@dataclass(frozen=True, slots=True)
class Policy:
    """A resolved rule for turning a probability into an `Outcome`.

    Exactly one mode is active, which `policy()` enforces:

    - **cost mode** -- `cost_false_yes` and `cost_false_no` are set, `cost_human` optionally.
    - **threshold mode** -- `threshold` is set.
    - **band mode** -- `unsure_band` is set.

    Construct instances through `policy()`; it validates which arguments were actually supplied,
    which this class cannot see once defaults have been filled in.
    """

    cost_false_yes: float | None = None
    """What it costs to act on a YES that turns out to be wrong."""
    cost_false_no: float | None = None
    """What it costs to act on a NO that turns out to be wrong."""
    cost_human: float | None = None
    """What it costs to put the decision in front of a person. `None` means no human is available,
    which makes `UNSURE` impossible."""
    threshold: float | None = None
    """Escape hatch: decide YES strictly above this probability, NO at or below it."""
    unsure_band: tuple[float, float] | None = None
    """Escape hatch: `(lo, hi)`; UNSURE inside the closed interval, NO below it, YES above it."""

    def __post_init__(self) -> None:
        """Validate ranges and that exactly one mode is active."""
        modes = [
            self.cost_false_yes is not None or self.cost_false_no is not None,
            self.threshold is not None,
            self.unsure_band is not None,
        ]
        if sum(modes) != 1:
            raise PolicyError(
                "A policy must use exactly one of costs, threshold, or unsure_band; "
                f"got {sum(modes)}."
            )

        if self.threshold is not None:
            _set(self, "threshold", _check_probability("threshold", self.threshold))
            if self.cost_human is not None:
                raise PolicyError(
                    "cost_human has no effect alongside threshold, which can never return UNSURE. "
                    "Use unsure_band=(lo, hi) to carve out an UNSURE region, or switch to costs."
                )

        if self.unsure_band is not None:
            band = self.unsure_band
            if len(band) != 2:
                raise PolicyError(f"unsure_band must be a (lo, hi) pair, got {band!r}.")
            lo = _check_probability("unsure_band lower bound", band[0])
            hi = _check_probability("unsure_band upper bound", band[1])
            if lo > hi:
                raise PolicyError(f"unsure_band lower bound {lo!r} exceeds upper bound {hi!r}.")
            _set(self, "unsure_band", (lo, hi))
            if self.cost_human is not None:
                raise PolicyError(
                    "cost_human has no effect alongside unsure_band, which already fixes where "
                    "UNSURE applies."
                )

        if modes[0]:
            if self.cost_false_yes is None or self.cost_false_no is None:
                raise PolicyError(
                    "cost_false_yes and cost_false_no must be given together: it is their ratio "
                    "that sets the decision threshold, so half a cost model is not a cost model."
                )
            _set(self, "cost_false_yes", _check_cost("cost_false_yes", self.cost_false_yes))
            _set(self, "cost_false_no", _check_cost("cost_false_no", self.cost_false_no))
            if self.cost_human is not None:
                _set(self, "cost_human", _check_cost("cost_human", self.cost_human))

    def expected_costs(self, p: float) -> dict[Outcome, float]:
        """Expected cost of each available action at probability `p`.

        Only defined in cost mode; the escape hatches express a boundary directly rather than a
        cost model, so there is nothing to compute.

        Raises:
            PolicyError: This policy is not in cost mode, or `p` is not a probability.
        """
        if self.cost_false_yes is None or self.cost_false_no is None:
            raise PolicyError(
                "expected_costs() is only meaningful for a cost-based policy; this one uses "
                f"{'threshold' if self.threshold is not None else 'unsure_band'}."
            )
        p = _check_probability("p", p)
        costs = {
            Outcome.YES: (1.0 - p) * self.cost_false_yes,
            Outcome.NO: p * self.cost_false_no,
        }
        if self.cost_human is not None:
            costs[Outcome.UNSURE] = self.cost_human
        return costs

    def decide(self, p: float) -> Outcome:
        """Return the cheapest action at probability `p`.

        Ties prefer `UNSURE`, then `NO` -- the conservative order, in that order of preference a
        tie never resolves to taking the irreversible action.

        Raises:
            PolicyError: `p` is not a probability between 0 and 1.
        """
        p = _check_probability("p", p)

        if self.unsure_band is not None:
            lo, hi = self.unsure_band
            if p < lo:
                return Outcome.NO
            if p > hi:
                return Outcome.YES
            return Outcome.UNSURE

        if self.threshold is not None:
            return Outcome.YES if p > self.threshold else Outcome.NO

        costs = self.expected_costs(p)
        cheapest = min(costs.values())
        # Iteration order encodes the tie preference: UNSURE, then NO, then YES.
        for outcome in (Outcome.UNSURE, Outcome.NO, Outcome.YES):
            if outcome in costs and costs[outcome] == cheapest:
                return outcome
        raise AssertionError("unreachable: costs always contains YES and NO")  # pragma: no cover

    @property
    def max_useful_cost_human(self) -> float | None:
        """The largest `cost_human` at which asking a person is ever the cheapest option.

        Expected cost of a human is flat, while the cheaper of YES and NO peaks where the two
        cross, at `cost_false_yes * cost_false_no / (cost_false_yes + cost_false_no)`. Above that
        peak there is no probability at which asking a person is worth it, and UNSURE becomes
        unreachable. Exactly *at* the peak all three options tie, and the tie rule prefers UNSURE,
        so the boundary is still reachable.

        `None` for a policy that is not cost-based.
        """
        if self.cost_false_yes is None or self.cost_false_no is None:
            return None
        total = self.cost_false_yes + self.cost_false_no
        if total == 0:
            return 0.0
        return self.cost_false_yes * self.cost_false_no / total

    @property
    def unsure_reachable(self) -> bool:
        """Whether any probability at all would send this decision to a human."""
        if self.unsure_band is not None:
            return True
        if self.cost_human is None:
            return False
        ceiling = self.max_useful_cost_human
        # `<=`: at the peak all three tie, and ties prefer UNSURE.
        return ceiling is not None and self.cost_human <= ceiling

    @property
    def implied_threshold(self) -> float | None:
        """The probability above which this policy says YES, when that is a single number.

        `None` for a policy with an `UNSURE` region, where one number cannot describe the rule.
        Mostly useful for showing a developer what their costs actually amount to.
        """
        if self.threshold is not None:
            return self.threshold
        if self.unsure_band is not None or self.cost_human is not None:
            return None
        assert self.cost_false_yes is not None and self.cost_false_no is not None
        total = self.cost_false_yes + self.cost_false_no
        if total == 0:
            # Both mistakes are free, so nothing is ever worth saying yes to; the tie rule picks NO.
            return 1.0
        return self.cost_false_yes / total


def policy(
    *,
    cost_false_yes: float | None = None,
    cost_false_no: float | None = None,
    cost_human: float | None = None,
    threshold: float | None = None,
    unsure_band: tuple[float, float] | None = None,
) -> Policy:
    """Build a `Policy` from whichever knobs the caller supplied.

    Costs are the documented default. `threshold` and `unsure_band` are escape hatches for people
    who already know the number they want, and cannot be combined with costs or with each other:
    outside an `unsure_band` the answer is fixed by which side of the band `p` falls on, so a
    threshold there would never be consulted.

    Given nothing at all, the result is symmetric costs with no human option -- plain `p > 0.5`.

    Raises:
        PolicyError: The arguments are contradictory, incomplete, or out of range.
    """
    has_costs = cost_false_yes is not None or cost_false_no is not None
    if has_costs and threshold is not None:
        raise PolicyError(
            "Pass costs or a threshold, not both: the costs already determine the threshold, "
            "as cost_false_yes / (cost_false_yes + cost_false_no)."
        )
    if has_costs and unsure_band is not None:
        raise PolicyError(
            "Pass costs or an unsure_band, not both: the costs, including cost_human, already "
            "determine where UNSURE applies."
        )
    if threshold is not None and unsure_band is not None:
        raise PolicyError(
            "Pass a threshold or an unsure_band, not both: outside the band the outcome is fixed "
            "by which side p falls on, so the threshold would never be consulted."
        )

    if threshold is not None:
        return Policy(threshold=threshold, cost_human=cost_human)
    if unsure_band is not None:
        return Policy(unsure_band=unsure_band, cost_human=cost_human)
    if not has_costs:
        # Nothing specified at all: symmetric costs, which is plain p > 0.5.
        cost_false_yes = cost_false_no = DEFAULT_COST
    # A half-specified cost model is left as-is; Policy rejects it with a targeted message.
    rule = Policy(
        cost_false_yes=cost_false_yes,
        cost_false_no=cost_false_no,
        cost_human=cost_human,
    )
    if cost_human is not None and not rule.unsure_reachable:
        ceiling = rule.max_useful_cost_human
        warnings.warn(
            f"cost_human={rule.cost_human!r} can never be the cheapest option, so this decision "
            f"will never return UNSURE. Asking a person only pays off below "
            f"{ceiling:.4g}, where the cheaper of yes and no peaks. Lower cost_human, or drop it "
            f"and accept a two-way decision.",
            UserWarning,
            stacklevel=3,
        )
    return rule


DEFAULT_POLICY: Final = policy()
"""Symmetric costs, no human option: YES above 0.5, NO at or below it."""
