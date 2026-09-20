"""Measuring what `gut` adds, on top of what the model already knew.

The distinction this module exists to keep straight: **accuracy, Brier and ECE are the model's
numbers.** They belong in a benchmark as the baseline, and reporting them as `gut`'s result would
be taking credit for someone else's work.

What `gut` contributes is what happens to those probabilities afterwards. Two things are measurable:

**Risk-coverage.** A hard threshold has exactly one operating point: it decides everything, and its
error rate is whatever it is. A posture with `ask_human=True` gives up some coverage -- the share of
inputs decided without a person -- in exchange for a lower error rate on what it does decide. The
question is the exchange rate, and it is the whole argument for a third branch.

**Whether the words travel.** `stakes="medium"` is supposed to mean the same thing on a support
ticket, a GitHub issue and an SMS. Whether it does depends entirely on whether the model's
probabilities are comparable across those tasks, which is not something anyone should assume. The
spread of realized error rates across datasets at a fixed posture is the test, and calibration is
what should narrow it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from benchmarks._core import Asked

from gut._backends.base import ChoiceAnswer, NoulAnswer
from gut._calibration import Calibration, calibrate
from gut._calibrators import Calibrator, Identity
from gut._outcomes import Outcome
from gut._posture import Lean, Stakes, min_confidence_for, policy_for

POSTURES: Final[list[tuple[str, Stakes | None, Lean | None]]] = [
    ("no arguments", None, None),
    *[
        (f"stakes={stakes:<6} lean={lean or 'none':<4}", stakes, lean)
        for stakes in ("low", "medium", "high")
        for lean in (None, "yes", "no")
    ],
]

HARD_THRESHOLD: Final = 0.7
"""What `before.py` hard-codes, and what a developer writes when nothing suggests otherwise."""


@dataclass(frozen=True, slots=True)
class Row:
    """One operating point: what it decided, and how well."""

    label: str
    total: int
    automatic: int
    wrong: int
    """Wrong among the decisions made without a person."""
    false_yes: int = 0
    false_no: int = 0

    @property
    def human(self) -> int:
        """Sent to a person."""
        return self.total - self.automatic

    @property
    def coverage(self) -> float:
        """Share decided without a person."""
        return self.automatic / self.total if self.total else 0.0

    @property
    def error_rate(self) -> float:
        """Share of the automatic decisions that were wrong. The number that matters."""
        return self.wrong / self.automatic if self.automatic else 0.0

    def line(self) -> str:
        """The row, for a table."""
        return (
            f"  {self.label:<26} {self.coverage:>7.0%} {self.error_rate:>9.1%}"
            f" {self.wrong:>7} {self.false_yes:>6} {self.false_no:>6}"
        )


HEADER: Final = f"  {'posture':<26} {'covered':>7} {'err|auto':>9} {'wrong':>7} {'FP':>6} {'FN':>6}"


def _scores(asked: Sequence[Asked], calibrator: Calibrator) -> list[float]:
    """The number each posture gates on, corrected if a calibrator was fitted."""
    values: list[float] = []
    for item in asked:
        raw = item.answer.p if isinstance(item.answer, NoulAnswer) else _confidence(item)
        values.append(calibrator.apply(raw))
    return values


def _confidence(item: Asked) -> float:
    answer = item.answer
    assert isinstance(answer, ChoiceAnswer)
    return answer.confidence


# --------------------------------------------------------------------------- binary tasks


def sweep_binary(
    asked: Sequence[Asked],
    positive: str,
    *,
    calibrator: Calibrator | None = None,
) -> list[Row]:
    """Every posture over a yes/no task, scored on identical model answers."""
    scores = _scores(asked, calibrator or Identity())
    truth = [item.example.label == positive for item in asked]

    rows: list[Row] = []
    for label, stakes, lean in POSTURES:
        rule = (
            policy_for(lean=None)
            if stakes is None and lean is None
            else policy_for(stakes=stakes, lean=lean, ask_human=True)
        )
        automatic = wrong = false_yes = false_no = 0
        for score, actual in zip(scores, truth, strict=True):
            outcome = rule.decide(score)
            if outcome is Outcome.UNSURE:
                continue
            automatic += 1
            said_yes = outcome is Outcome.YES
            if said_yes and not actual:
                wrong += 1
                false_yes += 1
            elif not said_yes and actual:
                wrong += 1
                false_no += 1
        rows.append(
            Row(
                label=label,
                total=len(asked),
                automatic=automatic,
                wrong=wrong,
                false_yes=false_yes,
                false_no=false_no,
            )
        )
    return rows


def threshold_binary(
    asked: Sequence[Asked], positive: str, *, threshold: float = HARD_THRESHOLD
) -> Row:
    """The baseline: one number, no third branch, everything decided."""
    scores = _scores(asked, Identity())
    false_yes = false_no = 0
    for score, item in zip(scores, asked, strict=True):
        actual = item.example.label == positive
        said_yes = score > threshold
        if said_yes and not actual:
            false_yes += 1
        elif not said_yes and actual:
            false_no += 1
    return Row(
        label=f"threshold={threshold:g}",
        total=len(asked),
        automatic=len(asked),
        wrong=false_yes + false_no,
        false_yes=false_yes,
        false_no=false_no,
    )


# --------------------------------------------------------------------------- choice tasks


def sweep_choice(
    asked: Sequence[Asked],
    *,
    calibrator: Calibrator | None = None,
    abstain_label: str | None = None,
) -> list[Row]:
    """Every posture over a classification, where the posture is a confidence floor.

    `abstain_label` names the catch-all member, if the enum has one: choosing it counts as
    declining rather than as an answer, which is the behaviour the out-of-scope test is about.
    """
    scores = _scores(asked, calibrator or Identity())

    rows: list[Row] = []
    for label, stakes, lean in POSTURES:
        if lean is not None:
            continue  # `lean` does not apply to a multiway choice (D21)
        floor = None if stakes is None else min_confidence_for(stakes, ask_human=True)
        automatic = wrong = 0
        for score, item in zip(scores, asked, strict=True):
            chosen = _chosen(item)
            if floor is not None and score < floor:
                continue
            if abstain_label is not None and chosen == abstain_label:
                continue
            automatic += 1
            if chosen != item.example.label:
                wrong += 1
        rows.append(Row(label=label, total=len(asked), automatic=automatic, wrong=wrong))
    return rows


def _chosen(item: Asked) -> str:
    answer = item.answer
    assert isinstance(answer, ChoiceAnswer)
    return answer.choice


def forced_choice(asked: Sequence[Asked]) -> Row:
    """The baseline: take the top choice every time, as a classifier without an abstain option."""
    wrong = sum(1 for item in asked if _chosen(item) != item.example.label)
    return Row(label="always answer", total=len(asked), automatic=len(asked), wrong=wrong)


# --------------------------------------------------------------------------- out of scope


@dataclass(frozen=True, slots=True)
class OutOfScope:
    """How an input that fits no category was handled."""

    out_of_scope: int
    caught_by_other: int
    caught_by_unsure: int
    confidently_wrong: int
    in_scope: int
    in_scope_lost: int
    """In-scope inputs wrongly declined -- the price of catching the rest."""

    @property
    def caught(self) -> int:
        """Out-of-scope inputs that did not become a confident answer."""
        return self.caught_by_other + self.caught_by_unsure

    @property
    def recall(self) -> float:
        """Share of out-of-scope inputs the system declined."""
        return self.caught / self.out_of_scope if self.out_of_scope else 0.0

    @property
    def in_scope_cost(self) -> float:
        """Share of in-scope inputs it declined by mistake."""
        return self.in_scope_lost / self.in_scope if self.in_scope else 0.0


def out_of_scope(
    asked: Sequence[Asked],
    *,
    oos_label: str,
    abstain_label: str,
    stakes: Stakes | None,
    calibrator: Calibrator | None = None,
) -> OutOfScope:
    """Split the out-of-scope inputs by how the system declined them, or failed to."""
    scores = _scores(asked, calibrator or Identity())
    floor = None if stakes is None else min_confidence_for(stakes, ask_human=True)

    counts = dict.fromkeys(("oos", "other", "unsure", "confident", "in_scope", "in_scope_lost"), 0)
    for score, item in zip(scores, asked, strict=True):
        chosen = _chosen(item)
        declined = (floor is not None and score < floor) or chosen == abstain_label
        if item.example.label == oos_label:
            counts["oos"] += 1
            if floor is not None and score < floor:
                counts["unsure"] += 1
            elif chosen == abstain_label:
                counts["other"] += 1
            else:
                counts["confident"] += 1
        else:
            counts["in_scope"] += 1
            if declined:
                counts["in_scope_lost"] += 1

    return OutOfScope(
        out_of_scope=counts["oos"],
        caught_by_other=counts["other"],
        caught_by_unsure=counts["unsure"],
        confidently_wrong=counts["confident"],
        in_scope=counts["in_scope"],
        in_scope_lost=counts["in_scope_lost"],
    )


# --------------------------------------------------------------------------- calibration


def pairs_binary(asked: Sequence[Asked], positive: str) -> list[tuple[float, bool]]:
    """`(probability, what happened)` for a yes/no task."""
    return [
        (
            item.answer.p if isinstance(item.answer, NoulAnswer) else 0.0,
            item.example.label == positive,
        )
        for item in asked
    ]


def pairs_choice(asked: Sequence[Asked]) -> list[tuple[float, bool]]:
    """`(confidence, was the pick right)` for a classification."""
    return [(_confidence(item), _chosen(item) == item.example.label) for item in asked]


def measure(
    pairs: Sequence[tuple[float, bool]], calibrator: Calibrator | None = None
) -> Calibration:
    """Brier and ECE, optionally after a correction."""
    curve = calibrator or Identity()
    return calibrate([(curve.apply(score), event) for score, event in pairs])


def keyword_rule(
    asked: Sequence[Asked], predicate: Callable[[str], bool], positive: str, *, label: str
) -> Row:
    """A hand-written rule over the same examples: the option the pitch says is brittle."""
    false_yes = false_no = 0
    for item in asked:
        actual = item.example.label == positive
        said_yes = predicate(item.example.text)
        if said_yes and not actual:
            false_yes += 1
        elif not said_yes and actual:
            false_no += 1
    return Row(
        label=label,
        total=len(asked),
        automatic=len(asked),
        wrong=false_yes + false_no,
        false_yes=false_yes,
        false_no=false_no,
    )
