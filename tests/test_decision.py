"""Tests for decision objects: truthiness, the UNSURE policy, equality and `match`."""

from __future__ import annotations

import enum
import threading

import pytest

import gut
from gut import ChoiceDecision, Decision, Outcome, ScoreDecision, UnsureDecision
from gut._rule import DEFAULT_POLICY, policy


class Team(enum.Enum):
    BILLING = "invoices, charges, refunds"
    PLATFORM = "outages, latency, API errors"
    OTHER = "anything else"


def make(outcome: Outcome, *, p: float = 0.5) -> Decision:
    return Decision(outcome=outcome, id="site-1", model="jev-1.13.0", p=p, policy=DEFAULT_POLICY)


def make_choice(outcome: Outcome) -> ChoiceDecision[Team]:
    value: Team | Outcome = Team.BILLING if outcome is Outcome.YES else Outcome.UNSURE
    return ChoiceDecision(
        outcome=outcome,
        id="site-2",
        model="jev-1.13.0",
        value=value,  # type: ignore[arg-type]
        probabilities={Team.BILLING: 0.8, Team.PLATFORM: 0.15, Team.OTHER: 0.05},
        confidence=0.8,
    )


# --------------------------------------------------------------------------- truthiness


def test_yes_is_true_and_no_is_false() -> None:
    assert bool(make(Outcome.YES)) is True
    assert bool(make(Outcome.NO)) is False


def test_unsure_raises_by_default() -> None:
    decision = make(Outcome.UNSURE)
    with pytest.raises(UnsureDecision) as caught:
        bool(decision)
    assert caught.value.decision is decision
    assert "site-1" in str(caught.value)


@pytest.mark.parametrize(("setting", "expected"), [("true", True), ("false", False)])
def test_configure_coerces_unsure(setting: str, expected: bool) -> None:
    gut.configure(on_unsure=setting)  # type: ignore[arg-type]
    assert bool(make(Outcome.UNSURE)) is expected
    # The coercion must not touch the decisions that do have an honest answer.
    assert bool(make(Outcome.YES)) is True
    assert bool(make(Outcome.NO)) is False


def test_a_callback_is_handed_the_decision() -> None:
    seen: list[object] = []

    def callback(decision: gut.BaseDecision) -> bool:
        seen.append(decision)
        return True

    gut.configure(on_unsure=callback)
    decision = make(Outcome.UNSURE)
    assert bool(decision) is True
    assert seen == [decision]


def test_scoped_override_applies_only_inside_the_block() -> None:
    with gut.on_unsure("false"):
        assert bool(make(Outcome.UNSURE)) is False
    with pytest.raises(UnsureDecision):
        bool(make(Outcome.UNSURE))


def test_scoped_overrides_nest_and_restore() -> None:
    gut.configure(on_unsure="true")
    with gut.on_unsure("false"):
        assert bool(make(Outcome.UNSURE)) is False
        with gut.on_unsure("raise"), pytest.raises(UnsureDecision):
            bool(make(Outcome.UNSURE))
        assert bool(make(Outcome.UNSURE)) is False
    assert bool(make(Outcome.UNSURE)) is True


def test_scoped_override_restores_after_an_exception() -> None:
    with pytest.raises(RuntimeError), gut.on_unsure("false"):
        raise RuntimeError("boom")
    with pytest.raises(UnsureDecision):
        bool(make(Outcome.UNSURE))


def test_scoped_override_does_not_leak_across_threads() -> None:
    seen: list[str] = []

    def worker() -> None:
        try:
            bool(make(Outcome.UNSURE))
        except UnsureDecision:
            seen.append("raised")
        else:  # pragma: no cover - only reached if the override leaked
            seen.append("coerced")

    with gut.on_unsure("false"):
        assert bool(make(Outcome.UNSURE)) is False
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert seen == ["raised"]


def test_configure_rejects_unknown_policies() -> None:
    with pytest.raises(gut.ConfigurationError, match="on_unsure must be one of"):
        gut.configure(on_unsure="sometimes")  # type: ignore[arg-type]
    with (
        pytest.raises(gut.ConfigurationError, match="on_unsure must be one of"),
        gut.on_unsure("perhaps"),  # type: ignore[arg-type]
    ):
        pass  # pragma: no cover - the context manager raises on entry


def test_configure_with_nothing_changes_nothing() -> None:
    gut.configure()
    with pytest.raises(UnsureDecision):
        bool(make(Outcome.UNSURE))


# --------------------------------------------------------------------------- match and equality


@pytest.mark.parametrize("outcome", list(Outcome))
def test_match_on_the_decision_itself(outcome: Outcome) -> None:
    def route(decision: Decision) -> str:
        match decision:
            case gut.YES:
                return "yes"
            case gut.NO:
                return "no"
            case gut.UNSURE:
                return "unsure"
            case _:
                return "unmatched"

    assert route(make(outcome)) == outcome.value


def test_decision_equals_its_outcome_in_both_directions() -> None:
    decision = make(Outcome.YES)
    assert decision == Outcome.YES
    # Deliberately reversed: this asserts Python falls back to Decision.__eq__ when the enum's
    # own comparison returns NotImplemented. Rewriting it the "correct" way tests nothing.
    assert Outcome.YES == decision  # noqa: SIM300
    assert decision != Outcome.NO


def test_decisions_compare_by_value_within_a_kind() -> None:
    assert make(Outcome.YES, p=0.9) == make(Outcome.YES, p=0.9)
    assert make(Outcome.YES, p=0.9) != make(Outcome.YES, p=0.8)


def test_decisions_of_different_kinds_are_never_equal() -> None:
    assert make(Outcome.YES) != make_choice(Outcome.YES)
    assert make(Outcome.YES) != "yes"
    assert make(Outcome.YES) != 1


def test_hash_agrees_with_equality_against_outcomes() -> None:
    decision = make(Outcome.YES)
    assert hash(decision) == hash(Outcome.YES)
    assert decision in {Outcome.YES}


# --------------------------------------------------------------------------- the other two kinds


def test_choice_decision_truthiness() -> None:
    assert bool(make_choice(Outcome.YES)) is True
    with pytest.raises(UnsureDecision):
        bool(make_choice(Outcome.UNSURE))


def test_choice_decision_carries_its_distribution() -> None:
    decision = make_choice(Outcome.YES)
    assert decision.value is Team.BILLING
    assert decision.probabilities[Team.BILLING] == 0.8
    assert sum(decision.probabilities.values()) == pytest.approx(1.0)


def test_unsure_choice_reports_unsure_as_its_value() -> None:
    assert make_choice(Outcome.UNSURE).value is Outcome.UNSURE


@pytest.mark.parametrize(
    ("score", "expected"), [(0.0, 0), (1.3, 1), (1.5, 2), (2.49, 2), (2.5, 3), (3.0, 3)]
)
def test_score_decision_nearest_level_rounds_half_up(score: float, expected: int) -> None:
    decision = ScoreDecision(
        outcome=Outcome.YES,
        id="site-3",
        model="jev-1.13.0",
        score=score,
        probabilities={0: 0.1, 1: 0.2, 2: 0.4, 3: 0.3},
        confidence=0.6,
        levels=("calm", "annoyed", "angry", "threatening to leave"),
    )
    assert decision.nearest_level == expected


def test_decision_records_how_it_was_produced() -> None:
    decision = Decision(
        outcome=Outcome.YES,
        id="site-4",
        model="jev-1.13.0",
        p=0.83,
        policy=policy(cost_false_yes=2, cost_false_no=50, cost_human=5),
        cached=True,
        latency_ms=None,
    )
    assert decision.model == "jev-1.13.0"
    assert decision.cached is True
    assert decision.latency_ms is None
    assert decision.policy.cost_false_no == 50
