"""Tests for the pure cost rule.

The rule is the part of `gut` that must be exactly right: everything else is plumbing around it,
and a subtly wrong threshold produces confidently wrong decisions rather than an error. So it gets
property-based tests of the three claims the README makes -- the rule minimises expected cost, it
reduces to the documented threshold formula when no human is available, and it is monotone in p --
alongside worked examples.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from gut._errors import PolicyError
from gut._outcomes import Outcome
from gut._rule import DEFAULT_POLICY, Policy, policy

# The property tests sweep arbitrary cost combinations, many of which make UNSURE unreachable.
# That is the point of those runs, not a problem with them; the warning has its own tests below.
pytestmark = pytest.mark.filterwarnings("ignore:cost_human=.*never be the cheapest")

probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
costs = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
positive_costs = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)


# --------------------------------------------------------------------------- properties


@given(p=probabilities, cfy=costs, cfn=costs, ch=st.one_of(st.none(), costs))
def test_decide_returns_the_cheapest_available_action(
    p: float, cfy: float, cfn: float, ch: float | None
) -> None:
    pol = policy(cost_false_yes=cfy, cost_false_no=cfn, cost_human=ch)
    chosen = pol.decide(p)
    expected = pol.expected_costs(p)

    assert chosen in expected
    assert expected[chosen] == min(expected.values())


@given(p=probabilities, cfy=positive_costs, cfn=positive_costs)
def test_without_a_human_the_rule_is_the_documented_threshold(
    p: float, cfy: float, cfn: float
) -> None:
    pol = policy(cost_false_yes=cfy, cost_false_no=cfn)
    threshold = cfy / (cfy + cfn)
    # Right at the boundary the two expected costs are equal up to float error and the tie rule,
    # not the formula, decides. Everywhere else the two must agree exactly.
    assume(not math.isclose(p, threshold, rel_tol=1e-9, abs_tol=1e-12))

    assert (pol.decide(p) is Outcome.YES) == (p > threshold)
    assert pol.implied_threshold == pytest.approx(threshold)


@given(p=probabilities, cfy=costs, cfn=costs)
def test_no_human_means_unsure_is_impossible(p: float, cfy: float, cfn: float) -> None:
    assert policy(cost_false_yes=cfy, cost_false_no=cfn).decide(p) is not Outcome.UNSURE


@given(
    ps=st.lists(probabilities, min_size=2, max_size=8),
    cfy=costs,
    cfn=costs,
    ch=st.one_of(st.none(), costs),
)
def test_outcome_is_monotone_in_the_probability(
    ps: list[float], cfy: float, cfn: float, ch: float | None
) -> None:
    """As p rises the outcome only ever moves NO -> UNSURE -> YES, never back."""
    pol = policy(cost_false_yes=cfy, cost_false_no=cfn, cost_human=ch)
    ranks = [pol.decide(p).rank for p in sorted(ps)]
    assert ranks == sorted(ranks)


@given(p=probabilities, threshold=probabilities)
def test_threshold_mode_is_strictly_above(p: float, threshold: float) -> None:
    assert (policy(threshold=threshold).decide(p) is Outcome.YES) == (p > threshold)


@given(p=probabilities, a=probabilities, b=probabilities)
def test_band_mode_is_a_closed_interval(p: float, a: float, b: float) -> None:
    lo, hi = min(a, b), max(a, b)
    outcome = policy(unsure_band=(lo, hi)).decide(p)
    if p < lo:
        assert outcome is Outcome.NO
    elif p > hi:
        assert outcome is Outcome.YES
    else:
        assert outcome is Outcome.UNSURE


@given(cfy=costs, cfn=costs, ch=st.one_of(st.none(), costs))
@settings(max_examples=50)
def test_costs_are_normalised_so_equal_policies_compare_equal(
    cfy: float, cfn: float, ch: float | None
) -> None:
    left = policy(cost_false_yes=cfy, cost_false_no=cfn, cost_human=ch)
    right = policy(cost_false_yes=cfy, cost_false_no=cfn, cost_human=ch)
    assert left == right
    assert hash(left) == hash(right)


# --------------------------------------------------------------------------- worked examples


def test_default_policy_is_p_above_one_half() -> None:
    assert DEFAULT_POLICY.implied_threshold == 0.5
    assert DEFAULT_POLICY.decide(0.500001) is Outcome.YES
    assert DEFAULT_POLICY.decide(0.5) is Outcome.NO  # strictly above, so the boundary is NO
    assert DEFAULT_POLICY.decide(0.499999) is Outcome.NO


def test_the_readme_example_threshold() -> None:
    """cost_false_yes=2, cost_false_no=50 is a threshold of 2/52, which nobody guesses."""
    pol = policy(cost_false_yes=2, cost_false_no=50)
    assert pol.implied_threshold == pytest.approx(0.038461, abs=1e-6)
    assert pol.decide(0.05) is Outcome.YES
    assert pol.decide(0.02) is Outcome.NO


@pytest.mark.filterwarnings("ignore:cost_human=.*never be the cheapest")
def test_expected_costs_match_the_formulas() -> None:
    pol = policy(cost_false_yes=2, cost_false_no=50, cost_human=5)
    assert pol.expected_costs(0.25) == {
        Outcome.YES: (1 - 0.25) * 2,
        Outcome.NO: 0.25 * 50,
        Outcome.UNSURE: 5,
    }


@pytest.mark.filterwarnings("ignore:cost_human=.*never be the cheapest")
def test_a_human_is_asked_only_where_it_is_genuinely_cheapest() -> None:
    pol = policy(cost_false_yes=2, cost_false_no=50, cost_human=5)
    # p=0.5: yes costs 1.0, no costs 25.0, human costs 5.0 -> yes is still cheapest.
    assert pol.decide(0.5) is Outcome.YES
    # p=0.02: yes costs 1.96, no costs 1.0 -> no is cheapest, human at 5.0 is not close.
    assert pol.decide(0.02) is Outcome.NO
    # Make the human cheap enough to win outright.
    assert policy(cost_false_yes=2, cost_false_no=50, cost_human=0.5).decide(0.5) is Outcome.UNSURE


def test_ties_prefer_unsure_then_no() -> None:
    # p=0.5 with symmetric costs of 2: yes and no both cost 1.0, and so does the human.
    three_way = policy(cost_false_yes=2, cost_false_no=2, cost_human=1)
    assert three_way.decide(0.5) is Outcome.UNSURE

    # Same tie without a human available: NO wins over YES.
    two_way = policy(cost_false_yes=2, cost_false_no=2)
    assert two_way.decide(0.5) is Outcome.NO

    # A human tying only with YES still wins, because UNSURE is preferred first.
    assert policy(cost_false_yes=2, cost_false_no=50, cost_human=1.0).decide(0.5) is Outcome.UNSURE


def test_zero_costs_never_say_yes() -> None:
    """Both mistakes free means nothing is worth acting on; the tie rule settles it as NO."""
    pol = policy(cost_false_yes=0, cost_false_no=0)
    assert pol.decide(0.0) is Outcome.NO
    assert pol.decide(1.0) is Outcome.NO
    assert pol.implied_threshold == 1.0


def test_unsure_is_unreachable_when_a_human_costs_more_than_the_peak() -> None:
    """Expected cost of a human is flat; the cheaper of yes and no peaks where they cross."""
    rule = policy(cost_false_yes=2, cost_false_no=50)
    assert rule.max_useful_cost_human == pytest.approx(2 * 50 / 52)

    # Above the peak, no probability sends it to a person.
    with pytest.warns(UserWarning, match="never be the cheapest option"):
        unreachable = policy(cost_false_yes=2, cost_false_no=50, cost_human=5)
    assert unreachable.unsure_reachable is False
    assert all(unreachable.decide(p / 100) is not Outcome.UNSURE for p in range(101))


def test_the_boundary_itself_is_reachable() -> None:
    """At the peak all three tie, and the tie rule prefers UNSURE."""
    rule = policy(cost_false_yes=2, cost_false_no=2, cost_human=1.0)
    assert rule.max_useful_cost_human == 1.0
    assert rule.unsure_reachable is True
    assert rule.decide(0.5) is Outcome.UNSURE


def test_a_workable_human_cost_is_not_warned_about(
    recwarn: pytest.WarningsRecorder,
) -> None:
    rule = policy(cost_false_yes=2, cost_false_no=20, cost_human=1)
    assert rule.unsure_reachable is True
    assert len(recwarn) == 0
    assert [rule.decide(p) for p in (0.01, 0.2, 0.9)] == [
        Outcome.NO,
        Outcome.UNSURE,
        Outcome.YES,
    ]


def test_no_human_option_means_no_warning_and_no_unsure(
    recwarn: pytest.WarningsRecorder,
) -> None:
    rule = policy(cost_false_yes=2, cost_false_no=50)
    assert rule.unsure_reachable is False
    assert len(recwarn) == 0


def test_reachability_is_undefined_for_the_escape_hatches() -> None:
    assert policy(threshold=0.5).max_useful_cost_human is None
    assert policy(threshold=0.5).unsure_reachable is False
    assert policy(unsure_band=(0.2, 0.8)).unsure_reachable is True


def test_free_mistakes_leave_no_room_for_a_human() -> None:
    rule = policy(cost_false_yes=0, cost_false_no=0, cost_human=0)
    assert rule.max_useful_cost_human == 0.0
    assert rule.unsure_reachable is True  # a free human ties with two free mistakes
    assert rule.decide(0.5) is Outcome.UNSURE


def test_implied_threshold_is_none_when_unsure_is_possible() -> None:
    assert policy(cost_false_yes=1, cost_false_no=1, cost_human=0.1).implied_threshold is None
    assert policy(unsure_band=(0.3, 0.7)).implied_threshold is None
    assert policy(threshold=0.9).implied_threshold == 0.9


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cost_false_yes": 1, "cost_false_no": 1, "threshold": 0.5}, "not both"),
        ({"cost_false_yes": 1, "cost_false_no": 1, "unsure_band": (0.2, 0.8)}, "not both"),
        ({"threshold": 0.5, "unsure_band": (0.2, 0.8)}, "not both"),
        ({"cost_false_yes": 1}, "must be given together"),
        ({"cost_false_no": 1}, "must be given together"),
        ({"cost_false_yes": -1, "cost_false_no": 1}, "non-negative"),
        ({"cost_false_yes": 1, "cost_false_no": math.inf}, "finite"),
        ({"cost_false_yes": 1, "cost_false_no": 1, "cost_human": -0.5}, "non-negative"),
        ({"threshold": 1.5}, "between 0 and 1"),
        ({"threshold": 0.5, "cost_human": 3}, "no effect alongside threshold"),
        ({"unsure_band": (0.8, 0.2)}, "exceeds upper bound"),
        ({"unsure_band": (0.2, 0.8), "cost_human": 3}, "no effect alongside unsure_band"),
        ({"unsure_band": (-0.1, 0.8)}, "between 0 and 1"),
    ],
)
def test_contradictory_policies_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        policy(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf])
def test_decide_rejects_non_probabilities(bad: float) -> None:
    with pytest.raises(PolicyError, match="between 0 and 1"):
        DEFAULT_POLICY.decide(bad)


def test_booleans_are_not_accepted_as_numbers() -> None:
    """`True` is an int in Python; silently reading it as a cost of 1 would hide a real mistake."""
    with pytest.raises(PolicyError, match="must be a number"):
        policy(cost_false_yes=True, cost_false_no=1)
    with pytest.raises(PolicyError, match="must be a number"):
        DEFAULT_POLICY.decide(True)


def test_policy_constructed_directly_still_validates() -> None:
    with pytest.raises(PolicyError, match="exactly one of"):
        Policy()
    with pytest.raises(PolicyError, match="exactly one of"):
        Policy(threshold=0.5, unsure_band=(0.2, 0.8))


def test_unsure_band_must_be_a_pair() -> None:
    with pytest.raises(PolicyError, match=r"must be a \(lo, hi\) pair"):
        policy(unsure_band=(0.1, 0.5, 0.9))  # type: ignore[arg-type]


def test_expected_costs_is_rejected_for_escape_hatch_policies() -> None:
    with pytest.raises(PolicyError, match="only meaningful for a cost-based policy"):
        policy(threshold=0.5).expected_costs(0.5)
    with pytest.raises(PolicyError, match="only meaningful for a cost-based policy"):
        policy(unsure_band=(0.2, 0.8)).expected_costs(0.5)
