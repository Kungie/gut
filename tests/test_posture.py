"""Tests for the posture layer: what the words actually mean in probabilities."""

from __future__ import annotations

import enum

import pytest

import gut
from gut import FakeBackend, Outcome, presets
from gut._errors import PolicyError
from gut._posture import (
    LEAN_THRESHOLD,
    STAKES_CONFIDENCE,
    Lean,
    Stakes,
    band_for,
    min_confidence_for,
    policy_for,
)

# The table the README publishes. If this changes, every posture in every user's code changes
# meaning, so it is pinned here rather than derived.
BANDS: dict[tuple[str, str | None], tuple[float, float]] = {
    ("low", None): (0.400, 0.600),
    ("medium", None): (0.250, 0.750),
    ("high", None): (0.100, 0.900),
    ("low", "yes"): (0.200, 0.400),
    ("medium", "yes"): (0.125, 0.625),
    ("high", "yes"): (0.050, 0.850),
    ("low", "no"): (0.600, 0.800),
    ("medium", "no"): (0.375, 0.875),
    ("high", "no"): (0.150, 0.950),
}


class Team(enum.Enum):
    A = "the first thing"
    B = "the second thing"
    OTHER = "anything else"


@pytest.fixture
def backend() -> FakeBackend:
    fake = FakeBackend(
        default=0.83,
        rule=lambda state, name, spec: "A" if spec.canonical()["type"] == "choice" else None,
    )
    gut.configure(backend=fake, cache=gut.NullCache())
    return fake


# --------------------------------------------------------------------------- the table


@pytest.mark.parametrize(
    ("key", "expected"), BANDS.items(), ids=[f"{stakes}-{lean}" for stakes, lean in BANDS]
)
def test_each_preset_produces_its_documented_band(
    key: tuple[Stakes, Lean | None], expected: tuple[float, float]
) -> None:
    stakes, lean = key
    assert band_for(stakes, lean) == pytest.approx(expected)

    rule = policy_for(stakes=stakes, lean=lean, ask_human=True)
    band = rule.implied_unsure_band
    assert band is not None
    assert band == pytest.approx(expected)


@pytest.mark.parametrize(
    ("key", "expected"), BANDS.items(), ids=[f"{stakes}-{lean}" for stakes, lean in BANDS]
)
def test_every_preset_band_is_reachable(
    key: tuple[Stakes, Lean | None], expected: tuple[float, float]
) -> None:
    """Structural, not lucky: a band is reachable exactly when lo < hi, and presets are bands."""
    stakes, lean = key
    rule = policy_for(stakes=stakes, lean=lean, ask_human=True)
    assert rule.unsure_reachable is True

    low, high = expected
    assert rule.decide((low + high) / 2) is Outcome.UNSURE


@pytest.mark.parametrize("stakes", ["low", "medium", "high"])
@pytest.mark.parametrize("lean", [None, "yes", "no"])
def test_lean_sets_the_threshold_and_stakes_never_moves_it(
    stakes: Stakes, lean: Lean | None
) -> None:
    """The two knobs are independent: being more careful must not change which way you err."""
    rule = policy_for(stakes=stakes, lean=lean, ask_human=True)
    low, high = BANDS[stakes, lean]
    assert rule.implied_unsure_band == pytest.approx((low, high))
    # Where yes overtakes no, ignoring the human option.
    assert low / (low + 1 - high) == pytest.approx(LEAN_THRESHOLD[lean])

    # And without a human it is exactly that threshold.
    assert policy_for(lean=lean).implied_threshold == pytest.approx(LEAN_THRESHOLD[lean])


@pytest.mark.parametrize("lean", [None, "yes", "no"])
def test_higher_stakes_only_widens(lean: Lean | None) -> None:
    widths = [
        BANDS[stakes, lean][1] - BANDS[stakes, lean][0] for stakes in ("low", "medium", "high")
    ]
    assert widths == sorted(widths)


def test_presets_lists_every_combination() -> None:
    table = presets()
    assert len(table) == 9
    assert {(p.stakes, p.lean) for p in table} == set(BANDS)
    for preset in table:
        assert preset.unsure_band == pytest.approx(BANDS[preset.stakes, preset.lean])
        assert preset.threshold == LEAN_THRESHOLD[preset.lean]
        assert "ask a person" in str(preset)


def test_presets_without_a_human_describe_a_threshold() -> None:
    for preset in presets(ask_human=False):
        assert preset.unsure_band is None
        assert "ask a person" not in str(preset)
        assert f"{preset.threshold:.3g}" in str(preset)


# --------------------------------------------------------------------------- at the call site


def test_level_zero_is_unchanged(backend: FakeBackend) -> None:
    decision = gut.likely("a ticket", "is a bug report")
    assert decision.policy.implied_threshold == 0.5
    assert decision.outcome is Outcome.YES
    assert decision.policy.implied_unsure_band is None


def test_words_alone_change_the_boundary(backend: FakeBackend) -> None:
    assert gut.likely("t", "q", lean="yes").policy.implied_threshold == 0.25
    assert gut.likely("t", "q", lean="no").policy.implied_threshold == 0.75


def test_ask_human_opens_the_third_branch(backend: FakeBackend) -> None:
    without = gut.likely("t", "q")
    with_human = gut.likely("t", "q", ask_human=True)

    assert without.policy.unsure_reachable is False
    assert with_human.policy.unsure_reachable is True
    # p=0.83 sits above the medium band, so it still decides.
    assert with_human.outcome is Outcome.YES


def test_high_stakes_sends_a_middling_answer_to_a_person(backend: FakeBackend) -> None:
    decision = gut.likely("t", "q", stakes="high", lean="yes", ask_human=True)
    assert decision.outcome is Outcome.UNSURE  # 0.83 is inside 0.05-0.85


def test_stakes_without_a_human_warns_and_does_nothing(backend: FakeBackend) -> None:
    """Inert, not merely inconsequential: the policy itself must come out identical."""
    with pytest.warns(UserWarning, match="has no effect"):
        low = gut.likely("t", "q", stakes="low")
    with pytest.warns(UserWarning, match="has no effect"):
        high = gut.likely("t", "q", stakes="high")
    assert low.policy == high.policy == policy_for(lean=None)


def test_the_two_layers_cannot_be_mixed(backend: FakeBackend) -> None:
    with pytest.raises(PolicyError, match="not both"):
        gut.likely("t", "q", stakes="high", ask_human=True, cost_false_yes=2, cost_false_no=5)
    with pytest.raises(PolicyError, match="not both"):
        gut.likely("t", "q", lean="yes", threshold=0.4)
    with pytest.raises(PolicyError, match="not both"):
        gut.likely("t", "q", ask_human=True, unsure_band=(0.2, 0.8))


def test_the_error_points_at_presets(backend: FakeBackend) -> None:
    with pytest.raises(PolicyError, match=r"gut\.presets\(\)"):
        gut.likely("t", "q", lean="no", cost_false_yes=1, cost_false_no=1)


def test_exact_costs_still_work(backend: FakeBackend) -> None:
    decision = gut.likely("t", "q", cost_false_yes=2, cost_false_no=50, cost_human=1)
    assert decision.policy.cost_false_no == 50


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stakes": "enormous", "ask_human": True}, "stakes must be one of"),
        ({"lean": "maybe"}, "lean must be"),
    ],
)
def test_unrecognised_words_are_rejected(
    backend: FakeBackend, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(PolicyError, match=message):
        gut.likely("t", "q", **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- classify and rate


@pytest.mark.filterwarnings("ignore:stakes/ask_human on rate")
@pytest.mark.parametrize(("stakes", "floor"), STAKES_CONFIDENCE.items())
def test_posture_becomes_a_confidence_floor(
    backend: FakeBackend, stakes: Stakes, floor: float
) -> None:
    assert gut.classify("t", Team, stakes=stakes, ask_human=True).min_confidence == floor
    assert gut.rate("t", ["a", "b"], stakes=stakes, ask_human=True).min_confidence == floor


@pytest.mark.filterwarnings("ignore:stakes/ask_human on rate")
def test_a_low_confidence_answer_reaches_a_person(backend: FakeBackend) -> None:
    # FakeBackend answers a choice with 0.9 confidence, which clears every floor...
    assert gut.classify("t", Team, stakes="high", ask_human=True).outcome is Outcome.YES
    # ...but a score landing between levels is only 0.5 confident.
    gut.configure(backend=FakeBackend(default=1.5))
    assert gut.rate("t", ["a", "b", "c"], stakes="high", ask_human=True).outcome is Outcome.UNSURE


def test_without_ask_human_nothing_is_gated(backend: FakeBackend) -> None:
    assert gut.classify("t", Team).min_confidence is None
    assert gut.rate("t", ["a", "b"]).min_confidence is None


@pytest.mark.filterwarnings("ignore:stakes/ask_human on rate")
def test_classify_cannot_mix_the_layers(backend: FakeBackend) -> None:
    with pytest.raises(PolicyError, match="not both"):
        gut.classify("t", Team, ask_human=True, min_confidence=0.9)
    with pytest.raises(PolicyError, match="not both"):
        gut.rate("t", ["a", "b"], stakes="low", ask_human=True, min_confidence=0.9)


def test_min_confidence_alone_still_works(backend: FakeBackend) -> None:
    assert gut.classify("t", Team, min_confidence=0.95).outcome is Outcome.UNSURE


def test_stakes_on_classify_without_a_human_warns(backend: FakeBackend) -> None:
    with pytest.warns(UserWarning, match="has no effect"):
        assert gut.classify("t", Team, stakes="high").min_confidence is None


def test_bad_stakes_on_classify_is_rejected(backend: FakeBackend) -> None:
    with pytest.raises(PolicyError, match="stakes must be one of"):
        gut.classify("t", Team, stakes="enormous", ask_human=True)  # type: ignore[arg-type]


def test_a_posture_on_rate_warns_about_what_we_measured(backend: FakeBackend) -> None:
    """D23 found `rate` confidence running the wrong way on a real task, so a floor built on it
    may route away exactly the answers worth keeping. Warn, do not refuse: the mechanism is fine
    and the caller's task may not be the one we measured."""
    with pytest.warns(UserWarning, match="spread statistic rather than a probability"):
        decision = gut.rate("t", ["a", "b", "c"], stakes="high", ask_human=True)
    assert decision.min_confidence == STAKES_CONFIDENCE["high"]  # it still applies


def test_classify_does_not_get_that_warning(backend: FakeBackend) -> None:
    """The same measurement found `classify` confidence to be the best calibrated of the five."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gut.classify("t", Team, stakes="high", ask_human=True)


def test_an_explicit_floor_on_rate_is_left_alone(backend: FakeBackend) -> None:
    """A number the caller chose is a decision, not a default worth second-guessing."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gut.rate("t", ["a", "b", "c"], min_confidence=0.8)


def test_min_confidence_for_is_usable_directly() -> None:
    assert min_confidence_for("high", ask_human=True) == 0.80
    assert min_confidence_for(None, ask_human=True) == STAKES_CONFIDENCE["medium"]
    assert min_confidence_for(None, ask_human=False) is None
