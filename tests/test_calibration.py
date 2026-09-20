"""Tests for the calibration measures."""

from __future__ import annotations

import pytest

from gut._calibration import MIN_EXAMPLES, calibrate


def test_a_perfect_forecaster_scores_zero() -> None:
    result = calibrate([(1.0, True)] * 20 + [(0.0, False)] * 20)
    assert result.brier == 0.0
    assert result.ece == 0.0
    assert result.count == 40


def test_confident_and_wrong_is_the_worst_score() -> None:
    assert calibrate([(1.0, False), (0.0, True)]).brier == 1.0


def test_hedging_at_a_half_scores_a_quarter() -> None:
    """Brier rewards being right *and* committed; 0.25 is what a coin flip costs."""
    result = calibrate([(0.5, True)] * 10 + [(0.5, False)] * 10)
    assert result.brier == 0.25
    assert result.ece == 0.0  # perfectly calibrated, just useless


def test_a_well_ranked_but_overconfident_model_shows_up_in_ece() -> None:
    """Every call is on the right side, so accuracy is perfect and the claim is still wrong."""
    pairs = [(0.9, True)] * 6 + [(0.9, False)] * 4
    result = calibrate(pairs)
    assert result.ece == pytest.approx(0.3)  # said 0.9, happened 0.6
    (bucket,) = result.bins
    assert bucket.predicted == pytest.approx(0.9)
    assert bucket.observed == pytest.approx(0.6)
    assert bucket.gap == pytest.approx(0.3)


def test_buckets_are_weighted_by_how_many_landed_in_them() -> None:
    # 90 predictions perfectly calibrated, 10 badly so.
    pairs = [(0.1, False)] * 90 + [(0.9, False)] * 10
    result = calibrate(pairs)
    assert result.ece == pytest.approx(0.9 * 0.1 + 0.1 * 0.9)


def test_empty_buckets_are_left_out() -> None:
    result = calibrate([(0.05, False), (0.95, True)], bins=10)
    assert [bucket.label for bucket in result.bins] == ["0.0-0.1", "0.9-1.0"]


def test_certainty_lands_in_the_last_bucket() -> None:
    """A probability of exactly 1.0 has no bucket unless the last one is closed."""
    result = calibrate([(1.0, True)], bins=5)
    assert [bucket.label for bucket in result.bins] == ["0.8-1.0"]


def test_nothing_measured_is_not_a_good_score() -> None:
    result = calibrate([])
    assert result.count == 0
    assert result.bins == ()
    assert result.reliable is False


def test_too_few_examples_says_so() -> None:
    thin = calibrate([(0.9, True)] * (MIN_EXAMPLES - 1))
    assert thin.reliable is False
    assert thin.caveat is not None
    assert "too few" in thin.caveat or "noise" in thin.caveat

    enough = calibrate([(0.9, True)] * MIN_EXAMPLES)
    assert enough.reliable is True
    assert enough.caveat is None


def test_bins_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        calibrate([(0.5, True)], bins=0)


def test_the_table_is_readable() -> None:
    table = calibrate([(0.9, True)] * 6 + [(0.9, False)] * 4).table()
    assert "p 0.8-1.0" in table
    assert "n=10" in table
    assert "said 0.90" in table
    assert "happened 0.60" in table


def test_json_round_trips_the_numbers() -> None:
    payload = calibrate([(0.9, True)] * 6 + [(0.9, False)] * 4).to_json()
    assert payload["count"] == 10
    assert payload["reliable"] is False
    assert payload["ece"] == pytest.approx(0.3)
    assert payload["bins"][0]["observed"] == pytest.approx(0.6)
