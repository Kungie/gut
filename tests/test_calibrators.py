"""Tests for fitting probability corrections."""

from __future__ import annotations

import pytest

from gut._calibration import calibrate
from gut._calibrators import (
    Calibrator,
    Identity,
    Isotonic,
    Platt,
    fit,
    from_json,
)
from gut._errors import CalibrationError


def labelled(probability: float, yes: int, no: int) -> list[tuple[float, bool]]:
    """`yes` positives and `no` negatives, all predicted at `probability`."""
    return [(probability, True)] * yes + [(probability, False)] * no


# The three shapes D23 found in real data.
UNDER_CONFIDENT = (
    labelled(0.07, 0, 85) + labelled(0.28, 2, 1) + labelled(0.52, 6, 0) + labelled(0.94, 7, 0)
)
OVER_CONFIDENT = (
    labelled(0.07, 0, 38) + labelled(0.31, 0, 13) + labelled(0.69, 5, 6) + labelled(0.92, 27, 9)
)
INVERTED = (
    labelled(0.31, 16, 2) + labelled(0.51, 19, 5) + labelled(0.70, 13, 15) + labelled(0.90, 27, 4)
)


# --------------------------------------------------------------------------- the pieces


def test_every_calibrator_satisfies_the_protocol() -> None:
    assert isinstance(Identity(), Calibrator)
    assert isinstance(Isotonic(points=((0.0, 0.0), (1.0, 1.0))), Calibrator)
    assert isinstance(Platt(a=1.0, b=0.0), Calibrator)


def test_identity_changes_nothing() -> None:
    identity = Identity()
    assert [identity.apply(p) for p in (0.0, 0.3, 1.0)] == [0.0, 0.3, 1.0]
    assert "no correction" in str(identity)


def test_platt_with_default_parameters_is_the_identity() -> None:
    """`a=1, b=0` is the no-op, so the fitted pair reads as how far off the model was."""
    platt = Platt(a=1.0, b=0.0)
    for probability in (0.05, 0.3, 0.5, 0.95):
        assert platt.apply(probability) == pytest.approx(probability, abs=1e-5)


def test_platt_stays_inside_the_unit_interval() -> None:
    extreme = Platt(a=8.0, b=4.0)
    assert 0.0 <= extreme.apply(0.0) <= 1.0
    assert 0.0 <= extreme.apply(1.0) <= 1.0


def test_isotonic_interpolates_and_flattens_outside() -> None:
    curve = Isotonic(points=((0.2, 0.1), (0.8, 0.9)))
    assert curve.apply(0.0) == 0.1
    assert curve.apply(0.2) == 0.1
    assert curve.apply(0.5) == pytest.approx(0.5)
    assert curve.apply(0.8) == 0.9
    assert curve.apply(1.0) == 0.9


def test_a_single_breakpoint_is_a_constant() -> None:
    curve = Isotonic(points=((0.5, 0.4),))
    assert curve.apply(0.0) == curve.apply(1.0) == 0.4
    assert curve.flat is True


def test_isotonic_rejects_a_curve_that_goes_down() -> None:
    with pytest.raises(CalibrationError, match="non-decreasing"):
        Isotonic(points=((0.2, 0.9), (0.8, 0.1)))
    with pytest.raises(CalibrationError, match="non-decreasing"):
        Isotonic(points=((0.8, 0.1), (0.2, 0.2)))
    with pytest.raises(CalibrationError, match="at least one breakpoint"):
        Isotonic(points=())


# --------------------------------------------------------------------------- fitting


def test_fitting_an_already_calibrated_model_barely_moves_it() -> None:
    pairs = labelled(0.1, 5, 45) + labelled(0.9, 45, 5)
    before = calibrate(pairs)
    curve = fit(pairs)
    after = calibrate([(curve.apply(p), event) for p, event in pairs])
    assert after.ece <= before.ece + 1e-9


@pytest.mark.parametrize("shape", [UNDER_CONFIDENT, OVER_CONFIDENT], ids=["under", "over"])
@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_fitting_improves_a_miscalibrated_model(
    shape: list[tuple[float, bool]], method: str
) -> None:
    before = calibrate(shape)
    curve = fit(shape, method=method)  # type: ignore[arg-type]
    after = calibrate([(curve.apply(p), event) for p, event in shape])
    assert after.ece < before.ece
    assert after.brier <= before.brier


def test_an_inverted_signal_collapses_to_a_constant() -> None:
    """The useful failure mode: a number that does not rank calibrates to "no information".

    Isotonic can only fit a non-decreasing curve, so the best fit to an inverted relationship is
    nearly flat -- which says the input carries no usable signal, instead of dressing it up.
    """
    curve = fit(INVERTED)
    assert isinstance(curve, Isotonic)
    outputs = [curve.apply(p) for p in (0.31, 0.51, 0.70)]
    assert len({round(value, 6) for value in outputs}) == 1
    assert outputs[0] == pytest.approx(0.686, abs=0.01)


def test_platt_reports_which_way_the_model_was_wrong() -> None:
    under = fit(UNDER_CONFIDENT, method="platt")
    assert isinstance(under, Platt)
    assert under.a > 1
    assert "under-confident" in str(under)

    over = fit(OVER_CONFIDENT, method="platt")
    assert isinstance(over, Platt)
    assert "confident" in str(over)


def test_identity_can_be_asked_for_explicitly() -> None:
    assert isinstance(fit([], method="identity"), Identity)


def test_fitting_on_too_little_data_is_refused() -> None:
    with pytest.raises(CalibrationError, match="would overfit"):
        fit(labelled(0.5, 5, 5))


def test_the_floor_can_be_waived() -> None:
    assert fit(labelled(0.5, 5, 5), minimum=0) is not None
    with pytest.raises(CalibrationError, match="Nothing to fit on"):
        fit([], minimum=0)


def test_an_unknown_method_is_rejected() -> None:
    with pytest.raises(CalibrationError, match="Unknown calibration method"):
        fit(UNDER_CONFIDENT, method="magic")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- serialising


@pytest.mark.parametrize(
    "curve",
    [Identity(), Isotonic(points=((0.1, 0.0), (0.9, 1.0))), Platt(a=1.5, b=-0.2)],
    ids=["identity", "isotonic", "platt"],
)
def test_calibrators_round_trip(curve: Calibrator) -> None:
    restored = from_json(curve.to_json())
    for probability in (0.05, 0.4, 0.95):
        assert restored.apply(probability) == pytest.approx(curve.apply(probability), abs=1e-5)


def test_a_fitted_curve_round_trips() -> None:
    curve = fit(UNDER_CONFIDENT)
    restored = from_json(curve.to_json())
    assert restored.apply(0.52) == pytest.approx(curve.apply(0.52))


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({}, "Unknown calibration method"),
        ({"method": "magic"}, "Unknown calibration method"),
        ({"method": "isotonic"}, "needs a 'points' list"),
    ],
)
def test_a_corrupt_document_is_rejected(document: dict[str, object], message: str) -> None:
    with pytest.raises(CalibrationError, match=message):
        from_json(document)
