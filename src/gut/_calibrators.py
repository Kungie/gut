"""Fitting a correction to a model's probabilities.

D23 measured the problem: on the demo data `cancel_threat` is under-confident in the middle
(it said `0.52` and was right every time), `bug_report` is over-confident at the top, and
`urgency`'s confidence is close to inverted. The cost rule and the posture presets both take those
numbers at face value, so a band drawn at `0.25-0.75` is not where anyone thinks it is.

A calibrator is a function from what the model said to what it should have said, fitted on examples
where the truth is known. Two are provided:

- **Isotonic** (the default) — the best non-decreasing fit, by pool-adjacent-violators. It assumes
  nothing about the shape, only that a higher number should not mean a *less* likely event. Needs
  more data than Platt, and its failure mode is the useful one: given a relationship that is
  actually inverted, the best non-decreasing fit is a **constant**, so a worthless signal calibrates
  to "I have no information" rather than to a confident lie. See D24.
- **Platt** — a logistic fit in log-odds space, two parameters. Works on far less data and can only
  stretch or shift the curve, never bend it. Identity is `a=1, b=0`, so the fitted parameters read
  as "how far off was it".

Neither invents information. A calibrator can fix a number that ranks well and is wrongly scaled; it
cannot fix a number that does not rank.
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    cast,
    runtime_checkable,
)

from gut._errors import CalibrationError

if TYPE_CHECKING:
    from gut._backends.base import Answer

AnswerT = TypeVar("AnswerT", bound="Answer")

Method: TypeAlias = Literal["isotonic", "platt", "identity"]

EPSILON: Final = 1e-6
"""Probabilities are clamped away from 0 and 1 before taking log-odds."""

MIN_FIT_EXAMPLES: Final = 30
"""Below this, fitting is refused rather than done badly. Matches the eval reporting floor."""


@runtime_checkable
class Calibrator(Protocol):
    """A correction from what the model said to what it should have said."""

    def apply(self, probability: float) -> float:
        """Map one probability onto its corrected value, in `[0, 1]`."""
        ...

    def to_json(self) -> dict[str, Any]:
        """A form that can be written to disk and loaded back."""
        ...


@dataclass(frozen=True, slots=True)
class Identity:
    """Changes nothing. What you get when there is no reason to correct anything."""

    def apply(self, probability: float) -> float:
        """Return `probability` unchanged."""
        return probability

    def to_json(self) -> dict[str, Any]:
        """The identity calibrator's serialised form."""
        return {"method": "identity"}

    def __str__(self) -> str:
        return "identity (no correction)"


@dataclass(frozen=True, slots=True)
class Isotonic:
    """A non-decreasing step fit, interpolated between its breakpoints.

    Args:
        points: `(input, output)` breakpoints, sorted by input, with outputs non-decreasing.
    """

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise CalibrationError("An isotonic calibrator needs at least one breakpoint.")
        inputs = [x for x, _ in self.points]
        outputs = [y for _, y in self.points]
        if inputs != sorted(inputs) or outputs != sorted(outputs):
            raise CalibrationError("Isotonic breakpoints must be sorted and non-decreasing.")

    def apply(self, probability: float) -> float:
        """Interpolate `probability` between the fitted breakpoints, flat outside them."""
        if len(self.points) == 1:
            return self.points[0][1]
        if probability <= self.points[0][0]:
            return self.points[0][1]
        if probability >= self.points[-1][0]:
            return self.points[-1][1]
        for (left_x, left_y), (right_x, right_y) in zip(self.points, self.points[1:], strict=False):
            if left_x <= probability <= right_x:
                if right_x == left_x:  # pragma: no cover - inputs are deduplicated when fitting
                    return right_y
                share = (probability - left_x) / (right_x - left_x)
                return left_y + share * (right_y - left_y)
        raise AssertionError("unreachable: the loop covers the whole range")  # pragma: no cover

    def to_json(self) -> dict[str, Any]:
        """The fitted breakpoints."""
        return {
            "method": "isotonic",
            "points": [[round(x, 6), round(y, 6)] for x, y in self.points],
        }

    @property
    def flat(self) -> bool:
        """Whether the fit carries no information: every input maps to the same output."""
        outputs = {round(y, 9) for _, y in self.points}
        return len(outputs) == 1

    def __str__(self) -> str:
        if self.flat:
            return f"isotonic, flat at {self.points[0][1]:.3f} (the input carries no signal)"
        return f"isotonic, {len(self.points)} breakpoints"


@dataclass(frozen=True, slots=True)
class Platt:
    """A logistic correction in log-odds space: `sigmoid(a * logit(p) + b)`.

    `a = 1, b = 0` is the identity, so the fitted pair reads directly as how far off the model was:
    `a < 1` means it was over-confident, `b` shifts the whole curve.
    """

    a: float
    b: float

    def apply(self, probability: float) -> float:
        """Stretch and shift `probability` in log-odds space."""
        return _sigmoid(self.a * _logit(probability) + self.b)

    def to_json(self) -> dict[str, Any]:
        """The two fitted parameters."""
        return {"method": "platt", "a": round(self.a, 6), "b": round(self.b, 6)}

    def __str__(self) -> str:
        direction = "over-confident" if self.a < 1 else "under-confident"
        return f"platt a={self.a:.3f} b={self.b:.3f} (the model was {direction})"


def _clamp(probability: float) -> float:
    return min(1.0 - EPSILON, max(EPSILON, probability))


def _logit(probability: float) -> float:
    clamped = _clamp(probability)
    return math.log(clamped / (1.0 - clamped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)  # pragma: no branch - both sides are exercised
    return exponential / (1.0 + exponential)


def _pool_adjacent_violators(
    pairs: Sequence[tuple[float, bool]],
) -> list[tuple[float, float]]:
    """The best non-decreasing fit to `(probability, event)`, as `(input, output)` breakpoints.

    Points sharing an input are pooled first, then adjacent blocks that violate monotonicity are
    merged into their weighted mean until none remain -- which is the whole algorithm.
    """
    grouped: dict[float, list[float]] = {}
    for probability, event in pairs:
        grouped.setdefault(probability, []).append(float(event))

    # (input, mean outcome, weight)
    blocks: list[tuple[float, float, float]] = [
        (probability, sum(events) / len(events), float(len(events)))
        for probability, events in sorted(grouped.items())
    ]

    merged: list[tuple[float, float, float]] = []
    for block in blocks:
        merged.append(block)
        while len(merged) > 1 and merged[-2][1] > merged[-1][1]:
            right_x, right_y, right_w = merged.pop()
            _, left_y, left_w = merged.pop()
            weight = left_w + right_w
            merged.append((right_x, (left_y * left_w + right_y * right_w) / weight, weight))
    # Each block keeps the largest input it covers, so the fit steps up at the right places.
    return [(x, y) for x, y, _ in merged]


def _fit_platt(pairs: Sequence[tuple[float, bool]], *, steps: int = 60) -> Platt:
    """Fit `a` and `b` by gradient descent on log-loss.

    Sixty steps of plain gradient descent rather than Newton: the surface is convex and
    two-dimensional, the data is small, and avoiding a Hessian keeps this dependency-free.
    """
    features = [_logit(probability) for probability, _ in pairs]
    targets = [float(event) for _, event in pairs]
    count = len(pairs)

    a, b = 1.0, 0.0
    rate = 0.5
    for _ in range(steps):
        gradient_a = 0.0
        gradient_b = 0.0
        for feature, target in zip(features, targets, strict=True):
            error = _sigmoid(a * feature + b) - target
            gradient_a += error * feature
            gradient_b += error
        a -= rate * gradient_a / count
        b -= rate * gradient_b / count
    return Platt(a=a, b=b)


def fit(
    pairs: Sequence[tuple[float, bool]],
    *,
    method: Method = "isotonic",
    minimum: int = MIN_FIT_EXAMPLES,
) -> Calibrator:
    """Fit a correction from `(what the model said, what happened)` pairs.

    Args:
        pairs: Observations to fit on.
        method: `"isotonic"` (default), `"platt"`, or `"identity"`.
        minimum: Refuse to fit below this many examples. Pass `0` to fit anyway.

    Raises:
        CalibrationError: Too few examples, or an unknown method.
    """
    if method == "identity":
        return Identity()
    if len(pairs) < minimum:
        raise CalibrationError(
            f"Fitting a calibrator on {len(pairs)} examples would overfit it. Collect at least "
            f"{minimum}, or pass minimum=0 if you know what you are doing."
        )
    if not pairs:
        raise CalibrationError("Nothing to fit on.")
    if method == "isotonic":
        return Isotonic(points=tuple(_pool_adjacent_violators(pairs)))
    if method == "platt":
        return _fit_platt(pairs)
    raise CalibrationError(f"Unknown calibration method {method!r}.")


def from_json(document: dict[str, Any]) -> Calibrator:
    """Rebuild a calibrator from `to_json` output.

    Raises:
        CalibrationError: The document names no method, or one that is not recognised.
    """
    method = document.get("method")
    if method == "identity":
        return Identity()
    if method == "isotonic":
        points = document.get("points")
        if not isinstance(points, list):
            raise CalibrationError("An isotonic calibrator needs a 'points' list.")
        return Isotonic(points=tuple((float(x), float(y)) for x, y in points))
    if method == "platt":
        return Platt(a=float(document["a"]), b=float(document["b"]))
    raise CalibrationError(f"Unknown calibration method {method!r}.")


@dataclass
class CalibrationSet:
    """Calibrators keyed by the question they were fitted for.

    Keyed per question, never globally: a correction fitted on "is this a churn threat" says
    nothing about "is this a bug report", and applying one to the other would be worse than
    applying nothing. See D25.
    """

    entries: dict[str, Entry] = field(default_factory=dict)
    default: Calibrator = field(default_factory=Identity)
    """Applied to questions with no fitted calibrator. The identity, unless you mean otherwise."""

    def add(self, entry: Entry) -> None:
        """Register a fitted calibrator."""
        self.entries[entry.fingerprint] = entry

    def for_question(self, fingerprint: str, model: str | None = None) -> Calibrator:
        """The calibrator for this question, or the default.

        Warns:
            UserWarning: A calibrator exists but was fitted against a different model. It is still
                applied -- refusing would be worse -- but a correction is a claim about one model's
                distribution, so this is worth knowing about once.
        """
        entry = self.entries.get(fingerprint)
        if entry is None:
            return self.default
        if model is not None and entry.model is not None and entry.model != model:
            key = (fingerprint, model)
            if key not in self._warned:
                self._warned.add(key)
                warnings.warn(
                    f"The calibrator for this question was fitted against {entry.model!r} but "
                    f"{model!r} answered. A correction is a claim about one model's distribution; "
                    f"refit before trusting it.",
                    UserWarning,
                    stacklevel=3,
                )
        return entry.calibrator

    _warned: set[tuple[str, str]] = field(default_factory=set, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.entries)

    def to_json(self) -> dict[str, Any]:
        """The whole set, in a form meant to be committed and reviewed."""
        return {
            "version": 1,
            "entries": [entry.to_json() for entry in self.entries.values()],
        }

    def save(self, path: str | Path) -> None:
        """Write the set to disk as readable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_json(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibrationSet:
        """Read a set written by `save`.

        Raises:
            CalibrationError: The file is not a calibration set.
        """
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CalibrationError(
                f"{path}: cannot be read as a calibration set. {error}"
            ) from error
        if not isinstance(document, dict) or "entries" not in document:
            raise CalibrationError(f"{path}: not a calibration set; it has no 'entries'.")
        found = cls()
        for raw in document["entries"]:
            found.add(Entry.from_json(raw))
        return found


@dataclass(frozen=True, slots=True)
class Entry:
    """One fitted calibrator, with enough context to review it."""

    fingerprint: str
    calibrator: Calibrator
    question: dict[str, Any] | None = None
    """The canonical question, so a diff is readable without the code that produced it."""
    model: str | None = None
    examples: int | None = None
    before: dict[str, float] | None = None
    after: dict[str, float] | None = None
    """Brier and calibration error either side of the fit, so the file shows what it bought."""

    def to_json(self) -> dict[str, Any]:
        """This entry as JSON, leaving out what was not recorded."""
        payload: dict[str, Any] = {
            "fingerprint": self.fingerprint,
            "calibrator": self.calibrator.to_json(),
        }
        for name in ("question", "model", "examples", "before", "after"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> Entry:
        """Rebuild an entry.

        Raises:
            CalibrationError: The document is missing what an entry needs.
        """
        if "fingerprint" not in document or "calibrator" not in document:
            raise CalibrationError("A calibration entry needs a fingerprint and a calibrator.")
        return cls(
            fingerprint=str(document["fingerprint"]),
            calibrator=from_json(document["calibrator"]),
            question=document.get("question"),
            model=document.get("model"),
            examples=document.get("examples"),
            before=document.get("before"),
            after=document.get("after"),
        )


def correct(
    answer: AnswerT, fingerprint: str, model: str, calibration: CalibrationSet | None
) -> tuple[AnswerT, AnswerT | None]:
    """Apply a correction to one answer, returning `(what to use, what the model said)`.

    The second element is `None` when nothing was corrected, so callers can tell "unchanged" from
    "corrected to the same number".
    """
    from gut._backends.base import NoulAnswer

    if calibration is None:
        return answer, None
    curve = calibration.for_question(fingerprint, model)
    if isinstance(curve, Identity):
        return answer, None
    if isinstance(answer, NoulAnswer):
        return cast("AnswerT", dataclasses.replace(answer, p=curve.apply(answer.p))), answer
    corrected = dataclasses.replace(answer, confidence=curve.apply(answer.confidence))
    return cast("AnswerT", corrected), answer
