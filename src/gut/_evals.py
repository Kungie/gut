"""Example files: testing a judgment the way you test anything else.

A semantic decision is not untestable, it is just tested differently. You cannot assert an exact
output, but you can assert that a predicate agrees with you on cases you have written down, often
enough to be worth relying on.

```yaml
# predicates/cancel_threat.yaml
question: "the customer threatens to cancel"
examples:
  - text: "If this happens again I'm cancelling my subscription."
    expected: yes
  - text: "How do I cancel my subscription?"
    expected: no
min_accuracy: 0.9
```

The hard cases are the point. A file of obvious examples proves nothing; the ones worth writing down
are the ones you had to think about, and the ones that went wrong in production.

`min_accuracy` defaults to `1.0`. Judgment tasks rarely justify that, but a threshold that silently
tolerates failures is worse than one you had to lower on purpose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml

from gut._backends.base import Backend, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._batching import fetch
from gut._calibration import DEFAULT_BINS, Calibration, calibrate
from gut._errors import EvalError
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State

Kind: TypeAlias = Literal["noul", "choice", "score"]

CalibrationKind: TypeAlias = Literal["probability", "confidence"]
"""What the number being calibrated claims to be.

A yes/no answer's `p` is a claim about the event. A choice or score answer's `confidence` is a
statistic over its own distribution, which nobody claimed was a probability of being right --
measuring it against correctness is how you find out whether treating it as one holds up.
"""

DEFAULT_MIN_ACCURACY = 1.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_TOLERANCE = 0.5


@dataclass(frozen=True, slots=True)
class Example:
    """One case the predicate is expected to get right."""

    state: State
    expected: object
    tolerance: float | None = None
    """Score files only: how far the score may land from `expected`."""

    @property
    def label(self) -> str:
        """A short description for failure output."""
        text = self.state if isinstance(self.state, str) else str(self.state)
        return text if len(text) <= 70 else text[:67] + "..."


@dataclass(frozen=True, slots=True)
class EvalSuite:
    """A predicate example file, parsed."""

    path: Path
    kind: Kind
    spec: QuestionSpec
    examples: tuple[Example, ...]
    min_accuracy: float = DEFAULT_MIN_ACCURACY
    max_ece: float | None = None
    """Optional: fail the file if its expected calibration error exceeds this."""
    threshold: float = DEFAULT_THRESHOLD
    """Noul files only: the probability above which the predicate counts as yes."""
    tolerance: float = DEFAULT_TOLERANCE
    """Score files only: the default distance allowed from the expected level."""

    @property
    def name(self) -> str:
        """The file's stem, used as the test name."""
        return self.path.stem


@dataclass(frozen=True, slots=True)
class ExampleResult:
    """How one example turned out."""

    example: Example
    correct: bool
    actual: str
    """What the model said, rendered for a report."""
    probability: float
    """The number the model attached to its answer: `p` for yes/no, confidence otherwise."""
    event: bool
    """What that number was a claim about, so the two can be calibrated against each other."""


@dataclass(frozen=True, slots=True)
class EvalResult:
    """How a whole file turned out."""

    suite: EvalSuite
    results: tuple[ExampleResult, ...]

    @property
    def correct(self) -> int:
        """How many examples the model got right."""
        return sum(1 for result in self.results if result.correct)

    @property
    def accuracy(self) -> float:
        """Fraction correct. An empty file scores zero rather than a vacuous one."""
        return self.correct / len(self.results) if self.results else 0.0

    @property
    def calibration(self) -> Calibration:
        """How well the model's numbers matched what happened."""
        return self.calibration_with()

    def calibration_with(self, bins: int = DEFAULT_BINS) -> Calibration:
        """The same, at a chosen number of buckets."""
        return calibrate([(r.probability, r.event) for r in self.results], bins=bins)

    @property
    def calibration_kind(self) -> CalibrationKind:
        """Whether the calibrated number is a probability of the event or a confidence."""
        return "probability" if self.suite.kind == "noul" else "confidence"

    @property
    def ece_exceeded(self) -> bool:
        """Whether the file declared a calibration bar and missed it."""
        return self.suite.max_ece is not None and self.calibration.ece > self.suite.max_ece

    @property
    def passed(self) -> bool:
        """Whether the file met its own bars.

        Accuracy always. Calibration only if the file opted in with `max_ece`, because a file
        written before anyone measured calibration should not start failing.
        """
        return self.accuracy >= self.suite.min_accuracy and not self.ece_exceeded

    @property
    def failures(self) -> tuple[ExampleResult, ...]:
        """The examples that went the wrong way."""
        return tuple(result for result in self.results if not result.correct)


# --------------------------------------------------------------------------- loading


def _require(document: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in document:
        raise EvalError(f"{path}: missing required key {key!r}.")
    return document[key]


def _number(value: object, key: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvalError(f"{path}: {key} must be a number, got {value!r}.")
    return float(value)


def _kind_of(document: Mapping[str, Any], path: Path) -> Kind:
    declared = document.get("type")
    if declared is not None:
        if declared == "noul":
            return "noul"
        if declared == "choice":
            return "choice"
        if declared == "score":
            return "score"
        raise EvalError(f"{path}: type must be noul, choice or score, got {declared!r}.")
    if "options" in document:
        return "choice"
    if "levels" in document:
        return "score"
    return "noul"


def _example_from(raw: object, kind: Kind, path: Path, index: int) -> Example:
    where = f"{path}: example {index}"
    if not isinstance(raw, Mapping):
        raise EvalError(f"{where} must be a mapping, got {type(raw).__name__}.")
    if "text" in raw:
        state: State = raw["text"]
    elif "state" in raw:
        state = raw["state"]
    else:
        raise EvalError(f"{where} needs a 'text' or 'state' key.")
    if "expected" not in raw:
        raise EvalError(f"{where} needs an 'expected' key.")

    expected = raw["expected"]
    if kind == "noul" and not isinstance(expected, bool):
        raise EvalError(f"{where}: expected must be yes or no, got {expected!r}.")
    if kind == "choice" and not isinstance(expected, str):
        raise EvalError(f"{where}: expected must be an option name, got {expected!r}.")
    if kind == "score":
        expected = _number(expected, "expected", path)

    tolerance = raw.get("tolerance")
    return Example(
        state=state,
        expected=expected,
        tolerance=None if tolerance is None else _number(tolerance, "tolerance", path),
    )


def _spec_from(document: Mapping[str, Any], kind: Kind, path: Path) -> QuestionSpec:
    question = document.get("question")
    if question is not None and not isinstance(question, str):
        raise EvalError(f"{path}: question must be a string, got {question!r}.")

    if kind == "noul":
        if question is None:
            raise EvalError(f"{path}: a yes/no file needs a 'question'.")
        return NoulSpec(
            instructions=question,
            yes_means=document.get("yes_means"),
            no_means=document.get("no_means"),
        )
    if kind == "choice":
        options = _require(document, "options", path)
        if not isinstance(options, Mapping):
            raise EvalError(f"{path}: options must be a mapping of name to description.")
        return ChoiceSpec(instructions=question, criteria=dict(options))
    levels = _require(document, "levels", path)
    if isinstance(levels, str) or not isinstance(levels, Sequence):
        raise EvalError(f"{path}: levels must be a list of descriptions.")
    return ScoreSpec(instructions=question, criteria=list(levels))


def looks_like_suite(document: object) -> bool:
    """Whether a parsed YAML document is a predicate file rather than some other YAML."""
    return isinstance(document, Mapping) and "examples" in document


def load_suite(path: str | Path) -> EvalSuite:
    """Parse a predicate example file.

    Raises:
        EvalError: The file is not valid YAML, or is not a well-formed predicate file.
    """
    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise EvalError(f"{path}: not valid YAML. {error}") from error

    if not looks_like_suite(document):
        raise EvalError(f"{path}: not a predicate file; it has no 'examples'.")
    assert isinstance(document, Mapping)

    kind = _kind_of(document, path)
    raw_examples = document["examples"]
    if not isinstance(raw_examples, Sequence) or isinstance(raw_examples, str):
        raise EvalError(f"{path}: examples must be a list.")
    if not raw_examples:
        raise EvalError(f"{path}: examples is empty; there is nothing to check.")

    try:
        spec = _spec_from(document, kind, path)
    except EvalError:
        raise
    except Exception as error:
        raise EvalError(f"{path}: {error}") from error

    return EvalSuite(
        path=path,
        kind=kind,
        spec=spec,
        examples=tuple(
            _example_from(raw, kind, path, index) for index, raw in enumerate(raw_examples)
        ),
        min_accuracy=_number(
            document.get("min_accuracy", DEFAULT_MIN_ACCURACY), "min_accuracy", path
        ),
        max_ece=(
            None
            if document.get("max_ece") is None
            else _number(document["max_ece"], "max_ece", path)
        ),
        threshold=_number(document.get("threshold", DEFAULT_THRESHOLD), "threshold", path),
        tolerance=_number(document.get("tolerance", DEFAULT_TOLERANCE), "tolerance", path),
    )


# --------------------------------------------------------------------------- running


def _judge_example(suite: EvalSuite, example: Example, backend: Backend) -> ExampleResult:
    answers = fetch(example.state, [suite.spec], backend)
    answer = answers[suite.spec].answer

    if isinstance(answer, NoulAnswer):
        said_yes = answer.p > suite.threshold
        return ExampleResult(
            example=example,
            correct=said_yes == example.expected,
            actual=f"{'yes' if said_yes else 'no'} (p={answer.p:.3f})",
            probability=answer.p,
            event=bool(example.expected),
        )
    if isinstance(answer, ChoiceAnswer):
        correct = answer.choice == example.expected
        return ExampleResult(
            example=example,
            correct=correct,
            actual=f"{answer.choice} (confidence={answer.confidence:.2f})",
            probability=answer.confidence,
            event=correct,
        )
    if isinstance(answer, ScoreAnswer):
        tolerance = suite.tolerance if example.tolerance is None else example.tolerance
        expected = float(example.expected)  # type: ignore[arg-type]
        correct = abs(answer.score - expected) <= tolerance
        return ExampleResult(
            example=example,
            correct=correct,
            actual=f"{answer.score:.2f} (tolerance {tolerance})",
            probability=answer.confidence,
            event=correct,
        )
    raise EvalError(  # pragma: no cover - the answer kind always matches the question kind
        f"{suite.path}: unexpected answer type {type(answer).__name__}."
    )


def run_suite(suite: EvalSuite, backend: Backend) -> EvalResult:
    """Run every example in `suite` and report how it went."""
    return EvalResult(
        suite=suite,
        results=tuple(_judge_example(suite, example, backend) for example in suite.examples),
    )


def format_failures(result: EvalResult) -> str:
    """A readable account of what went wrong, for a test report."""
    suite = result.suite
    headline = (
        f"{suite.name}: accuracy {result.accuracy:.0%} "
        f"({result.correct}/{len(result.results)}), below min_accuracy {suite.min_accuracy:.0%}"
    )
    if result.ece_exceeded:
        assert suite.max_ece is not None
        headline = (
            f"{suite.name}: calibration error {result.calibration.ece:.3f}, "
            f"above max_ece {suite.max_ece:.3f}"
        )
    lines = [headline, f"  question: {suite.spec.instructions or suite.spec.canonical()}", ""]
    for failure in result.failures:
        lines.append(f"  expected {failure.example.expected!r}, got {failure.actual}")
        lines.append(f"    {failure.example.label}")
    return "\n".join(lines)
