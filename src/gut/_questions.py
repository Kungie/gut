"""Backend-agnostic question specifications.

These are `gut`'s own shapes, not the vendor's. Keeping a layer here is what lets a second backend
exist later without the public API changing, and it is where the limits documented for Jev are
enforced *before* a request goes out: the SDK's wire schema only requires a score rubric to be
nonempty, while the API wants at least two levels, so an unchecked one-level rubric costs a round
trip and returns an opaque 422. See D5 in DECISIONS.md.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, TypeAlias

from gut._errors import QuestionError

State: TypeAlias = "str | Mapping[str, Any] | Sequence[Any]"
"""What a question is asked about: text, a JSON object, or an array. Text only -- no binary."""

MIN_SCORE_LEVELS: Final = 2
MAX_SCORE_LEVELS: Final = 10
MIN_CHOICE_OPTIONS: Final = 2
MAX_CHOICE_OPTIONS: Final = 255


def canonical_json(value: object) -> str:
    """Serialise `value` so that equal content always produces an identical string.

    Keys are sorted and whitespace removed, so cache keys and decision-site ids stay stable across
    runs, dict ordering and Python versions.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuestionError(f"{name} must be a non-empty string, got {value!r}.")
    return value


@dataclass(frozen=True, slots=True)
class NoulSpec:
    """A yes/no question."""

    instructions: str
    """The question or statement to evaluate."""
    yes_means: str | None = None
    """Optional description of what counts as yes."""
    no_means: str | None = None
    """Optional description of what counts as no."""
    _cached_fingerprint: str = field(init=False, repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        _require_text("question", self.instructions)
        for name, value in (("yes_means", self.yes_means), ("no_means", self.no_means)):
            if value is not None:
                _require_text(name, value)
        object.__setattr__(self, "_cached_fingerprint", _fingerprint(self.canonical()))

    def canonical(self) -> dict[str, Any]:
        """The stable, comparable form of this question."""
        spec: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        criteria = {
            key: value
            for key, value in (("true", self.yes_means), ("false", self.no_means))
            if value is not None
        }
        if criteria:
            spec["criteria"] = criteria
        return spec

    @property
    def fingerprint(self) -> str:
        """A short, stable hash of this question, used in cache keys and decision-site ids."""
        return self._cached_fingerprint

    def __hash__(self) -> int:
        """Hash on the fingerprint.

        `frozen=True` would otherwise generate a hash over the fields, which raises for a spec
        holding a `dict`. Equal specs have equal canonical forms and so equal fingerprints, which
        is exactly the contract a hash needs.
        """
        return hash(self._cached_fingerprint)


@dataclass(frozen=True, slots=True)
class ChoiceSpec:
    """A question that selects one of several named options."""

    instructions: str | None
    """What to decide. `None` leaves it to the options to speak for themselves, which the API
    allows and which is often enough for a well-described set of categories."""
    criteria: Mapping[str, str | None]
    """Option names mapped to descriptions of when each applies; `None` means the name speaks
    for itself."""
    _cached_fingerprint: str = field(init=False, repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        if self.instructions is not None:
            _require_text("question", self.instructions)
        if not isinstance(self.criteria, Mapping):
            raise QuestionError(f"criteria must be a mapping, got {type(self.criteria).__name__}.")
        count = len(self.criteria)
        if not MIN_CHOICE_OPTIONS <= count <= MAX_CHOICE_OPTIONS:
            raise QuestionError(
                f"A choice needs between {MIN_CHOICE_OPTIONS} and {MAX_CHOICE_OPTIONS} options, "
                f"got {count}."
            )
        for name, description in self.criteria.items():
            _require_text("An option name", name)
            if description is not None:
                _require_text(f"The description for option {name!r}", description)
        object.__setattr__(self, "_cached_fingerprint", _fingerprint(self.canonical()))

    def canonical(self) -> dict[str, Any]:
        """The stable, comparable form of this question."""
        spec: dict[str, Any] = {"type": "choice", "criteria": dict(self.criteria)}
        if self.instructions is not None:
            spec["instructions"] = self.instructions
        return spec

    @property
    def fingerprint(self) -> str:
        """A short, stable hash of this question, used in cache keys and decision-site ids."""
        return self._cached_fingerprint

    def __hash__(self) -> int:
        """Hash on the fingerprint.

        `frozen=True` would otherwise generate a hash over the fields, which raises for a spec
        holding a `dict`. Equal specs have equal canonical forms and so equal fingerprints, which
        is exactly the contract a hash needs.
        """
        return hash(self._cached_fingerprint)


@dataclass(frozen=True, slots=True)
class ScoreSpec:
    """A question that rates the state against an ordered rubric."""

    instructions: str | None
    """What to rate. `None` leaves it to the rubric to speak for itself."""
    criteria: Sequence[str]
    """Level descriptions in order; the first is level 0.

    Normalised to a `tuple` on construction, so two specs built from a list and a tuple of the same
    strings compare equal and fingerprint alike.
    """
    _cached_fingerprint: str = field(init=False, repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        if self.instructions is not None:
            _require_text("question", self.instructions)
        if isinstance(self.criteria, str) or not isinstance(self.criteria, Sequence):
            raise QuestionError(
                f"levels must be a sequence of strings, got {type(self.criteria).__name__}."
            )
        object.__setattr__(self, "criteria", tuple(self.criteria))
        count = len(self.criteria)
        if not MIN_SCORE_LEVELS <= count <= MAX_SCORE_LEVELS:
            raise QuestionError(
                f"A score needs between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels, "
                f"got {count}."
            )
        for index, level in enumerate(self.criteria):
            _require_text(f"Level {index}", level)
        object.__setattr__(self, "_cached_fingerprint", _fingerprint(self.canonical()))

    def canonical(self) -> dict[str, Any]:
        """The stable, comparable form of this question."""
        spec: dict[str, Any] = {"type": "score", "criteria": list(self.criteria)}
        if self.instructions is not None:
            spec["instructions"] = self.instructions
        return spec

    @property
    def fingerprint(self) -> str:
        """A short, stable hash of this question, used in cache keys and decision-site ids."""
        return self._cached_fingerprint

    def __hash__(self) -> int:
        """Hash on the fingerprint.

        `frozen=True` would otherwise generate a hash over the fields, which raises for a spec
        holding a `dict`. Equal specs have equal canonical forms and so equal fingerprints, which
        is exactly the contract a hash needs.
        """
        return hash(self._cached_fingerprint)


QuestionSpec: TypeAlias = NoulSpec | ChoiceSpec | ScoreSpec
"""Any question `gut` knows how to ask."""


def state_fingerprint(state: State) -> str:
    """A short, stable hash of the state a question is asked about."""
    return _fingerprint(state)
