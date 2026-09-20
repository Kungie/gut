"""The decision log: a record of every judgment, and of how it turned out.

This is the foundation for calibration, which is not built yet and cannot be built later without the
data. A decision that was never recorded cannot be checked against reality, so the record is written
at the moment the decision is made -- with the site id that ties today's decision to next month's
outcome, the probability that produced it, the costs that were applied, and the exact model version
that answered.

`decision.resolve(actual=...)` closes the loop by recording what actually happened under the same
decision id. Nothing consumes that yet. The point is that the data path exists from the start, so
the first question anyone asks -- "is this model actually calibrated on *my* data?" -- is answerable
from logs that were already being written rather than from an instrumentation project.

Logging is observability, never correctness: a sink that raises is reported and swallowed. A broken
log must not break a decision.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gut._rule import Policy
    from gut._site import CallSite

logger = logging.getLogger("gut")

RecordKind = Literal["noul", "choice", "score"]


def _now() -> str:
    """An ISO-8601 timestamp in UTC: sortable as text, unambiguous across machines."""
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: object) -> Any:
    """Render a resolved outcome as something a log file can hold."""
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Everything known about one decision at the moment it was made."""

    id: str
    """The decision-site id, stable across runs. The join key for calibration."""
    kind: RecordKind
    timestamp: str
    outcome: str
    model: str
    """The exact version that answered, never an alias."""
    source: str
    """`backend`, `cache` or `prefetch`."""
    question: dict[str, Any]
    """The canonical question, so a record stays interpretable without the code that made it."""
    site: dict[str, Any]
    """Where the decision was made: module, function, file and line."""
    latency_ms: float | None = None
    p: float | None = None
    """Probability of yes, for a yes/no decision."""
    value: str | None = None
    """The chosen member's name, for a classification."""
    score: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    costs: dict[str, Any] | None = None
    """The policy that turned the probability into an outcome."""
    min_confidence: float | None = None

    def to_json(self) -> dict[str, Any]:
        """The record as a JSON-compatible dict, with empty fields left out."""
        payload: dict[str, Any] = {"type": "decision"}
        for name in self.__slots__:
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True, slots=True)
class ResolutionRecord:
    """What actually happened, recorded against the decision that predicted it."""

    id: str
    """The same decision-site id the decision was logged under."""
    timestamp: str
    actual: Any
    """Ground truth, however the caller expressed it."""
    decided: str
    """The outcome that was acted on, so a record is self-contained."""
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        """The record as a JSON-compatible dict, with empty fields left out."""
        payload: dict[str, Any] = {"type": "resolution"}
        for name in self.__slots__:
            value = getattr(self, name)
            if value is not None or name == "actual":
                payload[name] = value
        return payload


@runtime_checkable
class Sink(Protocol):
    """Somewhere decisions and their outcomes are written."""

    def decision(self, record: DecisionRecord) -> None:
        """Record a decision as it is made."""
        ...

    def resolution(self, record: ResolutionRecord) -> None:
        """Record how a decision turned out."""
        ...


class NullSink:
    """Writes nothing. The default, so `gut` never logs anywhere nobody asked for."""

    def decision(self, record: DecisionRecord) -> None:
        """Discard the record."""

    def resolution(self, record: ResolutionRecord) -> None:
        """Discard the record."""


@dataclass
class MemorySink:
    """Keeps records in a list. For tests, and for reports that run in one process."""

    decisions: list[DecisionRecord] = field(default_factory=list)
    resolutions: list[ResolutionRecord] = field(default_factory=list)

    def decision(self, record: DecisionRecord) -> None:
        """Append the decision."""
        self.decisions.append(record)

    def resolution(self, record: ResolutionRecord) -> None:
        """Append the resolution."""
        self.resolutions.append(record)

    def clear(self) -> None:
        """Forget everything recorded so far."""
        self.decisions.clear()
        self.resolutions.clear()


class JSONLSink:
    """Appends one JSON object per line to a file.

    Args:
        path: File to append to. Parent directories are created.

    Decisions and resolutions share the file and are told apart by their `type` field, so a run's
    predictions and their outcomes stay in one ordered stream.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8")

    def _write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def decision(self, record: DecisionRecord) -> None:
        """Append the decision."""
        self._write(record.to_json())

    def resolution(self, record: ResolutionRecord) -> None:
        """Append the resolution."""
        self._write(record.to_json())

    def close(self) -> None:
        """Close the file."""
        with self._lock:
            self._handle.close()

    def __enter__(self) -> JSONLSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def policy_to_json(rule: Policy) -> dict[str, Any]:
    """The costs or boundary a decision was made under."""
    payload: dict[str, Any] = {}
    for name in ("cost_false_yes", "cost_false_no", "cost_human", "threshold", "unsure_band"):
        value = getattr(rule, name)
        if value is not None:
            payload[name] = list(value) if isinstance(value, tuple) else value
    return payload


def site_to_json(site: CallSite) -> dict[str, Any]:
    """Where a decision was made.

    `module` and `function` are what the id is built from; `file` and `line` are for a human
    chasing it down, and deliberately do not affect identity (D11).
    """
    return {
        "module": site.module,
        "function": site.function,
        "file": site.filename,
        "line": site.lineno,
    }


def emit_decision(record: DecisionRecord, sink: Sink) -> None:
    """Write a decision record, never letting a broken sink break the decision."""
    try:
        sink.decision(record)
    except Exception:
        logger.warning("gut: decision sink %r failed", type(sink).__name__, exc_info=True)


def emit_resolution(record: ResolutionRecord, sink: Sink) -> None:
    """Write a resolution record, never letting a broken sink break the caller."""
    try:
        sink.resolution(record)
    except Exception:
        logger.warning("gut: resolution sink %r failed", type(sink).__name__, exc_info=True)
