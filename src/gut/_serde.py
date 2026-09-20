"""Converting answers to and from JSON.

Used by the on-disk cache and, later, by record/replay cassettes. Kept in one place so that a
change to the wire form invalidates both at once rather than leaving one of them silently reading
a shape the other no longer writes.

One asymmetry to know about: JSON object keys are always strings, but score answers key their
`probabilities` and `legend` by integer level. Round-tripping therefore has to coerce those keys
back, and a payload whose keys are not integers is corrupt rather than merely unexpected.
"""

from __future__ import annotations

from typing import Any

from gut._backends.base import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._errors import BackendError

SERIALISATION_VERSION = 1
"""Bumped when the JSON form changes incompatibly, so stale entries can be discarded."""


def answer_to_json(answer: Answer) -> dict[str, Any]:
    """Render an answer as a JSON-compatible dict."""
    match answer:
        case NoulAnswer():
            return {"type": "noul", "p": answer.p}
        case ChoiceAnswer():
            return {
                "type": "choice",
                "choice": answer.choice,
                "confidence": answer.confidence,
                "probabilities": dict(answer.probabilities),
            }
        case ScoreAnswer():  # pragma: no branch - exhaustive, proven by mypy
            return {
                "type": "score",
                "score": answer.score,
                "confidence": answer.confidence,
                "probabilities": {str(k): v for k, v in answer.probabilities.items()},
                "legend": {str(k): v for k, v in answer.legend.items()},
            }


def _require(data: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in data:
        raise BackendError(f"Serialised answer is missing {key!r}.")
    value = data[key]
    # bool is an int in Python; a probability arriving as True is corrupt, not merely odd.
    if isinstance(value, bool) or not isinstance(value, kind):
        expected = kind if isinstance(kind, tuple) else (kind,)
        names = " or ".join(cls.__name__ for cls in expected)
        raise BackendError(
            f"Serialised answer has {key!r} as {type(value).__name__}, expected {names}."
        )
    return value


def _int_keys(mapping: dict[str, Any], key: str) -> dict[int, Any]:
    try:
        return {int(k): v for k, v in mapping.items()}
    except (TypeError, ValueError) as error:
        raise BackendError(f"Serialised answer has non-integer level keys in {key!r}.") from error


def answer_from_json(data: dict[str, Any]) -> Answer:
    """Rebuild an answer from `answer_to_json` output.

    Raises:
        BackendError: The payload is missing fields, has the wrong types, or names an unknown
            answer kind.
    """
    kind = data.get("type")
    match kind:
        case "noul":
            return NoulAnswer(p=float(_require(data, "p", (int, float))))
        case "choice":
            return ChoiceAnswer(
                choice=_require(data, "choice", str),
                confidence=float(_require(data, "confidence", (int, float))),
                probabilities=dict(_require(data, "probabilities", dict)),
            )
        case "score":
            return ScoreAnswer(
                score=float(_require(data, "score", (int, float))),
                confidence=float(_require(data, "confidence", (int, float))),
                probabilities=_int_keys(_require(data, "probabilities", dict), "probabilities"),
                legend=_int_keys(_require(data, "legend", dict), "legend"),
            )
        case _:
            raise BackendError(f"Serialised answer has unknown type {kind!r}.")
