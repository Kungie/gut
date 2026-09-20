"""Record/replay: running the suite offline, deterministically, against real answers.

A cassette is a JSON file of real backend answers. Record it once against the live model, commit it,
and every run afterwards is offline, instant and identical -- while still testing against what the
model actually said rather than what a fixture author imagined.

```bash
GUT_RECORD=1 pytest    # ask for real, and write down the answers
pytest                 # replay them
```

Two design choices are worth knowing.

**Entries are per question, not per request.** The key is the state and the question, the same pair
the cache uses -- not the batch they happened to travel in. So a cassette recorded before adding
`@semantic` still replays afterwards, and regrouping questions does not invalidate it.

**A miss is an error, never a silent call.** In replay mode an unrecorded question fails loudly and
names itself. The alternative -- quietly reaching for the network -- turns one forgotten re-record
into a test suite that passes on someone's laptop, fails in CI, and bills you either way.

Cassettes store the state, the question and the answer in full rather than hashes, so a reviewer can
read a diff and see what changed about the model's behaviour.

**That means a cassette contains whatever you asked about, verbatim.** Record against real customer
tickets and commit the file, and you have committed customer text to your repository. Record against
fixtures you are happy to publish, or keep the cassette out of version control and regenerate it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gut._backends.base import Answer, Backend, BackendResponse
from gut._errors import CassetteMissError
from gut._questions import QuestionSpec, State, canonical_json
from gut._serde import SERIALISATION_VERSION, answer_from_json, answer_to_json

RECORD_ENV = "GUT_RECORD"
"""Set this to `1` (or `true`/`yes`) to record instead of replay."""

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def record_requested() -> bool:
    """Whether the environment is asking for a recording run."""
    return os.environ.get(RECORD_ENV, "").strip().lower() in _TRUTHY


def _entry_key(state: State, question: Mapping[str, Any], model: str) -> str:
    return canonical_json({"state": state, "question": dict(question), "model": model})


@dataclass
class Cassette:
    """A file of recorded answers.

    Args:
        path: Where the cassette lives. It need not exist yet in record mode.
    """

    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    dirty: bool = False

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            self.load()

    def load(self) -> None:
        """Read the cassette from disk, replacing anything held in memory."""
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = {
            _entry_key(entry["state"], entry["question"], entry["model"]): entry
            for entry in document.get("entries", [])
        }
        self.dirty = False

    def save(self) -> None:
        """Write the cassette to disk, sorted so that diffs are readable."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "version": SERIALISATION_VERSION,
            "entries": sorted(
                self.entries.values(),
                key=lambda entry: canonical_json([entry["question"], entry["state"]]),
            ),
        }
        self.path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.dirty = False

    def get(self, state: State, spec: QuestionSpec, model: str) -> Answer | None:
        """The recorded answer for this question, or `None`."""
        entry = self.entries.get(_entry_key(state, spec.canonical(), model))
        return None if entry is None else answer_from_json(entry["answer"])

    def put(self, state: State, spec: QuestionSpec, model: str, answer: Answer) -> None:
        """Record an answer."""
        question = spec.canonical()
        self.entries[_entry_key(state, question, model)] = {
            "state": state,
            "question": question,
            "model": model,
            "answer": answer_to_json(answer),
        }
        self.dirty = True

    def __len__(self) -> int:
        return len(self.entries)


class CassetteBackend:
    """A backend that replays a cassette, or records one.

    Args:
        cassette: The cassette, or a path to one.
        inner: The real backend, needed only when recording.
        record: Force record or replay mode. Defaults to reading `GUT_RECORD`.
        model: The model id to report. Defaults to the inner backend's, or `"cassette"` when
            replaying without one.
    """

    def __init__(
        self,
        cassette: Cassette | str | Path,
        inner: Backend | None = None,
        *,
        record: bool | None = None,
        model: str | None = None,
    ) -> None:
        self.cassette = cassette if isinstance(cassette, Cassette) else Cassette(Path(cassette))
        self.inner = inner
        self.recording = record_requested() if record is None else record
        if self.recording and inner is None:
            raise CassetteMissError(
                f"Recording to {self.cassette.path} needs a real backend to record from; "
                f"pass inner=."
            )
        if model is not None:
            self._model = model
        elif inner is not None:
            self._model = inner.model_id
        else:
            self._model = "cassette"

    @property
    def model_id(self) -> str:
        """The model the recorded answers belong to."""
        return self._model

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """Serve every question from the cassette, recording the misses if allowed.

        Raises:
            CassetteMissError: Replaying, and something was never recorded.
        """
        answers: dict[str, Answer] = {}
        missing: dict[str, QuestionSpec] = {}
        for name, spec in questions.items():
            recorded = self.cassette.get(state, spec, self._model)
            if recorded is None:
                missing[name] = spec
            else:
                answers[name] = recorded

        if missing and not self.recording:
            described = ", ".join(
                repr(spec.instructions or spec.canonical()) for spec in missing.values()
            )
            raise CassetteMissError(
                f"{self.cassette.path} has no recording for {described}. Re-record with "
                f"{RECORD_ENV}=1, or check whether the question or the subject changed."
            )

        if missing:
            assert self.inner is not None
            response = self.inner.ask(state, missing)
            for name, spec in missing.items():
                answer = response.answers[name]
                self.cassette.put(state, spec, self._model, answer)
                answers[name] = answer

        return BackendResponse(answers=answers, model=self._model)

    def save(self) -> None:
        """Write the cassette out if anything new was recorded."""
        if self.cassette.dirty:
            self.cassette.save()

    def __enter__(self) -> CassetteBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.save()
