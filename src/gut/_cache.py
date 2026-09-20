"""Answer caching.

Asking the same question about the same state twice is pure waste: the answer is deterministic
enough that the second call buys nothing, and billing is on input tokens, so the state is paid for
again every time. An in-memory LRU is on by default; `SQLiteCache` survives process restarts, and
`NullCache` turns caching off.

The key covers the state, the question *and the model asked for*, because an answer is only
interchangeable with another answer from the same model. See D12 for what that means when the model
is named by a moving alias.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from gut._backends.base import Answer
from gut._questions import QuestionSpec, State, canonical_json
from gut._serde import SERIALISATION_VERSION, answer_from_json, answer_to_json

DEFAULT_MAXSIZE: Final = 1024
"""Entries kept by the default in-memory cache before the least recently used is evicted."""


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A cached answer together with the model that actually produced it."""

    answer: Answer
    model: str
    """The resolved model version from the response, which may be more specific than the alias in
    the key."""


def cache_key(state: State, spec: QuestionSpec, model: str) -> str:
    """The identity of "this question, about this state, asked of this model"."""
    payload = canonical_json(
        {
            "v": SERIALISATION_VERSION,
            "state": state,
            "question": spec.canonical(),
            "model": model,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class Cache(Protocol):
    """Somewhere to keep answers between calls."""

    def get(self, key: str) -> CacheEntry | None:
        """Return the entry for `key`, or `None` on a miss."""
        ...

    def set(self, key: str, entry: CacheEntry) -> None:
        """Store `entry` under `key`."""
        ...

    def clear(self) -> None:
        """Forget everything."""
        ...


class NullCache:
    """Caches nothing. Use it to turn caching off."""

    def get(self, key: str) -> CacheEntry | None:
        """Always a miss."""
        return None

    def set(self, key: str, entry: CacheEntry) -> None:
        """Discard the entry."""

    def clear(self) -> None:
        """Nothing to forget."""


class MemoryCache:
    """A bounded, thread-safe LRU cache held in memory.

    Args:
        maxsize: Entries to keep before evicting the least recently used. Zero disables storage
            while still reporting hits and misses honestly.
    """

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE) -> None:
        if maxsize < 0:
            raise ValueError(f"maxsize must be non-negative, got {maxsize}.")
        self._maxsize = maxsize
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def maxsize(self) -> int:
        """The configured bound."""
        return self._maxsize

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> CacheEntry | None:
        """Return the entry for `key` and mark it most recently used."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        """Store `entry`, evicting the least recently used if the cache is full."""
        if self._maxsize == 0:
            return
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Forget everything."""
        with self._lock:
            self._entries.clear()


class SQLiteCache:
    """An on-disk cache that survives process restarts.

    Args:
        path: Database file. Parent directories are created. `":memory:"` works for tests.

    The connection is shared across threads under a lock rather than opened per call, so a long-
    lived process does not pay connection setup on every question.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS answers (
            key        TEXT PRIMARY KEY,
            model      TEXT NOT NULL,
            payload    TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """

    def __init__(self, path: str | Path = ".gut_cache/answers.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        with self._transaction() as connection:
            connection.execute(self._SCHEMA)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._connection:
            yield self._connection

    def get(self, key: str) -> CacheEntry | None:
        """Return the entry for `key`, or `None` on a miss or an unreadable row."""
        with self._lock:
            row = self._connection.execute(
                "SELECT model, payload FROM answers WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        model, payload = row
        return CacheEntry(answer=answer_from_json(json.loads(payload)), model=str(model))

    def set(self, key: str, entry: CacheEntry) -> None:
        """Store `entry`, replacing any existing row for `key`."""
        payload = json.dumps(answer_to_json(entry.answer), separators=(",", ":"))
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO answers (key, model, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, entry.model, payload, time.time()),
            )

    def clear(self) -> None:
        """Delete every row."""
        with self._transaction() as connection:
            connection.execute("DELETE FROM answers")

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._connection.close()

    def __len__(self) -> int:
        with self._lock:
            (count,) = self._connection.execute("SELECT COUNT(*) FROM answers").fetchone()
        return int(count)
