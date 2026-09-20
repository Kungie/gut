"""Pulling code out of the project's own documentation, so it can be checked.

Shared by the README and skill tests. Duplicating this is how two copies drift and one of them
quietly stops asserting anything, which has already happened once here.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BLOCK = re.compile(r"^```(\w+)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def read(name: str) -> str:
    """One of the project's documents, by path relative to the repository root."""
    return (ROOT / name).read_text(encoding="utf-8")


def blocks(name: str, language: str) -> list[tuple[int, str]]:
    """Every fenced block of `language` in the document, with the line it starts on."""
    document = read(name)
    return [
        (document[: match.start()].count("\n") + 2, match.group(2))
        for match in BLOCK.finditer(document)
        if match.group(1) == language
    ]
