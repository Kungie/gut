"""Roast your git history: what kind of change each commit is, and which ones say nothing at all.

    python examples/commit_roast.py                  # the last 25 commits of the repo you are in
    python examples/commit_roast.py --backend jev    # a model that knows commits better

Counting words is Python's job; telling a bug fix from a release is `gut`'s. The whole history goes
to the model in one `gut.each()` call. The jokes are ours.
"""

from __future__ import annotations

import enum
import subprocess
from dataclasses import dataclass

import gut


class Kind(enum.Enum):
    FEATURE = "a new feature"
    FIX = "a bug fix"
    REFACTOR = "restructuring code without changing behaviour"
    DOCS = "documentation"
    CHORE = "a release, a dependency update or tooling"
    OTHER = "something impossible to tell"


JOKES = {
    Kind.FEATURE: "Shipped. Somebody tell the changelog.",
    Kind.FIX: "A bug went to live on a farm upstate.",
    Kind.REFACTOR: "Same behaviour, new furniture.",
    Kind.DOCS: "Someone, somewhere, will read this. Maybe.",
    Kind.CHORE: "The unglamorous work that keeps the lights on.",
    Kind.OTHER: "Even the model is not sure what happened here.",
}

VAGUE = "Two words or fewer. Future you, reading git blame at 2 a.m., is not amused."

SAMPLE = [
    "fix",
    "wip",
    "stuff",
    "Fix off-by-one in pagination that skipped the last page",
    "Bump httpx to 0.28 so the new timeout API is available",
    "refactor",
    "Make the retry delay configurable for slow regions",
    "final final v2",
    "Update README",
]


@dataclass(frozen=True)
class Roast:
    message: str
    kind: str
    vague: bool
    joke: str


def commits(limit: int = 25) -> list[str]:
    """Subjects of the latest commits here, or a sample when this is not a git repository."""
    try:
        log = subprocess.run(
            ["git", "log", f"-{limit}", "--format=%s"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return SAMPLE
    return [line for line in log.stdout.splitlines() if line.strip()] or SAMPLE


def roast(messages: list[str]) -> list[Roast]:
    kinds = gut.each(messages).classify(Kind)
    roasts = []
    for message, decision in zip(messages, kinds, strict=True):
        kind = decision.value if isinstance(decision.value, Kind) else Kind.OTHER
        vague = len(message.split()) <= 2  # arithmetic: not a question for a model
        roasts.append(Roast(message, kind.name.lower(), vague, VAGUE if vague else JOKES[kind]))
    return roasts


def main() -> None:
    from _pick import backend_from_argv

    gut.configure(backend=backend_from_argv(__doc__ or ""))
    for item in roast(commits()):
        print(f"{'!!' if item.vague else '  '} {item.kind:<8} {item.message[:46]:<46}  {item.joke}")


if __name__ == "__main__":
    main()
