"""Comment moderation on a budget: a free local model settles the obvious, a bigger one the rest.

    python examples/moderation.py                    # NLI model first, then Qwen3-0.6B
    python examples/moderation.py --backend openai   # NLI model first, then gpt-4.1-nano

`gut.Cascade` asks its backends in order and keeps an answer as soon as one is sure enough. The
comments only a bigger model can read go to it; everything else never leaves the machine.
"""

from __future__ import annotations

from collections import Counter

import gut


def moderate(comment: str) -> tuple[str, set[str]]:
    """What to do with a comment, and which models it took to decide."""
    with gut.judge(comment) as j:
        spam = j.likely("is spam", ask_human=True)
        abuse = j.likely("is abusive", stakes="high", ask_human=True)

    models = {spam.model, abuse.model}
    if spam == gut.YES or abuse == gut.YES:
        return "hide", models
    if spam == gut.UNSURE or abuse == gut.UNSURE:
        return "review", models
    return "publish", models


COMMENTS = [
    "Great write-up, the section on retries saved me an afternoon.",
    "Buy followers now!!! 10k for $5, link in bio",
    "You clearly have no idea what you're talking about, idiot.",
    "Has anyone tried this with Postgres 17?",
    "I made $3,000 last week working from home, ask me how",
]


def run(
    cascade: gut.Cascade, comments: list[str]
) -> tuple[list[tuple[str, str, set[str]]], Counter[str]]:
    gut.configure(backend=cascade)
    rows = [(comment, *moderate(comment)) for comment in comments]
    return rows, cascade.answered_by


def main() -> None:
    from _pick import backend_from_argv

    bigger = backend_from_argv(__doc__ or "", default="qwen")
    rows, answered_by = run(gut.Cascade(gut.ZeroShotBackend(), bigger), COMMENTS)
    for comment, action, models in rows:
        print(f"{action:<8} {comment[:60]:<60}  ({', '.join(sorted(models))})")
    print("\nanswers per model:", dict(answered_by))


if __name__ == "__main__":
    main()
