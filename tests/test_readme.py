"""The README has to be true.

Documentation that drifts from the code is worse than no documentation, and this project has already
shipped one README example that could not compile (`case YES:`) and two whose cost models had an
unreachable branch. So every Python block in the README is compiled here, and the quickstart is
executed, assertions and all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)


def python_blocks() -> list[tuple[int, str]]:
    """Every fenced Python block in the README, with the line it starts on."""
    text = README.read_text(encoding="utf-8")
    return [
        (text[: match.start()].count("\n") + 2, match.group(1)) for match in BLOCK.finditer(text)
    ]


def test_the_readme_has_examples() -> None:
    assert len(python_blocks()) >= 8


@pytest.mark.parametrize(
    ("line", "source"), python_blocks(), ids=[f"L{line}" for line, _ in python_blocks()]
)
def test_every_example_compiles(line: int, source: str) -> None:
    """Catches the class of bug that shipped once already: a documented spelling Python rejects."""
    compile(source, f"README.md:{line}", "exec")


def test_the_quickstart_runs() -> None:
    """The offline quickstart must work exactly as written, assertions included."""
    quickstart = [
        source
        for _, source in python_blocks()
        if "gut.configure(backend=gut.FakeBackend" in source and "assert" in source
    ]
    assert len(quickstart) == 1, "the runnable quickstart block moved or was edited away"

    namespace: dict[str, object] = {}
    exec(compile(quickstart[0], "README.md:quickstart", "exec"), namespace)

    decision = namespace["decision"]
    assert getattr(decision, "p") == 0.83  # noqa: B009


def test_no_example_uses_an_unreachable_human_cost() -> None:
    """A documented cost model whose UNSURE branch can never fire teaches the wrong thing."""
    from gut._rule import policy

    pattern = re.compile(
        r"cost_false_yes=(?P<yes>[\d.]+),\s*\n?\s*cost_false_no=(?P<no>[\d.]+),\s*\n?\s*"
        r"cost_human=(?P<human>[\d.]+)",
        re.MULTILINE,
    )
    found = 0
    for match in pattern.finditer(README.read_text(encoding="utf-8")):
        found += 1
        rule = policy(
            cost_false_yes=float(match["yes"]),
            cost_false_no=float(match["no"]),
            cost_human=float(match["human"]),
        )
        assert rule.unsure_reachable, f"{match.group(0)!r} can never return UNSURE"
    assert found >= 2


def test_the_documented_threshold_is_the_real_one() -> None:
    """The README claims 2/52 is 0.038 with no human in the loop."""
    from gut._rule import policy

    assert "0.038" in README.read_text(encoding="utf-8")
    assert policy(cost_false_yes=2, cost_false_no=50).implied_threshold == pytest.approx(
        0.038, abs=0.001
    )


def test_the_documented_boundaries_are_the_real_ones() -> None:
    """The headline claims the 2/50/1 example auto-replies below 0.02 and escalates above 0.5.

    An earlier draft quoted `0.09` here, which belongs to a different cost model entirely. This
    test walks the actual rule rather than trusting the prose.
    """
    from gut._outcomes import Outcome
    from gut._rule import policy

    text = README.read_text(encoding="utf-8")
    assert "below `0.02`" in text
    assert "above `0.5`" in text

    rule = policy(cost_false_yes=2, cost_false_no=50, cost_human=1)
    assert rule.decide(0.019) is Outcome.NO
    assert rule.decide(0.021) is Outcome.UNSURE
    assert rule.decide(0.499) is Outcome.UNSURE
    assert rule.decide(0.501) is Outcome.YES


def test_the_documented_ceiling_formula_is_right() -> None:
    """The README says the ceiling for 2 and 50 is 1.92."""
    from gut._rule import policy

    assert "1.92" in README.read_text(encoding="utf-8")
    ceiling = policy(cost_false_yes=2, cost_false_no=50).max_useful_cost_human
    assert ceiling is not None
    assert round(ceiling, 2) == 1.92
