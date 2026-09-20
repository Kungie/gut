"""The README has to be true.

Documentation that drifts from the code is worse than none, and this project has already shipped a
README example that could not compile (`case YES:`), two whose cost models had an unreachable
branch, and a headline quoting a threshold from a different cost model. So every Python block here
is compiled, the quickstart is executed with its assertions intact, and every number the page
claims is checked against the code that produces it.
"""

from __future__ import annotations

import re

import pytest

from tests._markdown import blocks, read

README = "README.md"
PRESET_ROW = re.compile(
    r"^\|\s*\*\*(low|medium|high)\*\*\s*\|(.+?)\|(.+?)\|(.+?)\|\s*$", re.MULTILINE
)
BAND = re.compile("ask\\s+([\\d.]+)\\s*[\u2013-]\\s*([\\d.]+)")


def text() -> str:
    return read(README)


def python_blocks() -> list[tuple[int, str]]:
    """Every fenced Python block, with the line it starts on."""
    return blocks(README, "python")


def test_the_readme_has_examples() -> None:
    assert len(python_blocks()) >= 10


@pytest.mark.parametrize(
    ("line", "source"), python_blocks(), ids=[f"L{line}" for line, _ in python_blocks()]
)
def test_every_example_compiles(line: int, source: str) -> None:
    """Catches the class of bug that shipped once: a documented spelling Python rejects."""
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
    assert getattr(namespace["decision"], "p") == 0.83  # noqa: B009


# --------------------------------------------------------------------------- the numbers


def test_the_preset_table_matches_the_code() -> None:
    """The published bands are what `gut.presets()` actually produces.

    Every posture in every user's code changes meaning if these drift, so the table is checked
    rather than trusted.
    """
    from gut import presets

    published: dict[tuple[str, str | None], tuple[float, float]] = {}
    for row in PRESET_ROW.finditer(text()):
        stakes = row.group(1)
        for lean, cell in zip((None, "yes", "no"), row.groups()[1:], strict=True):
            match = BAND.search(cell)
            assert match is not None, f"unreadable band for {stakes}/{lean}: {cell!r}"
            published[stakes, lean] = (float(match.group(1)), float(match.group(2)))

    assert len(published) == 9, "the preset table lost a row or a column"
    for preset in presets():
        assert preset.unsure_band == pytest.approx(published[preset.stakes, preset.lean])


def test_no_documented_cost_model_has_an_unreachable_human() -> None:
    """A documented cost model whose UNSURE branch can never fire teaches the wrong thing."""
    from gut._rule import policy

    document = text()
    triples = [
        [float(value) for value in re.findall(rf'{name}["\']?[=:]\s*([\d.]+)', document)]
        for name in ("cost_false_yes", "cost_false_no", "cost_human")
    ]
    assert len({len(found) for found in triples}) == 1, "costs are not quoted in complete sets"
    assert len(triples[0]) >= 2

    for yes, no, human in zip(*triples, strict=True):
        rule = policy(cost_false_yes=yes, cost_false_no=no, cost_human=human)
        assert rule.unsure_reachable, f"cost_human={human} can never fire against {yes}/{no}"


def test_the_documented_threshold_is_the_real_one() -> None:
    """The README says 2 and 50 work out to 0.038 with no human in the loop."""
    from gut._rule import policy

    assert "0.038" in text()
    assert policy(cost_false_yes=2, cost_false_no=50).implied_threshold == pytest.approx(
        0.038, abs=0.001
    )


def test_the_documented_ceiling_is_the_real_one() -> None:
    """The README says the ceiling for 2 and 50 is 1.92."""
    from gut._rule import policy

    assert "`1.92`" in text()
    ceiling = policy(cost_false_yes=2, cost_false_no=50).max_useful_cost_human
    assert ceiling is not None
    assert round(ceiling, 2) == 1.92


def test_the_bare_case_syntax_error_is_quoted_correctly() -> None:
    """The README quotes the exact message; a paraphrase would be worse than nothing."""
    quoted = "SyntaxError: name capture 'YES' makes remaining patterns unreachable"
    assert quoted in text()

    source = (
        "YES = 'yes'\nNO = 'no'\n"
        "def f(x):\n    match x:\n        case YES: pass\n        case NO: pass\n"
    )
    with pytest.raises(SyntaxError, match="makes remaining patterns unreachable"):
        compile(source, "<bare-case>", "exec")
