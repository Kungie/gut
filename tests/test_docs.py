"""The documentation has to be true.

Every Python block in the README and in `docs/` is **executed**, not just compiled. Documentation
that drifts from the code is worse than none, and this project has already shipped a README example
that could not compile, two whose cost models had an unreachable branch, a headline quoting a
threshold from a different cost model, and a check that quietly stopped asserting anything.

Illustrative blocks refer to names a reader is expected to supply -- `email`, `ticket`, `Team` --
so a prelude provides them and a `FakeBackend` answers whatever gets asked. A page runs top to
bottom in one namespace, the way it is read.
"""

from __future__ import annotations

import enum
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

import gut
from tests._markdown import ROOT, blocks, read

README = "README.md"
DOC_PAGES = sorted(str(page.relative_to(ROOT)) for page in (ROOT / "docs").glob("*.md"))
ALL_PAGES = [README, *DOC_PAGES]

PRESET_ROW = re.compile(
    r"^\|\s*\*\*(low|medium|high)\*\*\s*\|(.+?)\|(.+?)\|(.+?)\|\s*$", re.MULTILINE
)
BAND = re.compile("ask\\s+([\\d.]+)\\s*[\u2013-]\\s*([\\d.]+)")


class Team(enum.Enum):
    """The routing enum the docs use without always defining it."""

    BILLING = "invoices, charges, refunds"
    PLATFORM = "outages, latency, API errors"
    OTHER = "anything else"


def prelude() -> dict[str, Any]:
    """The names an illustrative snippet expects its reader to already have."""
    gut.configure(
        backend=gut.FakeBackend(rule=gut.deterministic_rule),
        cache=gut.NullCache(),
    )
    return {
        "email": "If this happens again I'm cancelling my subscription.",
        "ticket": {"subject": "502s", "body": "Nothing loads since 09:00."},
        "comment": "Check out my crypto newsletter!",
        "message": "The dashboard is down and my team is blocked.",
        "review": "The charger got hot enough to scorch the desk.",
        "agent_state": {"step": 4, "notes": "all checks passed"},
        "Team": Team,
        "escalate": lambda *_: None,
        "auto_reply": lambda *_: None,
        "send_to_a_person": lambda *_: None,
        "review_queue": type("Queue", (), {"add": staticmethod(lambda *_: None)})(),
        "route": lambda *_: None,
        "decision": gut.likely("a ticket", "is a bug report"),
    }


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Somewhere a snippet can write a cache or a log without touching the repository.

    A dummy key lets `JevBackend(...)` construct, which some snippets do; an unroutable base URL
    means that if one ever tried to *call* it, the test fails immediately instead of reaching the
    real API.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "documentation-test-not-a-real-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.delenv("GUT_RECORD", raising=False)

    # Files a reader following the docs would already have by the time they reach the snippet
    # that loads them.
    calibration = gut.CalibrationSet()
    calibration.add(
        gut.Entry(
            fingerprint=gut.NoulSpec("is a bug report").fingerprint,
            calibrator=gut.Isotonic(points=((0.0, 0.0), (1.0, 1.0))),
            model="fake-1.0",
        )
    )
    calibration.save(tmp_path / "calibration.json")
    return tmp_path


# --------------------------------------------------------------------------- every block runs


@pytest.mark.parametrize("page", ALL_PAGES)
def test_every_python_block_runs(page: str, sandbox: Path) -> None:
    found = blocks(page, "python")
    if not found:
        pytest.skip(f"{page} is prose")

    namespace = prelude()
    for line, source in found:
        # Re-seed before each block: a snippet that reconfigures gut must not break the next one.
        namespace.update(prelude())
        try:
            exec(compile(source, f"{page}:{line}", "exec"), namespace)
        except Exception as error:
            pytest.fail(f"{page}:{line} raised {type(error).__name__}: {error}\n\n{source}")


def test_the_pages_that_should_have_examples_do() -> None:
    """Prose-only pages are skipped above, so make sure the how-to pages did not go quiet."""
    for page in (
        README,
        "docs/getting-started.md",
        "docs/knowing-when-it-doesnt-know.md",
        "docs/batching.md",
        "docs/trusting-it.md",
        "docs/exact-costs.md",
        "docs/caching-and-logging.md",
    ):
        assert blocks(page, "python"), f"{page} lost all of its examples"


@pytest.mark.parametrize("page", ALL_PAGES)
def test_no_block_is_left_unlabelled(page: str) -> None:
    """An unlabelled fence is invisible to every check here, so there should not be prose in one."""
    for match in re.finditer(r"^```(\w*)\n(.*?)^```", read(page), re.MULTILINE | re.DOTALL):
        if match.group(1):
            continue
        body = match.group(2)
        assert "import gut" not in body, f"{page} has Python in an unlabelled fence"


def test_the_readme_stays_a_pitch() -> None:
    """Its job is thirty seconds, not completeness. Everything else belongs in docs/."""
    text = read(README)
    assert len(text.splitlines()) < 120

    for banned, where in (
        ("cost_false_yes", "docs/exact-costs.md"),
        ("on_unsure", "docs/knowing-when-it-doesnt-know.md"),
        ("GUT_RECORD", "docs/trusting-it.md"),
        ("SQLiteCache", "docs/caching-and-logging.md"),
    ):
        assert banned not in text, f"{banned} belongs in {where}, not the README"


def test_every_link_in_the_docs_resolves() -> None:
    for page in ALL_PAGES:
        base = (ROOT / page).parent
        for target in re.findall(r"\]\((?!https?:)([^)#]+)", read(page)):
            assert (base / target).exists(), f"{page} links to {target}, which is not there"


# --------------------------------------------------------------------------- the claims


def test_the_headline_result_matches_the_demo() -> None:
    """The README's one number has to be the one the demo actually produces."""
    tickets = json.loads((ROOT / "examples/support_tickets/tickets.json").read_text())
    threats = sum(1 for ticket in tickets if ticket["labels"]["cancel_threat"])

    text = read(README)
    assert f"**8 of {threats}**" in text
    assert "101 synthetic support tickets" in text
    # And the demo's own table is where that 8 comes from.
    demo = read("examples/support_tickets/README.md")
    assert "threshold=0.7               100%      0%          -         8" in demo


def test_the_readme_says_what_the_numbers_measure() -> None:
    """Labels written for this repository measure agreement with themselves."""
    assert "written for this repository" in read(README)
    assert "rather than real-world performance" in read(README)


def test_the_vendor_numbers_are_marked_as_the_vendors() -> None:
    page = read("docs/caching-and-logging.md")
    assert "self-reported vendor benchmarks" in page
    assert "40x-200x faster" in page
    assert "$0.042 / MTok" in page
    assert "on the higher end of real world gains" in page
    # The README links to that page rather than repeating the figures.
    assert "hundreds of times faster and\ncheaper" in read(README)
    assert "444" not in read(README)


def test_the_preset_table_matches_the_code() -> None:
    """Every posture in every user's code changes meaning if these drift."""
    from gut import presets

    published: dict[tuple[str, str | None], tuple[float, float]] = {}
    for row in PRESET_ROW.finditer(read("docs/knowing-when-it-doesnt-know.md")):
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

    document = "\n".join(read(page) for page in ALL_PAGES)
    triples = [
        [float(value) for value in re.findall(rf'{name}["\']?[=:]\s*([\d.]+)', document)]
        for name in ("cost_false_yes", "cost_false_no", "cost_human")
    ]
    assert len({len(found) for found in triples}) == 1, "costs are not quoted in complete sets"
    assert len(triples[0]) >= 2

    for yes, no, human in zip(*triples, strict=True):
        rule = policy(cost_false_yes=yes, cost_false_no=no, cost_human=human)
        assert rule.unsure_reachable, f"cost_human={human} can never fire against {yes}/{no}"


def test_the_documented_threshold_and_ceiling_are_real() -> None:
    from gut._rule import policy

    page = read("docs/exact-costs.md")
    assert "0.038" in page
    assert "`1.92`" in page

    rule = policy(cost_false_yes=2, cost_false_no=50)
    assert rule.implied_threshold == pytest.approx(0.038, abs=0.001)
    ceiling = rule.max_useful_cost_human
    assert ceiling is not None
    assert round(ceiling, 2) == 1.92


def test_the_bare_case_error_is_quoted_exactly() -> None:
    quoted = "SyntaxError: name capture 'YES' makes remaining patterns unreachable"
    assert quoted in read("docs/knowing-when-it-doesnt-know.md")

    source = (
        "YES = 'yes'\nNO = 'no'\n"
        "def f(x):\n    match x:\n        case YES: pass\n        case NO: pass\n"
    )
    with pytest.raises(SyntaxError, match="makes remaining patterns unreachable"):
        compile(source, "<bare-case>", "exec")


def test_the_docs_index_lists_every_page() -> None:
    index = read("docs/README.md")
    for page in DOC_PAGES:
        name = Path(page).name
        if name == "README.md":
            continue
        assert name in index, f"docs/README.md does not link to {name}"


def test_nothing_still_points_at_the_old_readme_sections() -> None:
    """The README lost its anchors when it became a pitch."""
    stale = ("README.md#knowing", "README.md#exact-costs", "README.md#batching")
    for page in [*ALL_PAGES, "skills/gut/SKILL.md", "llms.txt"]:
        text = read(page)
        for anchor in stale:
            assert anchor not in text, f"{page} links to a section that no longer exists"


def test_the_environment_is_left_alone() -> None:
    """The doc tests set a dummy key; it must not leak out of them."""
    assert os.environ.get("TYPESAFE_BASE_URL") != "http://127.0.0.1:1"
