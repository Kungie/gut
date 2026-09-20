"""The agent skill and the machine-readable index have to be true.

These are the two documents most likely to be read by something that will not notice a mistake and
argue about it, so they are checked harder than prose usually is: every example compiles, every
name exists, and every value the skill says is accepted really is.
"""

from __future__ import annotations

import pytest
import yaml

import gut
from tests._markdown import ROOT, blocks, read

SKILL = "skills/gut/SKILL.md"
INDEX = "llms.txt"


def test_the_skill_has_frontmatter_an_agent_can_route_on() -> None:
    text = read(SKILL)
    assert text.startswith("---\n")
    frontmatter = yaml.safe_load(text.split("---", 2)[1])

    assert frontmatter["name"] == "gut"
    description = frontmatter["description"]
    # A description is a routing decision: it must say when to reach for this, not just what
    # it is.
    assert "Use when" in description
    assert len(description) < 500


@pytest.mark.parametrize(
    ("line", "source"), blocks(SKILL, "python"), ids=[f"L{n}" for n, _ in blocks(SKILL, "python")]
)
def test_every_skill_example_compiles(line: int, source: str) -> None:
    compile(source, f"{SKILL}:{line}", "exec")


def test_the_skill_names_only_real_api() -> None:
    text = read(SKILL)
    for name in (
        "likely",
        "classify",
        "rate",
        "semantic",
        "judge",
        "configure",
        "FakeBackend",
        "UnsureDecision",
        "PolicyError",
        "Outcome",
    ):
        assert name in text, f"the skill stopped mentioning {name}"
        assert name in gut.__all__, f"the skill names {name}, which gut no longer exports"


def test_the_skill_lists_the_values_that_are_actually_accepted() -> None:
    """`stakes="low"|"medium"|"high"` has to be exactly what the code takes."""
    from gut._posture import LEAN_THRESHOLD, STAKES_CERTAINTY

    text = read(SKILL)
    for stakes in STAKES_CERTAINTY:
        assert f'"{stakes}"' in text
    for lean in LEAN_THRESHOLD:
        if lean is not None:
            assert f'lean="{lean}"' in text or f'"{lean}"' in text


def test_the_skill_quotes_the_bare_case_error_exactly() -> None:
    quoted = "SyntaxError: name capture 'YES' makes remaining patterns unreachable"
    assert quoted in read(SKILL)
    with pytest.raises(SyntaxError, match="makes remaining patterns unreachable"):
        compile(
            "YES = 1\nNO = 2\n"
            "def f(x):\n    match x:\n        case YES: pass\n        case NO: pass\n",
            "<bare>",
            "exec",
        )


def test_the_skill_states_the_ceiling_formula_correctly() -> None:
    from gut._rule import policy

    assert "cost_false_yes * cost_false_no / (cost_false_yes + cost_false_no)" in read(SKILL)
    rule = policy(cost_false_yes=2, cost_false_no=50)
    assert rule.max_useful_cost_human == pytest.approx(2 * 50 / (2 + 50))


def test_the_skills_predicate_file_is_a_valid_one(tmp_path: pytest.TempPathFactory) -> None:
    from gut._evals import load_suite

    (yaml_block,) = [source for _, source in blocks(SKILL, "yaml")]
    path = ROOT / "skills" / "gut" / "_example.yaml"
    try:
        path.write_text(yaml_block, encoding="utf-8")
        suite = load_suite(path)
        assert suite.kind == "noul"
        assert len(suite.examples) == 2
        assert suite.min_accuracy == 0.9
    finally:
        path.unlink(missing_ok=True)


def test_the_skills_offline_setup_works() -> None:
    """The one snippet a reader will paste before anything else."""
    setup = [source for _, source in blocks(SKILL, "python") if "FakeBackend" in source]
    assert len(setup) == 1
    exec(compile(setup[0], f"{SKILL}:fake", "exec"), {})

    assert gut.likely("a ticket", "is a bug report").p == 0.91


def test_the_skill_forbids_the_two_things_that_raise() -> None:
    """Both of these are PolicyError at runtime, so the skill has to say so."""
    text = read(SKILL)
    assert "Never mix these" in text
    assert "PolicyError" in text

    gut.configure(backend=gut.FakeBackend(default=0.5))
    with pytest.raises(gut.PolicyError):
        gut.likely("t", "q", stakes="high", ask_human=True, cost_false_yes=1, cost_false_no=1)


# --------------------------------------------------------------------------- the index


def test_every_document_the_index_points_at_exists() -> None:
    import re

    for target in re.findall(r"\]\(([^)]+)\)", read(INDEX)):
        assert (ROOT / target).exists(), f"llms.txt points at {target}, which is not there"


def test_the_index_names_the_commands_that_exist() -> None:
    from gut._cli import COMMANDS, build_parser

    text = read(INDEX)
    parser = build_parser()
    for command in COMMANDS:
        # Each name is a real subcommand, and the index mentions it.
        assert parser.parse_args([command]).handler is not None
        assert f"gut {command}" in text, f"llms.txt does not mention `gut {command}`"


def test_the_index_traps_match_the_code() -> None:
    text = read(INDEX)
    assert "must be dotted" in text
    assert "ask_human=True" in text
    assert "cannot be mixed" in text
    assert "cfy * cfn / (cfy + cfn)" in text
