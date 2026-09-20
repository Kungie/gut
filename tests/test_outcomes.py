"""Tests for the outcome constants, including the pattern-matching rules they exist to satisfy."""

from __future__ import annotations

import textwrap

import pytest

import gut
from gut import NO, UNSURE, YES, Outcome


def test_the_package_exports_the_outcomes() -> None:
    assert (gut.YES, gut.NO, gut.UNSURE) == (Outcome.YES, Outcome.NO, Outcome.UNSURE)
    assert (YES, NO, UNSURE) == (Outcome.YES, Outcome.NO, Outcome.UNSURE)


def test_rank_orders_them_by_rising_probability() -> None:
    assert [o.rank for o in (NO, UNSURE, YES)] == [0, 1, 2]


def test_str_is_the_bare_word() -> None:
    assert [str(o) for o in (YES, NO, UNSURE)] == ["yes", "no", "unsure"]


def test_bare_case_names_do_not_compile() -> None:
    """The spelling `case YES:` cannot work in Python, which is why `Outcome` is an enum (D7).

    A bare name in a `case` clause is a capture pattern, not a value pattern: it binds anything and
    makes every later clause unreachable. This test pins the reason down so nobody 'fixes' the
    documented dotted spelling back to the one that looks nicer and does not compile.
    """
    source = textwrap.dedent("""
        YES = "yes"
        NO = "no"
        def f(outcome):
            match outcome:
                case YES: return "yes"
                case NO: return "no"
    """)
    with pytest.raises(SyntaxError, match="makes remaining patterns unreachable"):
        compile(source, "<bare-case>", "exec")


@pytest.mark.parametrize("outcome", list(Outcome))
def test_dotted_value_patterns_match(outcome: Outcome) -> None:
    """Both documented spellings -- `gut.YES` and `Outcome.YES` -- are real value patterns."""

    def via_module(value: Outcome) -> str:
        match value:
            case gut.YES:
                return "yes"
            case gut.NO:
                return "no"
            case gut.UNSURE:
                return "unsure"

    def via_enum(value: Outcome) -> str:
        match value:
            case Outcome.YES:
                return "yes"
            case Outcome.NO:
                return "no"
            case Outcome.UNSURE:
                return "unsure"

    assert via_module(outcome) == outcome.value
    assert via_enum(outcome) == outcome.value
