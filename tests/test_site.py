"""Tests for decision-site identification."""

from __future__ import annotations

import pytest

from gut import _site
from gut._site import UNKNOWN_SITE, CallSite, caller_site, decision_id


def test_the_site_names_the_caller_not_the_library() -> None:
    site = caller_site()
    assert site.module == __name__
    assert site.function == "test_the_site_names_the_caller_not_the_library"
    assert site.filename.endswith("test_site.py")
    assert site.lineno > 0


def test_location_is_module_and_function() -> None:
    site = CallSite(module="app.handlers", function="handle", filename="/x/app.py", lineno=12)
    assert site.location == "app.handlers:handle"
    assert str(site) == "/x/app.py:12 in app.handlers:handle"


def test_ids_depend_on_the_question_and_the_place() -> None:
    here = CallSite(module="a", function="f", filename="a.py", lineno=1)
    there = CallSite(module="a", function="g", filename="a.py", lineno=2)

    assert decision_id("q1", here) == decision_id("q1", here)
    assert decision_id("q1", here) != decision_id("q2", here)
    assert decision_id("q1", here) != decision_id("q1", there)


def test_ids_ignore_the_line_the_call_sits_on() -> None:
    """Moving a call down the file must not reset its calibration history. See D11."""
    early = CallSite(module="a", function="f", filename="a.py", lineno=3)
    late = CallSite(module="a", function="f", filename="a.py", lineno=300)
    assert decision_id("q1", early) == decision_id("q1", late)


def test_a_stack_with_no_caller_still_yields_a_site(monkeypatch: pytest.MonkeyPatch) -> None:
    """If every frame looks internal the walk runs out, and an id must still be available."""
    monkeypatch.setattr(_site, "_is_internal", lambda frame: True)
    assert caller_site() is UNKNOWN_SITE
    assert decision_id("q1", UNKNOWN_SITE)
