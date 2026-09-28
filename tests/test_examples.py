"""The examples run, and do what they say, with no model at all.

Each example is written against `gut` alone and picks its model from the command line, so here it
gets a `FakeBackend` with answers chosen to walk every branch.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest

import gut
from gut import FakeBackend
from tests._markdown import ROOT


@pytest.fixture
def examples(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    yield
    for name in (
        "triage",
        "moderation",
        "custom_backend",
        "commit_roast",
        "meeting_or_email",
        "recipe_or_memoir",
        "_pick",
    ):
        sys.modules.pop(name, None)


def load(name: str) -> ModuleType:
    return importlib.import_module(name)


def test_triage_routes_every_way_it_can(examples: None) -> None:
    triage = load("triage")

    def by_ticket(state: gut.State, name: str, spec: gut.QuestionSpec) -> object:
        text = str(state)
        if isinstance(spec, gut.NoulSpec):
            return 0.9 if "vendor" in text else (0.5 if "outage" in text else 0.05)
        if isinstance(spec, gut.ChoiceSpec):
            return "BILLING" if "charged" in text else "ENGINEERING"
        return 2 if "5pm" in text else 0

    backend = FakeBackend(rule=by_ticket)  # type: ignore[arg-type]
    gut.configure(backend=backend, cache=gut.NullCache())
    routes = dict(triage.run([*triage.TICKETS, "Minor outage earlier, all fine now."]))

    assert routes[triage.TICKETS[0]] == triage.Route("billing", 2, "")
    assert routes[triage.TICKETS[1]].priority == 0
    assert routes[triage.TICKETS[3]] == triage.Route("retention", 0, "said they might leave")
    assert routes["Minor outage earlier, all fine now."].note == "might be a churn risk"
    # One request per ticket: @semantic batched all three judgments.
    assert backend.call_count == len(routes)
    assert triage.triage.gut_plan.questions


def test_triage_sends_an_unsure_classification_to_the_front_desk(examples: None) -> None:
    triage = load("triage")
    unsure = gut.ChoiceAnswer(
        choice="OTHER",
        confidence=0.3,
        probabilities={"BILLING": 0.2, "ENGINEERING": 0.2, "ACCOUNT": 0.3, "OTHER": 0.3},
    )

    def vague(state: gut.State, name: str, spec: gut.QuestionSpec) -> object:
        if isinstance(spec, gut.ChoiceSpec):
            return unsure
        return 0.05 if isinstance(spec, gut.NoulSpec) else 0

    gut.configure(backend=FakeBackend(rule=vague), cache=gut.NullCache())  # type: ignore[arg-type]
    ((_, route),) = triage.run(["hello?"])
    assert route.queue == "front desk"


def test_moderation_hides_reviews_and_publishes(examples: None) -> None:
    moderation = load("moderation")
    local = FakeBackend(
        rule=lambda state, name, spec: (
            0.95 if "Buy" in str(state) else 0.5 if "$3,000" in str(state) else 0.02
        ),
        model="nli",
    )
    bigger = FakeBackend(rule=lambda state, name, spec: 0.5, model="bigger")
    rows, answered_by = moderation.run(gut.Cascade(local, bigger), moderation.COMMENTS)
    actions = {comment: action for comment, action, _ in rows}

    assert actions[moderation.COMMENTS[0]] == "publish"
    assert actions[moderation.COMMENTS[1]] == "hide"
    assert actions[moderation.COMMENTS[4]] == "review"
    # "nli" could settle neither question about the money comment, so both went on.
    assert answered_by == {"nli": 8, "bigger": 2}
    assert next(models for c, _, models in rows if "$3,000" in c) == {"bigger"}
    assert next(models for c, _, models in rows if "Postgres" in c) == {"nli"}


def test_a_keyword_backend_settles_what_it_knows_and_hands_on_the_rest(examples: None) -> None:
    custom = load("custom_backend")
    fallback = FakeBackend(default=0.9, model="fallback")
    rows, answered_by = custom.run(fallback, custom.TICKETS)

    first, _, third, fourth = rows
    assert first[1] == "refund=yes (keywords-1)"
    assert third[1] == "refund=yes (fallback)"
    assert fourth[2] == "bug=yes (fallback)"
    assert answered_by["keywords-1"] == 2
    assert isinstance(custom.KeywordBackend({}), gut.Backend)


def test_the_keyword_backend_says_nothing_about_other_shapes(examples: None) -> None:
    custom = load("custom_backend")
    backend = custom.KeywordBackend({})
    response = backend.ask(
        {"ticket": "x"},
        {
            "team": gut.ChoiceSpec(None, {"A": None, "B": None}),
            "urgency": gut.ScoreSpec(None, ["low", "high"]),
        },
    )
    team, urgency = response.answers["team"], response.answers["urgency"]
    assert isinstance(team, gut.ChoiceAnswer)
    assert team.confidence == 0.5
    assert isinstance(urgency, gut.ScoreAnswer)
    assert urgency.score == 0.5


@pytest.mark.parametrize("name", ["nli", "qwen", "ollama", "openai", "jev", "fake"])
def test_every_backend_an_example_offers_exists(examples: None, name: str) -> None:
    pick = load("_pick")
    assert name in pick.BACKENDS


def test_the_fake_choice_runs_an_example_end_to_end(
    examples: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    triage = load("triage")
    monkeypatch.setattr(sys, "argv", ["triage.py", "--backend", "fake"])
    triage.main()
    out = capsys.readouterr().out
    assert out.startswith("Answering with no model at all.")
    assert len(out.strip().splitlines()) == 2 + len(triage.TICKETS)


def test_commit_roast_sorts_commits_and_counts_words_itself(examples: None) -> None:
    roast = load("commit_roast")

    def kind(state: gut.State, name: str, spec: gut.QuestionSpec) -> object:
        return "FIX" if str(state).lower().startswith("fix") else "CHORE"

    backend = FakeBackend(rule=kind)  # type: ignore[arg-type]
    gut.configure(backend=backend, cache=gut.NullCache())
    graded = {item.message: item for item in roast.roast(roast.SAMPLE)}

    assert graded["fix"].vague
    assert graded["fix"].joke == roast.VAGUE
    long_fix = graded["Fix off-by-one in pagination that skipped the last page"]
    assert (long_fix.kind, long_fix.vague) == ("fix", False)
    assert long_fix.joke == roast.JOKES[roast.Kind.FIX]
    assert graded["Bump httpx to 0.28 so the new timeout API is available"].kind == "chore"
    assert backend.call_count == len(set(roast.SAMPLE))  # one each() call, one subject apiece


def test_commit_roast_reads_git_or_falls_back(
    examples: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    roast = load("commit_roast")
    assert roast.commits(limit=3)  # this repository has history
    monkeypatch.setattr(roast.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert roast.commits() == roast.SAMPLE


def test_meeting_or_email_gives_all_three_verdicts(examples: None) -> None:
    meeting = load("meeting_or_email")

    def judge(state: gut.State, name: str, spec: gut.QuestionSpec) -> object:
        text = str(state)
        return 0.9 if "launch" in text else 0.5 if "sync" in text else 0.05

    gut.configure(backend=FakeBackend(rule=judge), cache=gut.NullCache())  # type: ignore[arg-type]
    advice = [meeting.advise(invite) for invite in meeting.INVITES]
    assert advice[2].startswith("Go.")
    assert advice[0].startswith("Ask for an agenda.")
    assert advice[4].startswith("Email.")


def test_recipe_or_memoir_covers_every_combination(examples: None) -> None:
    recipes = load("recipe_or_memoir")
    pages = ["ingredients only", "ingredients and a story", "story only", "neither"]

    def judge(state: gut.State, name: str, spec: gut.QuestionSpec) -> object:
        text, claim = str(state), spec.instructions or ""
        wanted = "ingredients" if "ingredients" in claim else "story"
        return 0.97 if wanted in text else 0.02

    gut.configure(backend=FakeBackend(rule=judge), cache=gut.NullCache())  # type: ignore[arg-type]
    assert recipes.verdicts(pages) == [
        "Straight to the recipe. A rare and beautiful thing.",
        "Recipe found, after a life story. Scroll on.",
        "No recipe. Only a memoir.",
        "Neither a recipe nor a story. What is this page?",
    ]
