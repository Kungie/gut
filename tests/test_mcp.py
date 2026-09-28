"""Tests for the MCP server: gut's judgments as tools an agent can call."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("mcp", reason='the MCP server needs the extra: "gutfeel[mcp]"')

from mcp import Client, StdioServerParameters
from mcp.types import CallToolResult

import gut
from gut import ConfigurationError, FakeBackend
from gut._mcp import BACKENDS, OLLAMA_URL, backend_from_env, build, options_enum

pytestmark = pytest.mark.anyio


def by_text(state: gut.State, name: str, spec: gut.QuestionSpec) -> float | str | int | None:
    match spec:
        case gut.NoulSpec():
            return 0.97 if "WIN" in str(state) else 0.5 if "maybe" in str(state) else 0.02
        case gut.ChoiceSpec():
            return "billing" if "invoice" in str(state) else "other"
        case _:
            return 2 if "down" in str(state) else 0


def backend() -> FakeBackend:
    return FakeBackend(rule=by_text)


async def call(name: str, arguments: dict[str, Any], **server: Any) -> CallToolResult:
    async with Client(build(server.get("backend", backend()))) as client:
        return await client.call_tool(name, arguments)


def structured(result: CallToolResult) -> dict[str, Any]:
    assert not result.is_error, result.content
    content: dict[str, Any] | None = result.structured_content
    assert content is not None
    return content


def error_text(result: CallToolResult) -> str:
    assert result.is_error
    text = result.content[0]
    assert text.type == "text"
    return text.text


# --------------------------------------------------------------------------- the tools


async def test_the_server_offers_four_read_only_tools() -> None:
    async with Client(build(backend())) as client:
        listed = await client.list_tools()
        info = client.server_info
        instructions = client.instructions
    assert [tool.name for tool in listed.tools] == ["likely", "classify", "rate", "each"]
    for tool in listed.tools:
        assert tool.description
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is not True
    assert info is not None
    assert info.name == "gut"
    assert info.version == gut.__version__
    assert instructions is not None
    assert "unsure" in instructions


async def test_arguments_carry_descriptions_the_agent_can_read() -> None:
    async with Client(build(backend())) as client:
        listed = await client.list_tools()
    likely = next(tool for tool in listed.tools if tool.name == "likely")
    properties = likely.input_schema["properties"]
    assert likely.input_schema["required"] == ["subject", "question"]
    assert "claim" in properties["question"]["description"]
    assert properties["stakes"]["anyOf"][0]["enum"] == ["low", "medium", "high"]


async def test_likely_answers_with_the_outcome_and_the_probability() -> None:
    answer = structured(await call("likely", {"subject": "WIN a prize", "question": "is spam"}))
    assert answer["outcome"] == "yes"
    assert answer["p"] == pytest.approx(0.97)
    assert answer["model"] == "fake-1.0"
    assert answer["question"] == "is spam"
    assert "id" not in answer
    assert "source" not in answer


async def test_likely_says_unsure_only_when_asked_to() -> None:
    plain = {"subject": "maybe", "question": "is spam"}
    assert structured(await call("likely", plain))["outcome"] in ("yes", "no")
    unsure = structured(await call("likely", plain | {"ask_human": True, "stakes": "high"}))
    assert unsure["outcome"] == "unsure"


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("likely", {"subject": "x", "question": "is spam"}),
        ("classify", {"subject": "x", "options": ["billing", "other"]}),
        ("rate", {"subject": "x", "levels": ["low", "high"]}),
        ("each", {"subjects": ["x"], "question": "is spam"}),
    ],
)
async def test_stakes_without_ask_human_is_an_error_not_a_silent_warning(
    tool: str, arguments: dict[str, Any]
) -> None:
    """A program gets a warning; an agent would never see one."""
    assert "ask_human=true" in error_text(await call(tool, arguments | {"stakes": "high"}))


async def test_classify_takes_labels_with_descriptions() -> None:
    options = {"billing": "invoices and refunds", "other": "anything else"}
    answer = structured(await call("classify", {"subject": "my invoice", "options": options}))
    assert answer["value"] == "billing"
    assert set(answer["probabilities"]) == {"billing", "other"}
    assert "note" not in answer


async def test_classify_takes_bare_labels_and_notes_a_missing_catch_all() -> None:
    options = ["billing", "platform"]
    answer = structured(await call("classify", {"subject": "my invoice", "options": options}))
    assert answer["value"] == "billing"
    assert "catch-all" in answer["note"]


async def test_rate_answers_with_the_nearest_level() -> None:
    levels = ["can wait", "today", "right now"]
    answer = structured(await call("rate", {"subject": "the site is down", "levels": levels}))
    assert answer["score"] == pytest.approx(2)
    assert answer["level"] == "right now"


async def test_a_rubric_of_one_level_is_an_error_the_agent_reads() -> None:
    result = await call("rate", {"subject": "x", "levels": ["only"]})
    assert "between 2 and 10" in error_text(result)


async def test_each_asks_likely_classify_or_rate_about_many_subjects() -> None:
    fake = backend()
    subjects = ["WIN now", "hello", "WIN again"]
    spam = structured(
        await call("each", {"subjects": subjects, "question": "is spam"}, backend=fake)
    )
    assert [r["outcome"] for r in spam["results"]] == ["yes", "no", "yes"]

    teams = structured(
        await call("each", {"subjects": ["invoice", "hi"], "options": ["billing", "other"]})
    )
    assert [r["value"] for r in teams["results"]] == ["billing", "other"]
    assert "note" not in teams

    levels = ["can wait", "today", "now"]
    urgency = structured(await call("each", {"subjects": ["down", "fine"], "levels": levels}))
    assert [r["level"] for r in urgency["results"]] == ["now", "can wait"]


async def test_each_notes_a_missing_catch_all() -> None:
    result = await call("each", {"subjects": ["invoice"], "options": ["billing", "platform"]})
    assert "catch-all" in structured(result)["note"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"subjects": ["a"]}, "a question, options or levels"),
        ({"subjects": ["a"], "options": ["x", "other"], "levels": ["a", "b"]}, "not both"),
        ({"subjects": ["a"] * 1001, "question": "is spam"}, "At most 1000"),
    ],
)
async def test_each_says_what_it_needs(arguments: dict[str, Any], message: str) -> None:
    assert message in error_text(await call("each", arguments))


# --------------------------------------------------------------------------- the options


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["only"], "at least two"),
        (["a", "a"], "different"),
        (["_private", "other"], "must not start with '_'"),
        (["", "other"], "non-empty"),
        ({"a": "same", "b": "same"}, "same description"),
    ],
)
def test_options_that_would_not_make_an_honest_enum_are_refused(
    options: Mapping[str, str] | list[str], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        options_enum(options)


def test_an_empty_description_falls_back_to_the_label() -> None:
    enum_class = options_enum({"billing": "", "other": "anything else"})
    assert [(m.name, m.value) for m in enum_class] == [
        ("billing", "billing"),
        ("other", "anything else"),
    ]


# --------------------------------------------------------------------------- the backend


def test_no_backend_named_leaves_the_choice_to_gut() -> None:
    assert backend_from_env({}) is None


def test_an_unknown_backend_is_refused_with_the_known_ones() -> None:
    with pytest.raises(ConfigurationError, match="ollama"):
        backend_from_env({"GUT_BACKEND": "gpt"})


def test_the_fake_backend_answers_anything() -> None:
    fake = backend_from_env({"GUT_BACKEND": "FAKE"})
    assert isinstance(fake, FakeBackend)


@pytest.mark.parametrize("name", ["openai", "ollama"])
def test_a_server_backend_needs_a_model(name: str) -> None:
    with pytest.raises(ConfigurationError, match="GUT_MODEL"):
        backend_from_env({"GUT_BACKEND": name})


def test_ollama_defaults_to_its_local_url() -> None:
    chosen = backend_from_env({"GUT_BACKEND": "ollama", "GUT_MODEL": "qwen3:0.6b"})
    assert isinstance(chosen, gut.OpenAICompatibleBackend)
    assert chosen.base_url == OLLAMA_URL
    assert chosen.model_id == "qwen3:0.6b"


def test_openai_takes_a_base_url() -> None:
    env = {"GUT_BACKEND": "openai", "GUT_MODEL": "m", "GUT_BASE_URL": "http://gpu.local:8000/v1/"}
    chosen = backend_from_env(env)
    assert isinstance(chosen, gut.OpenAICompatibleBackend)
    assert chosen.base_url == "http://gpu.local:8000/v1"


def test_jev_is_built_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("typesafe_sdk")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    chosen = backend_from_env({"GUT_BACKEND": "jev"})
    assert isinstance(chosen, gut.JevBackend)


def test_every_backend_named_in_the_docs_is_known() -> None:
    from pathlib import Path

    chapter = (Path(__file__).parent.parent / "docs" / "mcp.md").read_text(encoding="utf-8")
    for name in BACKENDS:
        assert f"`{name}`" in chapter


async def test_without_a_model_the_agent_is_told_how_to_configure_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("GUT_BACKEND", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    async with Client(build()) as client:
        result = await client.call_tool("likely", {"subject": "x", "question": "is spam"})
    text = error_text(result)
    assert "TYPESAFE_API_KEY" in text
    assert "GUT_BACKEND" in text


async def test_the_backend_is_built_once_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUT_BACKEND", "fake")
    async with Client(build()) as client:
        first = await client.call_tool("likely", {"subject": "x", "question": "is spam"})
        second = await client.call_tool("likely", {"subject": "y", "question": "is spam"})
    assert structured(first)["model"] == structured(second)["model"] == "fake-1.0"


async def test_a_misnamed_backend_is_reported_on_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUT_BACKEND", "nope")
    async with Client(build()) as client:
        result = await client.call_tool("likely", {"subject": "x", "question": "is spam"})
    assert "GUT_BACKEND='nope'" in error_text(result)


# --------------------------------------------------------------------------- over stdio


async def test_the_server_runs_over_stdio() -> None:
    """The real thing: a subprocess speaking MCP on stdin and stdout, as a client launches it."""
    env = {key: value for key, value in os.environ.items() if key != "TYPESAFE_API_KEY"}
    env["GUT_BACKEND"] = "fake"
    parameters = StdioServerParameters(command=sys.executable, args=["-m", "gut._mcp"], env=env)
    async with Client(parameters) as client:
        listed = await client.list_tools()
        result = await client.call_tool("likely", {"subject": "x", "question": "is spam"})
    assert len(listed.tools) == 4
    assert structured(result)["model"] == "fake-1.0"
