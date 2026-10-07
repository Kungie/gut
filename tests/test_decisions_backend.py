"""Tests for `OpenAIDecisionsBackend`, against a fake server rather than a real one.

The fake answers the way OpenAI's guide to the Decisions API says the real one does. What is checked
here is that `gut` asks it the right thing and reads its answer correctly.
"""

from __future__ import annotations

import asyncio
import enum
import json
import threading
from collections.abc import Callable
from typing import Any

import httpx
import pytest

import gut
from gut import BackendError, OpenAIDecisionsBackend
from gut._env import backend_from_env
from gut._questions import ChoiceSpec, NoulSpec, ScoreSpec


class Team(enum.Enum):
    BILLING = "invoices and refunds"
    PLATFORM = "outages"
    OTHER = "anything else"


TEAM = ChoiceSpec(instructions=None, criteria={member.name: member.value for member in Team})
URGENCY = ScoreSpec(instructions=None, criteria=["can wait", "this week", "right now"])


def answer_to(question: dict[str, Any]) -> dict[str, Any]:
    """One answer the way the Decisions API shapes it, for whatever the question asks."""
    name = question["name"]
    if question["type"] == "predicate":
        return {"type": "predicate", "name": name, "probability": 0.92}
    if question["type"] == "choice":
        values = [choice["value"] for choice in question["choices"]]
        rest = 0.1 / (len(values) - 1)
        return {
            "type": "choice",
            "name": name,
            "choice": values[1],
            "probabilities": [
                {"value": value, "probability": 0.9 if index == 1 else rest}
                for index, value in enumerate(values)
            ],
            "confidence": 0.85,
        }
    return {
        "type": "score",
        "name": name,
        "score": 1.1,
        "probabilities": [
            {"value": index, "label": level["label"], "probability": p}
            for index, (level, p) in enumerate(
                zip(question["levels"], (0.1, 0.7, 0.2), strict=False)
            )
        ],
        "confidence": 0.55,
    }


class FakeServer:
    """Answers each request from its questions, and remembers every request."""

    def __init__(self, reply: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self._reply = reply or self.default
        self._lock = threading.Lock()

    @staticmethod
    def default(body: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "dec_123",
            "model": "gpt-6-luna-2026-09-22",
            "answers": [answer_to(question) for question in body["questions"]],
            "usage": {"input_tokens": 57, "output_tokens": 0},
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        with self._lock:
            self.requests.append(request)
            self.bodies.append(body)
        return httpx.Response(200, json=self._reply(body))

    def backend(self, **options: Any) -> OpenAIDecisionsBackend:
        client = httpx.Client(transport=httpx.MockTransport(self))
        return OpenAIDecisionsBackend(client=client, **options)


def replying(payload: object, status: int = 200) -> OpenAIDecisionsBackend:
    """A backend whose server sends `payload` whatever it is asked."""

    def respond(request: httpx.Request) -> httpx.Response:
        if isinstance(payload, str):
            return httpx.Response(status, text=payload)
        return httpx.Response(status, json=payload)

    return OpenAIDecisionsBackend(client=httpx.Client(transport=httpx.MockTransport(respond)))


@pytest.fixture(autouse=True)
def _no_ambient_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


# --------------------------------------------------------------------------- what is asked


def test_a_yes_no_question_is_one_predicate() -> None:
    server = FakeServer()
    response = server.backend().ask("the ticket", {"q": NoulSpec("is a bug report")})

    answer = response.answers["q"]
    assert isinstance(answer, gut.NoulAnswer)
    assert answer.p == pytest.approx(0.92)
    assert response.model == "gpt-6-luna-2026-09-22"
    assert response.input_tokens == 57
    assert response.models is None
    (body,) = server.bodies
    assert body == {
        "model": "gpt-6-luna",
        "input": "the ticket",
        "questions": [{"type": "predicate", "name": "q", "instructions": "is a bug report"}],
    }
    assert str(server.requests[0].url) == "https://api.openai.com/v1/decisions"


def test_what_counts_as_yes_and_no_goes_into_the_instructions() -> None:
    server = FakeServer()
    spec = NoulSpec("is urgent", yes_means="someone is blocked", no_means="it can wait a day")
    server.backend().ask("the ticket", {"q": spec})
    assert server.bodies[0]["questions"][0]["instructions"] == (
        "is urgent\n"
        "It counts as true if: someone is blocked\n"
        "It counts as false if: it can wait a day"
    )


def test_a_choice_sends_its_options_and_reads_the_distribution() -> None:
    server = FakeServer()
    response = server.backend().ask("the ticket", {"team": TEAM})

    answer = response.answers["team"]
    assert isinstance(answer, gut.ChoiceAnswer)
    assert answer.choice == "PLATFORM"
    assert answer.confidence == pytest.approx(0.85)
    assert answer.probabilities == pytest.approx({"BILLING": 0.05, "PLATFORM": 0.9, "OTHER": 0.05})
    assert server.bodies[0]["questions"] == [
        {
            "type": "choice",
            "name": "team",
            "instructions": "Which option fits the input best?",
            "choices": [
                {"value": "BILLING", "description": "invoices and refunds"},
                {"value": "PLATFORM", "description": "outages"},
                {"value": "OTHER", "description": "anything else"},
            ],
        }
    ]


def test_an_option_that_speaks_for_itself_is_sent_without_a_description() -> None:
    server = FakeServer()
    spec = ChoiceSpec(instructions="Who should read this?", criteria={"legal": None, "sales": "x"})
    server.backend().ask("the ticket", {"q": spec})
    (question,) = server.bodies[0]["questions"]
    assert question["instructions"] == "Who should read this?"
    assert question["choices"] == [{"value": "legal"}, {"value": "sales", "description": "x"}]


def test_a_rating_sends_its_levels_and_reads_the_score() -> None:
    server = FakeServer()
    response = server.backend().ask("the ticket", {"urgency": URGENCY})

    answer = response.answers["urgency"]
    assert isinstance(answer, gut.ScoreAnswer)
    assert answer.score == pytest.approx(1.1)
    assert answer.confidence == pytest.approx(0.55)
    assert answer.probabilities == pytest.approx({0: 0.1, 1: 0.7, 2: 0.2})
    assert answer.legend == {0: "can wait", 1: "this week", 2: "right now"}
    assert server.bodies[0]["questions"] == [
        {
            "type": "score",
            "name": "urgency",
            "instructions": "Where does the input fall on this scale?",
            "levels": [{"label": "can wait"}, {"label": "this week"}, {"label": "right now"}],
        }
    ]


def test_a_rating_keeps_a_question_of_its_own() -> None:
    server = FakeServer()
    spec = ScoreSpec(instructions="How angry is the customer?", criteria=["calm", "furious"])
    server.backend().ask("the ticket", {"q": spec})
    assert server.bodies[0]["questions"][0]["instructions"] == "How angry is the customer?"


def test_every_question_about_a_subject_rides_in_one_request() -> None:
    def backwards(body: dict[str, Any]) -> dict[str, Any]:
        reply = FakeServer.default(body)
        reply["answers"].reverse()  # answers are matched by name, not by position
        return reply

    server = FakeServer(backwards)
    response = server.backend().ask(
        "the ticket", {"bug": NoulSpec("is a bug report"), "team": TEAM, "urgency": URGENCY}
    )
    assert len(server.requests) == 1
    assert [question["name"] for question in server.bodies[0]["questions"]] == [
        "bug",
        "team",
        "urgency",
    ]
    assert isinstance(response.answers["bug"], gut.NoulAnswer)
    assert isinstance(response.answers["team"], gut.ChoiceAnswer)
    assert isinstance(response.answers["urgency"], gut.ScoreAnswer)


def test_a_subject_that_is_not_text_is_sent_as_json() -> None:
    server = FakeServer()
    server.backend().ask({"subject": "Refund", "body": "çift ödeme"}, {"q": NoulSpec("q")})
    assert json.loads(server.bodies[0]["input"]) == {"subject": "Refund", "body": "çift ödeme"}
    assert "çift ödeme" in server.bodies[0]["input"]


def test_extra_parameters_ride_along_with_every_request() -> None:
    server = FakeServer()
    server.backend(extra_body={"safety_identifier": "user-1"}).ask("t", {"q": NoulSpec("q")})
    assert server.bodies[0]["safety_identifier"] == "user-1"


def test_another_model_can_be_named() -> None:
    server = FakeServer()
    backend = server.backend(model="gpt-6-luna-2026-09-22")
    assert backend.model_id == "gpt-6-luna-2026-09-22"
    backend.ask("t", {"q": NoulSpec("q")})
    assert server.bodies[0]["model"] == "gpt-6-luna-2026-09-22"


# --------------------------------------------------------------------------- what comes back


def test_a_server_that_names_no_model_is_reported_as_the_one_asked_for() -> None:
    backend = replying({"answers": [{"type": "predicate", "name": "q", "probability": 0.3}]})
    response = backend.ask("t", {"q": NoulSpec("q")})
    assert response.model == "gpt-6-luna"
    assert response.input_tokens is None


def test_cost_is_unknown_unless_the_server_reports_it() -> None:
    assert FakeServer().backend().ask("t", {"q": NoulSpec("q")}).cost is None

    def priced(body: dict[str, Any]) -> dict[str, Any]:
        reply = FakeServer.default(body)
        reply["usage"]["cost"] = 0.0000057
        return reply

    response = FakeServer(priced).backend().ask("t", {"q": NoulSpec("q")})
    assert response.cost == pytest.approx(0.0000057)


def test_a_refusal_is_an_error_not_a_guess() -> None:
    backend = replying({"answers": [{"type": "refusal", "name": "q"}]})
    with pytest.raises(BackendError, match=r"gpt-6-luna at .* declined to answer 'q'"):
        backend.ask("t", {"q": NoulSpec("q")})


def test_an_answer_of_the_wrong_kind_is_an_error() -> None:
    backend = replying({"answers": [{"type": "score", "name": "q", "score": 1.0}]})
    with pytest.raises(BackendError, match="'q', a predicate question, with a 'score' answer"):
        backend.ask("t", {"q": NoulSpec("q")})


def test_a_missing_answer_is_an_error() -> None:
    backend = replying({"answers": [{"type": "predicate", "name": "other", "probability": 0.5}]})
    with pytest.raises(BackendError, match="answered without 'q'"):
        backend.ask("t", {"q": NoulSpec("q")})


@pytest.mark.parametrize(
    ("spec", "answer"),
    [
        (NoulSpec("q"), {"type": "predicate", "name": "q"}),
        (NoulSpec("q"), {"type": "predicate", "name": "q", "probability": "likely"}),
        (TEAM, {"type": "choice", "name": "q", "choice": "OTHER", "confidence": 0.5}),
        (TEAM, {"type": "choice", "name": "q", "choice": "OTHER", "probabilities": 3}),
        (URGENCY, {"type": "score", "name": "q", "score": 1.0, "confidence": 0.5}),
    ],
)
def test_an_answer_missing_its_numbers_is_an_error(
    spec: gut.QuestionSpec, answer: dict[str, Any]
) -> None:
    with pytest.raises(BackendError, match="answer for 'q' that could not be read"):
        replying({"answers": [answer]}).ask("t", {"q": spec})


@pytest.mark.parametrize("payload", ["not json", {"model": "m"}, {"answers": [3]}, [1, 2]])
def test_a_reply_that_is_not_a_decision_is_an_error(payload: object) -> None:
    with pytest.raises(BackendError, match="sent a reply with no answers in it"):
        replying(payload).ask("t", {"q": NoulSpec("q")})


def test_an_error_status_is_reported_with_the_servers_explanation() -> None:
    backend = replying({"error": {"message": "Incorrect API key provided"}}, status=401)
    with pytest.raises(BackendError, match=r"answered 401: .*Incorrect API key provided"):
        backend.ask("t", {"q": NoulSpec("q")})


def test_a_server_that_cannot_be_reached_is_a_backend_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = OpenAIDecisionsBackend(
        base_url="http://localhost:9000/v1",
        client=httpx.Client(transport=httpx.MockTransport(refuse)),
    )
    with pytest.raises(BackendError, match=r"gpt-6-luna at http://localhost:9000/v1 could not"):
        backend.ask("t", {"q": NoulSpec("q")})


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(BackendError, match="at least one question"):
        FakeServer().backend().ask("t", {})


def test_an_empty_model_name_is_refused() -> None:
    with pytest.raises(BackendError, match="model must be a non-empty string"):
        OpenAIDecisionsBackend(" ")


# --------------------------------------------------------------------------- where, and whose key


def test_openais_key_is_sent_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    server = FakeServer()
    server.backend().ask("t", {"q": NoulSpec("q")})
    assert server.requests[0].headers["authorization"] == "Bearer sk-test-not-real"


def test_openais_key_is_not_sent_anywhere_else(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    server = FakeServer()
    server.backend(base_url="https://gateway.example/v1/").ask("t", {"q": NoulSpec("q")})
    assert "authorization" not in server.requests[0].headers
    assert str(server.requests[0].url) == "https://gateway.example/v1/decisions"


def test_an_explicit_key_is_sent_wherever_it_is_pointed() -> None:
    server = FakeServer()
    server.backend(base_url="https://gateway.example/v1", api_key="gw-key").ask(
        "t", {"q": NoulSpec("q")}
    )
    assert server.requests[0].headers["authorization"] == "Bearer gw-key"


def test_the_base_url_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    server = FakeServer()
    server.backend().ask("t", {"q": NoulSpec("q")})
    assert str(server.requests[0].url) == "https://proxy.example/v1/decisions"
    assert server.requests[0].headers["authorization"] == "Bearer sk-test-not-real"


def test_the_environment_can_choose_it() -> None:
    chosen = backend_from_env({"GUT_BACKEND": "openai-decisions"})
    assert isinstance(chosen, OpenAIDecisionsBackend)
    assert chosen.model_id == "gpt-6-luna"
    assert chosen.base_url == "https://api.openai.com/v1"

    chosen = backend_from_env(
        {
            "GUT_BACKEND": "openai-decisions",
            "GUT_MODEL": "gpt-6-luna-2026-09-22",
            "GUT_BASE_URL": "https://gateway.example/v1",
        }
    )
    assert isinstance(chosen, OpenAIDecisionsBackend)
    assert chosen.model_id == "gpt-6-luna-2026-09-22"
    assert chosen.base_url == "https://gateway.example/v1"


# --------------------------------------------------------------------------- through gut


def test_the_three_primitives_read_its_answers() -> None:
    server = FakeServer()
    gut.configure(backend=server.backend(), cache=gut.NullCache())

    spam = gut.likely("the ticket", "is a bug report")
    assert spam.outcome is gut.YES
    assert spam.p == pytest.approx(0.92)
    assert spam.model == "gpt-6-luna-2026-09-22"

    team = gut.classify("the ticket", Team)
    assert team.value is Team.PLATFORM
    assert team.probabilities[Team.PLATFORM] == pytest.approx(0.9)

    urgency = gut.rate("the ticket", ["can wait", "this week", "right now"])
    assert urgency.score == pytest.approx(1.1)
    assert len(server.requests) == 3


def test_judge_batches_into_one_request() -> None:
    server = FakeServer()
    gut.configure(backend=server.backend(), cache=gut.NullCache())
    with gut.judge("the ticket") as j:
        bug = j.likely("is a bug report")
        team = j.classify(Team)
    assert bool(bug)
    assert team.value is Team.PLATFORM
    assert len(server.requests) == 1
    assert len(server.bodies[0]["questions"]) == 2


def test_each_asks_once_per_subject() -> None:
    server = FakeServer()
    gut.configure(backend=server.backend(), cache=gut.NullCache())
    decisions = gut.each(["a", "b", "c"]).likely("is spam")
    assert [d.outcome for d in decisions] == [gut.YES] * 3
    assert sorted(body["input"] for body in server.bodies) == ["a", "b", "c"]


def test_a_refusal_moves_on_in_a_cascade() -> None:
    refusing = replying({"answers": [{"type": "refusal", "name": "q"}]})
    gut.configure(
        backend=gut.Cascade(refusing, gut.FakeBackend(default=0.1)), cache=gut.NullCache()
    )
    decision = gut.likely("the ticket", "is spam")
    assert decision.outcome is gut.NO
    assert decision.model == "fake-1.0"


# --------------------------------------------------------------------------- awaitable


@pytest.mark.anyio
async def test_awaiting_reads_exactly_what_blocking_does() -> None:
    server = FakeServer()
    backend = OpenAIDecisionsBackend(
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(server))
    )
    awaited = await backend.aask("the ticket", {"bug": NoulSpec("is a bug report"), "team": TEAM})

    bug = awaited.answers["bug"]
    team = awaited.answers["team"]
    assert isinstance(bug, gut.NoulAnswer)
    assert isinstance(team, gut.ChoiceAnswer)
    assert bug.p == pytest.approx(0.92)
    assert team.choice == "PLATFORM"
    assert len(server.requests) == 1
    assert awaited.input_tokens == 57


@pytest.mark.anyio
async def test_awaited_failures_say_what_went_wrong() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    down = OpenAIDecisionsBackend(
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    )
    with pytest.raises(BackendError, match="could not be reached"):
        await down.aask("t", {"q": NoulSpec("q")})

    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Rate limit reached"}})

    rejecting = OpenAIDecisionsBackend(
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(reject))
    )
    with pytest.raises(BackendError, match="answered 429"):
        await rejecting.aask("t", {"q": NoulSpec("q")})


def test_each_event_loop_gets_its_own_client() -> None:
    backend = OpenAIDecisionsBackend(base_url="http://127.0.0.1:1/v1")

    async def current() -> httpx.AsyncClient:
        return backend._async_client()

    async def twice() -> tuple[httpx.AsyncClient, httpx.AsyncClient]:
        return backend._async_client(), backend._async_client()

    first, again = asyncio.run(twice())
    assert first is again
    second = asyncio.run(current())
    assert second is not first

    async def close() -> None:
        async with backend:
            pass

    asyncio.run(close())
    assert backend._async is None
    assert second.is_closed


@pytest.mark.anyio
async def test_a_supplied_async_client_is_left_open() -> None:
    supplied = httpx.AsyncClient()
    backend = OpenAIDecisionsBackend(async_client=supplied)
    assert backend._async_client() is supplied
    await backend.aclose()
    assert not supplied.is_closed
    await supplied.aclose()


def test_closing_closes_only_a_client_it_created() -> None:
    with OpenAIDecisionsBackend() as own:
        assert own.model_id == "gpt-6-luna"
    assert own._client.is_closed

    supplied = httpx.Client()
    OpenAIDecisionsBackend(client=supplied).close()
    assert not supplied.is_closed
    supplied.close()
