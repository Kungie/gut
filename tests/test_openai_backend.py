"""Tests for `OpenAICompatibleBackend`, against a fake server rather than a real one."""

from __future__ import annotations

import enum
import json
import math
import threading
from collections.abc import Callable
from typing import Any

import httpx
import pytest

import gut
from gut import BackendError, OpenAICompatibleBackend
from gut._questions import ChoiceSpec, NoulSpec, ScoreSpec

Handler = Callable[[httpx.Request], httpx.Response]


def p_of(answer: object) -> float:
    """The probability of a yes/no answer, narrowed for the type checker."""
    assert isinstance(answer, gut.NoulAnswer), answer
    return answer.p


class Team(enum.Enum):
    BILLING = "invoices and refunds"
    PLATFORM = "outages"
    OTHER = "anything else"


def completion(
    top: list[tuple[str, float]], *, model: str = "gpt-4.1-nano-2025-04-14", tokens: int = 42
) -> dict[str, Any]:
    """A chat completion the way OpenAI, Ollama and vLLM shape it."""
    return {
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": top[0][0]},
                "logprobs": {
                    "content": [
                        {
                            "token": top[0][0],
                            "logprob": math.log(top[0][1]),
                            "top_logprobs": [
                                {"token": token, "logprob": math.log(p)} for token, p in top
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 1},
    }


class FakeServer:
    """Answers each request from what its prompt asks for, and remembers every request."""

    def __init__(self, answer: Callable[[str], dict[str, Any]] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self._answer = answer or self.default
        self._lock = threading.Lock()

    @staticmethod
    def default(prompt: str) -> dict[str, Any]:
        if "Yes or No" in prompt or "No or Yes" in prompt:
            return completion([("Yes", 0.8), ("No", 0.2)])
        if "letter" in prompt:
            return completion([("B", 0.7), ("A", 0.2), ("C", 0.1)])
        return completion([("2", 0.6), ("1", 0.4)])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        with self._lock:
            self.requests.append(request)
            self.bodies.append(body)
        return httpx.Response(200, json=self._answer(body["messages"][-1]["content"]))

    def backend(self, **options: Any) -> OpenAICompatibleBackend:
        options.setdefault("model", "gpt-4.1-nano")
        client = httpx.Client(transport=httpx.MockTransport(self))
        return OpenAICompatibleBackend(client=client, **options)


@pytest.fixture(autouse=True)
def _no_ambient_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_an_unbalanced_question_is_one_single_token_request_with_logprobs() -> None:
    server = FakeServer()
    backend = server.backend(balanced=False)
    response = backend.ask("the ticket", {"q": NoulSpec("is a bug report")})

    assert p_of(response.answers["q"]) == pytest.approx(0.8)
    assert response.model == "gpt-4.1-nano-2025-04-14"
    assert response.input_tokens == 42
    assert response.models is None
    (body,) = server.bodies
    assert body["model"] == "gpt-4.1-nano"
    assert body["max_tokens"] == 1
    assert body["temperature"] == 0
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert str(server.requests[0].url) == "https://api.openai.com/v1/chat/completions"


def test_a_balanced_question_is_asked_both_ways_round_and_averaged() -> None:
    def leaning(prompt: str) -> dict[str, Any]:
        # A model that says whichever label it was offered first.
        first = "Yes" if prompt.endswith("Yes or No.") else "No"
        other = "No" if first == "Yes" else "Yes"
        return completion([(first, 0.9), (other, 0.1)])

    server = FakeServer(leaning)
    response = server.backend().ask("the ticket", {"q": NoulSpec("is a bug report")})
    assert p_of(response.answers["q"]) == pytest.approx(0.5)
    endings = sorted(body["messages"][-1]["content"][-10:] for body in server.bodies)
    assert endings == ["No or Yes.", "Yes or No."]
    assert response.input_tokens == 84


def test_every_question_shape_is_read_back() -> None:
    server = FakeServer()
    response = server.backend().ask(
        "the ticket",
        {
            "bug": NoulSpec("is a bug report"),
            "team": ChoiceSpec(instructions=None, criteria={m.name: m.value for m in Team}),
            "urgency": ScoreSpec(instructions=None, criteria=["later", "soon", "now"]),
        },
    )
    team = response.answers["team"]
    urgency = response.answers["urgency"]
    assert isinstance(team, gut.ChoiceAnswer)
    assert team.choice == "PLATFORM"
    assert isinstance(urgency, gut.ScoreAnswer)
    assert urgency.score == pytest.approx(1.6)
    assert len(server.requests) == 5  # yes/no and choice both ways round, the rating once
    assert response.input_tokens == 5 * 42


def test_a_batch_goes_out_concurrently() -> None:
    barrier = threading.Barrier(6, timeout=5)

    def answer(prompt: str) -> dict[str, Any]:
        barrier.wait()  # deadlocks unless all six requests are in flight at once
        return FakeServer.default(prompt)

    backend = FakeServer(answer).backend(max_concurrency=6)
    questions = {f"q{i}": NoulSpec(f"question {i}") for i in range(3)}
    assert len(backend.ask("t", questions).answers) == 3


def test_per_answer_models_are_reported_when_they_differ() -> None:
    def answer(prompt: str) -> dict[str, Any]:
        model = "a-1" if "first" in prompt else "a-2"
        return completion([("Yes", 0.9), ("No", 0.1)], model=model)

    backend = FakeServer(answer).backend()
    response = backend.ask("t", {"x": NoulSpec("first"), "y": NoulSpec("second")})
    assert response.models == {"x": "a-1", "y": "a-2"}
    assert response.model_for("y") == "a-2"


def test_a_server_without_usage_reports_no_tokens() -> None:
    def answer(prompt: str) -> dict[str, Any]:
        body = completion([("No", 0.9), ("Yes", 0.1)])
        del body["usage"]
        return body

    response = FakeServer(answer).backend().ask("t", {"q": NoulSpec("q")})
    assert response.input_tokens is None
    assert p_of(response.answers["q"]) == pytest.approx(0.1)


def test_the_chosen_token_is_used_when_no_alternatives_are_listed() -> None:
    def answer(prompt: str) -> dict[str, Any]:
        body = completion([("Yes", 0.97)])
        del body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        return body

    response = FakeServer(answer).backend(top_logprobs=1).ask("t", {"q": NoulSpec("q")})
    # "No" was not listed, so it gets the 3% the listed token leaves over -- never zero.
    assert p_of(response.answers["q"]) == pytest.approx(0.97)


def test_extra_body_is_merged_into_every_request() -> None:
    server = FakeServer()
    server.backend(extra_body={"seed": 7, "top_logprobs": 5}).ask("t", {"q": NoulSpec("q")})
    assert server.bodies[0]["seed"] == 7
    assert server.bodies[0]["top_logprobs"] == 5


def test_it_works_through_gut_likely_and_judge() -> None:
    server = FakeServer()
    gut.configure(backend=server.backend(), cache=gut.NullCache())

    decision = gut.likely("the ticket", "is a bug report")
    assert decision == gut.YES
    assert decision.p == pytest.approx(0.8)
    assert decision.model == "gpt-4.1-nano-2025-04-14"

    with gut.judge("the ticket") as j:
        bug = j.likely("is a bug report")
        team = j.classify(Team)
    assert bug
    assert team.value is Team.PLATFORM


# --------------------------------------------------------------------------- where it sends things


def test_a_local_server_is_reached_at_its_own_url() -> None:
    server = FakeServer()
    backend = server.backend(model="qwen2.5:0.5b", base_url="http://localhost:11434/v1/")
    backend.ask("t", {"q": NoulSpec("q")})
    assert str(server.requests[0].url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in server.requests[0].headers


def test_the_environment_key_goes_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environment")
    server = FakeServer()
    server.backend().ask("t", {"q": NoulSpec("q")})
    assert server.requests[0].headers["authorization"] == "Bearer sk-from-environment"


def test_the_environment_key_follows_the_environment_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proxy")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
    server = FakeServer()
    backend = server.backend()
    backend.ask("t", {"q": NoulSpec("q")})
    assert backend.base_url == "https://proxy.example/v1"
    assert server.requests[0].headers["authorization"] == "Bearer sk-proxy"


def test_the_environment_key_is_never_sent_to_another_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    server = FakeServer()
    server.backend(base_url="http://localhost:8000/v1").ask("t", {"q": NoulSpec("q")})
    assert "authorization" not in server.requests[0].headers


def test_an_explicit_key_is_sent_wherever_it_is_pointed() -> None:
    server = FakeServer()
    backend = server.backend(base_url="https://api.together.xyz/v1", api_key="tg-key")
    backend.ask("t", {"q": NoulSpec("q")})
    assert server.requests[0].headers["authorization"] == "Bearer tg-key"


# --------------------------------------------------------------------------- failures


def test_a_server_that_cannot_be_reached_is_a_backend_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = OpenAICompatibleBackend(
        "qwen2.5:0.5b",
        base_url="http://localhost:11434/v1",
        client=httpx.Client(transport=httpx.MockTransport(refuse)),
    )
    with pytest.raises(
        BackendError, match=r"qwen2\.5:0\.5b at http://localhost:11434/v1 could not"
    ):
        backend.ask("t", {"q": NoulSpec("q")})


def test_an_error_status_is_reported_with_the_servers_explanation() -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "logprobs is not supported"}})

    backend = OpenAICompatibleBackend(
        "gpt-5-nano", client=httpx.Client(transport=httpx.MockTransport(reject))
    )
    with pytest.raises(BackendError, match=r"answered 400: .*logprobs is not supported"):
        backend.ask("t", {"q": NoulSpec("q")})


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "m", "choices": [{"message": {"content": "Yes"}}]},
        {"model": "m", "choices": [{"logprobs": {"content": []}}]},
        {"model": "m", "choices": []},
        "not json",
    ],
)
def test_a_response_without_logprobs_says_why_that_matters(payload: object) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if isinstance(payload, str):
            return httpx.Response(200, text=payload)
        return httpx.Response(200, json=payload)

    backend = OpenAICompatibleBackend(
        "m", client=httpx.Client(transport=httpx.MockTransport(respond))
    )
    with pytest.raises(BackendError, match="returned no log-probabilities"):
        backend.ask("t", {"q": NoulSpec("q")})


def test_an_answer_that_is_not_a_label_is_an_error_not_a_guess() -> None:
    backend = FakeServer(lambda prompt: completion([("<think>", 0.99)])).backend()
    with pytest.raises(BackendError, match="thinking model"):
        backend.ask("t", {"q": NoulSpec("q")})


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"model": " "}, "model must be a non-empty string"),
        ({"model": "m", "max_concurrency": 0}, "max_concurrency must be at least 1"),
        ({"model": "m", "top_logprobs": 0}, "top_logprobs must be at least 1"),
    ],
)
def test_bad_arguments_are_refused(options: dict[str, Any], message: str) -> None:
    with pytest.raises(BackendError, match=message):
        OpenAICompatibleBackend(**options)


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(BackendError, match="at least one question"):
        FakeServer().backend().ask("t", {})


def test_closing_closes_only_a_client_it_created() -> None:
    with OpenAICompatibleBackend("m") as own:
        assert own.model_id == "m"
    assert own._client.is_closed

    supplied = httpx.Client()
    OpenAICompatibleBackend("m", client=supplied).close()
    assert not supplied.is_closed
    supplied.close()


def test_many_subjects_share_one_pool_of_requests() -> None:
    barrier = threading.Barrier(4, timeout=5)

    def answer(prompt: str) -> dict[str, Any]:
        barrier.wait()  # two subjects, both readings each: all four in flight together
        return FakeServer.default(prompt)

    backend = FakeServer(answer).backend(max_concurrency=4)
    responses = backend.ask_many(
        [("first", {"q": NoulSpec("is spam")}), ("second", {"q": NoulSpec("is spam")})]
    )
    assert [p_of(r.answers["q"]) for r in responses] == [pytest.approx(0.8)] * 2


def test_each_on_a_server_asks_every_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    gut.configure(backend=server.backend(balanced=False), cache=gut.NullCache())
    decisions = gut.each(["a", "b", "c"]).likely("is spam")
    assert [d.outcome for d in decisions] == [gut.YES] * 3
    assert sorted(body["messages"][-1]["content"][7] for body in server.bodies) == ["a", "b", "c"]


def test_a_batch_with_an_empty_question_set_is_refused() -> None:
    with pytest.raises(BackendError, match="at least one question"):
        FakeServer().backend().ask_many([("a", {})])
