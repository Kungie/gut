"""OpenAI's Decisions API as a backend.

`POST /v1/decisions` answers typed questions directly -- the probability that a condition holds, a
distribution over options, a score against ordered levels -- which are `gut`'s three questions under
other names. So, as with Jev, there is no prompt to build and no label to read: a question goes out
as it is and the answer comes back as numbers.

Every question about one subject rides in a single request, and billing is on input, so the subject
is paid for once however many questions come with it. That is what `@semantic` and `judge()` batch
for. Many subjects are many requests, sent concurrently by `each()`.

Four things this module decides, because the API leaves them open:

- **A refusal is an error.** The API may decline a question and say so with an answer of type
  `refusal`, which carries no probability. That becomes a `BackendError`, never a made-up `0.5`, so
  a `Cascade` can hand the question on.
- **`yes_means` and `no_means` go into the instructions.** A predicate has no field for them.
- **A choice or a rating with no question of its own gets a plain one**, since the API's questions
  all carry instructions.
- **Cost is not computed.** The API reports tokens, not dollars, and a price written down here would
  be wrong the day it changes. A gateway that puts `cost` in `usage` has it passed on.

The API is in public beta and its limits -- questions per request, options, levels -- are not
published, so none is enforced here beyond `gut`'s own; what the server refuses comes back as a
`BackendError` with its explanation.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from typing import Any, Final

import httpx

from gut._backends.base import (
    Answer,
    BackendResponse,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    reported_cost,
)
from gut._backends.openai import OPENAI_BASE_URL
from gut._errors import BackendError
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State

DECISIONS_MODEL: Final = "gpt-6-luna"
"""The one model the Decisions API serves so far."""

WHICH_OPTION: Final = "Which option fits the input best?"
WHERE_ON_SCALE: Final = "Where does the input fall on this scale?"


def _input(state: State) -> str:
    """The subject as the API takes it: text as-is, anything else as indented JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def _question(name: str, spec: QuestionSpec) -> dict[str, Any]:
    """Translate one of our specs into the API's question object."""
    match spec:
        case NoulSpec():
            lines = [spec.instructions]
            if spec.yes_means is not None:
                lines.append(f"It counts as true if: {spec.yes_means}")
            if spec.no_means is not None:
                lines.append(f"It counts as false if: {spec.no_means}")
            return {"type": "predicate", "name": name, "instructions": "\n".join(lines)}
        case ChoiceSpec():
            return {
                "type": "choice",
                "name": name,
                "instructions": spec.instructions or WHICH_OPTION,
                "choices": [
                    {"value": option, "description": description}
                    if description is not None
                    else {"value": option}
                    for option, description in spec.criteria.items()
                ],
            }
        case ScoreSpec():  # pragma: no branch - exhaustive, proven by mypy
            return {
                "type": "score",
                "name": name,
                "instructions": spec.instructions or WHERE_ON_SCALE,
                "levels": [{"label": level} for level in spec.criteria],
            }


def _answer(spec: QuestionSpec, raw: Mapping[str, Any]) -> Answer:
    """Translate one of the API's answers into ours.

    Raises:
        KeyError, TypeError, ValueError: The answer lacks a field its type should have.
    """
    match spec:
        case NoulSpec():
            return NoulAnswer(p=float(raw["probability"]))
        case ChoiceSpec():
            return ChoiceAnswer(
                choice=str(raw["choice"]),
                confidence=float(raw["confidence"]),
                probabilities={
                    str(entry["value"]): float(entry["probability"])
                    for entry in raw["probabilities"]
                },
            )
        case ScoreSpec():  # pragma: no branch - exhaustive, proven by mypy
            return ScoreAnswer(
                score=float(raw["score"]),
                confidence=float(raw["confidence"]),
                probabilities={
                    int(entry["value"]): float(entry["probability"])
                    for entry in raw["probabilities"]
                },
                legend=dict(enumerate(spec.criteria)),
            )


_KINDS: Final[Mapping[type[QuestionSpec], str]] = {
    NoulSpec: "predicate",
    ChoiceSpec: "choice",
    ScoreSpec: "score",
}


class OpenAIDecisionsBackend:
    """Answers questions with OpenAI's Decisions API.

    ```python
    gut.OpenAIDecisionsBackend()                 # gpt-6-luna, with OPENAI_API_KEY
    ```

    Args:
        model: The model to ask. `gpt-6-luna` is the only one the API serves so far.
        base_url: The server's `/v1` root. Defaults to `OPENAI_BASE_URL`, then to OpenAI.
        api_key: Sent as a bearer token. Left unset, `OPENAI_API_KEY` is used -- but only for the
            server that key belongs to, never forwarded to an explicitly chosen `base_url`.
        timeout: Seconds per request.
        extra_body: Merged into every request, for parameters such as `safety_identifier`.
        client: An `httpx.Client` to send requests with, instead of a new one. It is left open
            when this backend closes, since it is yours.
        async_client: The same for the awaitable methods, an `httpx.AsyncClient`. Left unset, one
            is made for each event loop that uses this backend.

    Raises:
        BackendError: `model` is empty.
    """

    def __init__(
        self,
        model: str = DECISIONS_MODEL,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        extra_body: Mapping[str, Any] | None = None,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip():
            raise BackendError(f"model must be a non-empty string, got {model!r}.")

        environment_url = os.environ.get("OPENAI_BASE_URL", "").strip() or OPENAI_BASE_URL
        self.base_url = (base_url or environment_url).rstrip("/")
        if api_key is None and self.base_url == environment_url.rstrip("/"):
            api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None

        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._extra_body = dict(extra_body or {})
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._supplied_async = async_client
        self._async: tuple[asyncio.AbstractEventLoop, httpx.AsyncClient] | None = None

    @property
    def model_id(self) -> str:
        """The model name as configured."""
        return self._model

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """Answer every question in one request.

        Raises:
            BackendError: The request failed, the API declined a question, or its reply did not
                contain the answers asked for.
        """
        body = self._body(state, questions)
        try:
            response = self._client.post(self._url, json=body, headers=self._headers)
        except httpx.HTTPError as error:
            raise BackendError(f"{self._where} could not be reached: {error}") from error
        return self._read(questions, response)

    async def aask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """`ask`, awaitable: the request goes out on the event loop, no thread involved."""
        body = self._body(state, questions)
        try:
            response = await self._async_client().post(self._url, json=body, headers=self._headers)
        except httpx.HTTPError as error:
            raise BackendError(f"{self._where} could not be reached: {error}") from error
        return self._read(questions, response)

    # ------------------------------------------------------------------ shared by both paths

    def _body(self, state: State, questions: Mapping[str, QuestionSpec]) -> dict[str, Any]:
        if not questions:
            raise BackendError("A backend call needs at least one question.")
        return {
            "model": self._model,
            "input": _input(state),
            "questions": [_question(name, spec) for name, spec in questions.items()],
            **self._extra_body,
        }

    @property
    def _url(self) -> str:
        return f"{self.base_url}/decisions"

    @property
    def _where(self) -> str:
        return f"{self._model} at {self.base_url}"

    def _read(
        self, questions: Mapping[str, QuestionSpec], response: httpx.Response
    ) -> BackendResponse:
        if response.status_code >= 400:
            raise BackendError(
                f"{self._where} answered {response.status_code}: {response.text[:300]}"
            )
        try:
            data = response.json()
            given = {str(raw["name"]): raw for raw in data["answers"]}
        except (ValueError, KeyError, TypeError) as error:
            raise BackendError(f"{self._where} sent a reply with no answers in it.") from error

        answers: dict[str, Answer] = {}
        for name, spec in questions.items():
            if name not in given:
                raise BackendError(f"{self._where} answered without {name!r}.")
            raw = given[name]
            kind = raw.get("type")
            if kind == "refusal":
                raise BackendError(f"{self._where} declined to answer {name!r}.")
            if kind != _KINDS[type(spec)]:
                raise BackendError(
                    f"{self._where} answered {name!r}, a {_KINDS[type(spec)]} question, with "
                    f"a {kind!r} answer."
                )
            try:
                answers[name] = _answer(spec, raw)
            except (KeyError, TypeError, ValueError) as error:
                raise BackendError(
                    f"{self._where} sent a {kind} answer for {name!r} that could not be read."
                ) from error

        usage = data.get("usage")
        tokens = usage.get("input_tokens") if isinstance(usage, Mapping) else None
        return BackendResponse(
            answers=answers,
            # The version that answered, when the server names one, rather than what was asked for.
            model=str(data.get("model") or self._model),
            input_tokens=tokens if isinstance(tokens, int) else None,
            cost=reported_cost(usage),
        )

    def _async_client(self) -> httpx.AsyncClient:
        """The async client for the running event loop.

        An `httpx.AsyncClient` belongs to the loop it first ran on, and a program may run several
        -- one `asyncio.run` after another -- so a loop that is not the last one gets its own.
        """
        if self._supplied_async is not None:
            return self._supplied_async
        loop = asyncio.get_running_loop()
        if self._async is None or self._async[0] is not loop:
            self._async = (loop, httpx.AsyncClient(timeout=self._timeout))
        return self._async[1]

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Close the HTTP client this backend created."""
        if self._owns_client:
            self._client.close()

    async def aclose(self) -> None:
        """Close both clients this backend created, from inside the event loop."""
        self.close()
        if self._async is not None:
            await self._async[1].aclose()
            self._async = None

    def __enter__(self) -> OpenAIDecisionsBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> OpenAIDecisionsBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
