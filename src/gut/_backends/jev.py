"""The Jev backend, built on TypeSafe AI's Python SDK.

Imported lazily: `typesafe-sdk` is an optional extra (`gutfeel[jev]`), and nothing in the core --
the cost rule, decision types, cache, batching, the whole test suite -- may depend on it.

What this module deliberately does **not** do is implement retries. The SDK already retries with
exponential backoff and honours both `retry-after` and `retry-after-ms`; wrapping it in a second
loop would stack two backoffs and double the delay on a 429. So the knobs here configure the SDK's
own `RetryPolicy` rather than replacing it.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
from collections.abc import Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Final

from gut._backends.base import (
    BackendResponse,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    reported_cost,
)
from gut._errors import BackendError
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State

if TYPE_CHECKING:
    import typesafe_sdk

MISSING_SDK_HINT: Final = (
    "The Jev backend needs the TypeSafe SDK, which is an optional extra. "
    'Install it with: pip install "gutfeel[jev]"'
)

CONTEXT_LIMIT_TOKENS: Final = 64_000
"""Documented ceiling for state plus every question in one request."""

SINGLE_QUESTION_LIMIT_TOKENS: Final = 32_000
"""Documented ceiling for state plus the single longest question."""


OPENROUTER_BASE_URL: Final = "https://openrouter.ai/api"
"""OpenRouter serves Jev at `/v1/systemone` under this, speaking the same protocol as TypeSafe."""

OPENROUTER_MODEL: Final = "~typesafe/jev-latest"

OLLAYA_HOST: Final = "127.0.0.1:11435"
"""Where Ollaya listens unless `OLLAYA_HOST` says otherwise. It serves TypeSafe's API there."""
"""OpenRouter's name for the latest Jev. Pin a version, such as `typesafe/jev-1.13`, to keep it."""

# The SDK parses `usage` into token counts and drops everything else, including the `cost` that
# OpenRouter reports. The transport below reads it off the wire into the slot of the call that is
# waiting for it: a context variable, so concurrent calls -- threads or tasks -- never mix them up.
_cost_slot: ContextVar[list[float | None] | None] = ContextVar("gut_jev_cost", default=None)


def _record_cost(response: Any) -> None:
    slot = _cost_slot.get()
    if slot is None or response.status_code != 200:
        return
    try:
        slot[0] = reported_cost(json.loads(response.content).get("usage"))
    except (ValueError, AttributeError):
        slot[0] = None


def _inner_transport() -> Any:
    import httpx2

    return httpx2.HTTPTransport()


def _inner_async_transport() -> Any:
    import httpx2

    return httpx2.AsyncHTTPTransport()


@functools.cache
def _cost_reading_transports() -> tuple[type[Any], type[Any]]:
    """Transports that pass everything through and note the cost of each answer."""
    import httpx2

    class CostReading(httpx2.BaseTransport):
        def __init__(self) -> None:
            self._inner = _inner_transport()

        def handle_request(self, request: Any) -> Any:
            response = self._inner.handle_request(request)
            response.read()
            _record_cost(response)
            return response

        def close(self) -> None:
            self._inner.close()

    class AsyncCostReading(httpx2.AsyncBaseTransport):
        def __init__(self) -> None:
            self._inner = _inner_async_transport()

        async def handle_async_request(self, request: Any) -> Any:
            response = await self._inner.handle_async_request(request)
            await response.aread()
            _record_cost(response)
            return response

        async def aclose(self) -> None:
            await self._inner.aclose()

    return CostReading, AsyncCostReading


def _sdk() -> Any:
    """Import the SDK, turning a missing optional dependency into a readable error."""
    try:
        import typesafe_sdk
    except ImportError as error:  # pragma: no cover - exercised only without the extra installed
        raise BackendError(MISSING_SDK_HINT) from error
    return typesafe_sdk


def _to_sdk_question(spec: QuestionSpec, sdk: Any) -> Any:
    """Translate one of our specs into the SDK's question object."""
    match spec:
        case NoulSpec():
            criteria = {
                key: value
                for key, value in (("true", spec.yes_means), ("false", spec.no_means))
                if value is not None
            }
            return sdk.Noul(instructions=spec.instructions, criteria=criteria or None)
        case ChoiceSpec():
            return sdk.Choice(instructions=spec.instructions, criteria=dict(spec.criteria))
        case ScoreSpec():  # pragma: no branch - exhaustive, proven by mypy
            return sdk.Score(instructions=spec.instructions, criteria=list(spec.criteria))


def _as_text(value: object) -> str:
    """Legend entries come back as whatever was sent; we only ever send strings."""
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))


def _from_sdk_answer(answer: Any, name: str, sdk: Any) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
    """Translate one SDK answer into ours."""
    if isinstance(answer, sdk.NoulAnswer):
        return NoulAnswer(p=float(answer.noul))
    if isinstance(answer, sdk.ChoiceAnswer):
        return ChoiceAnswer(
            choice=answer.choice,
            confidence=float(answer.confidence),
            probabilities={key: float(value) for key, value in answer.probabilities.items()},
        )
    if isinstance(answer, sdk.ScoreAnswer):
        return ScoreAnswer(
            score=float(answer.score),
            confidence=float(answer.confidence),
            probabilities={int(k): float(v) for k, v in answer.probabilities.items()},
            legend={int(k): _as_text(v) for k, v in answer.legend.items()},
        )
    raise BackendError(f"Answer for {name!r} has an unrecognised type {type(answer).__name__}.")


class JevBackend:
    """Answers questions with TypeSafe AI's Jev.

    Args:
        model: Model name or alias. Defaults to the SDK's own default. Pin a version when
            thresholds have been calibrated against it -- aliases move, and the cache keys on
            whatever you name here.
        api_key: Overrides `TYPESAFE_API_KEY`.
        base_url: Overrides `TYPESAFE_BASE_URL`.
        timeout: Seconds per HTTP operation.
        max_retries: Convenience for the SDK's retry policy. Mutually exclusive with `retry`.
        retry: A full SDK `RetryPolicy`, for anything `max_retries` does not cover.
        client: An already-configured `TypeSafeClient`, used as-is. Mutually exclusive with every
            other connection argument.
        async_client: An already-configured `AsyncTypeSafeClient` for the awaitable methods. Left
            unset, one is built from the same arguments for each event loop that uses this backend
            -- unless `client` was supplied, in which case awaiting runs that client in a worker
            thread rather than guessing at its configuration.

    Raises:
        BackendError: The SDK is not installed, or the arguments conflict.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry: typesafe_sdk.RetryPolicy | None = None,
        client: typesafe_sdk.TypeSafeClient | None = None,
        async_client: typesafe_sdk.AsyncTypeSafeClient | None = None,
    ) -> None:
        sdk = _sdk()
        self._sdk = sdk

        if max_retries is not None and retry is not None:
            raise BackendError(
                "Pass max_retries or retry, not both: max_retries is shorthand for a RetryPolicy."
            )
        supplied = client is not None or async_client is not None
        if supplied and any(
            argument is not None for argument in (api_key, base_url, timeout, max_retries, retry)
        ):
            raise BackendError(
                "A supplied client is used as configured; configure it there rather than passing "
                "api_key, base_url, timeout, max_retries or retry alongside it."
            )

        self._model = model or sdk.constants.DEFAULT_MODEL
        policy = retry
        if max_retries is not None:
            policy = sdk.RetryPolicy(max_retries=max_retries)
        connection: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "model": self._model,
            "timeout": timeout,
            "retry": policy,
        }
        # With a client supplied, its configuration is unknown, so no async twin is guessed at.
        self._connection: dict[str, Any] | None = None if client is not None else connection
        # With only an async client supplied, the blocking one is built if and when it is needed.
        self._client: typesafe_sdk.TypeSafeClient | None = client
        if client is None and async_client is None:
            self._client = self._new_client()
        self._supplied_async = async_client
        # A model on your own machine: its calls cost nothing, whatever the server says.
        self._local = False
        self._async: tuple[asyncio.AbstractEventLoop, typesafe_sdk.AsyncTypeSafeClient] | None = (
            None
        )

    @property
    def model_id(self) -> str:
        """The model this backend asks for -- an alias unless a version was pinned."""
        return self._model

    @property
    def client(self) -> typesafe_sdk.TypeSafeClient:
        """The underlying blocking SDK client."""
        if self._client is None:
            self._client = self._new_client()
        return self._client

    @classmethod
    def openrouter(
        cls,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> JevBackend:
        """Jev through OpenRouter, billed to your OpenRouter credit.

        The same model and the same answers as `JevBackend()`, for when you have an OpenRouter
        account rather than a TypeSafe key. OpenRouter reports what every call cost, and gut passes
        it on: see `gut.usage()`.

        Args:
            model: OpenRouter's name for the model. Defaults to the latest Jev.
            api_key: Overrides `OPENROUTER_API_KEY`.
            timeout: Seconds per HTTP operation.
            max_retries: How often the SDK retries a failed request.

        Raises:
            BackendError: No OpenRouter key was given or found.
        """
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip() or None
        if key is None:
            raise BackendError(
                "Jev through OpenRouter needs an OpenRouter key: set OPENROUTER_API_KEY, or pass "
                "api_key=."
            )
        return cls(
            model=model or OPENROUTER_MODEL,
            api_key=key,
            base_url=OPENROUTER_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def ollaya(
        cls,
        model: str,
        *,
        host: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> JevBackend:
        """An open decision model on your own machine, served by Ollaya.

        Ollaya (https://ollaya.dev) runs open, Jev-style decision models locally -- `winnow:e4b`,
        `laya`, `kev` and others -- and speaks TypeSafe's API, so this is the Jev backend pointed at
        it. Calls are counted as free.

        Args:
            model: The model to ask, as pulled with `ollaya pull`, such as `"winnow:e4b"` or
                `"laya"`. Required: Ollaya has no model called Jev's default.
            host: Where Ollaya listens. Defaults to `OLLAYA_HOST`, then `127.0.0.1:11435`.
            api_key: Only needed when the server sets `OLLAYA_API_KEY`; read from there by default.
            timeout: Seconds per HTTP operation. The first request to a model waits while it loads.
            max_retries: How often the SDK retries a failed request.

        Raises:
            BackendError: No model was named.
        """
        if not model or not model.strip():
            raise BackendError(
                'Name the Ollaya model to ask, such as JevBackend.ollaya("winnow:e4b") or "laya".'
            )
        address = (host or os.environ.get("OLLAYA_HOST", "").strip() or OLLAYA_HOST).rstrip("/")
        if "://" not in address:
            address = f"http://{address}"
        # Ollaya binds 0.0.0.0 to listen everywhere; a client reaches it on this machine.
        address = address.replace("//0.0.0.0", "//127.0.0.1")
        key = api_key or os.environ.get("OLLAYA_API_KEY", "").strip() or "local"
        backend = cls(
            model=model.strip(),
            api_key=key,
            base_url=address,
            timeout=timeout,
            max_retries=max_retries,
        )
        backend._local = True
        return backend

    def _failed(self, error: Exception) -> str:
        if not self._local:
            return f"Jev request failed: {error}"
        where = self._connection["base_url"] if self._connection else "Ollaya"
        hint = ""
        if "connect" in str(error).lower():
            hint = (
                " Is Ollaya running? Any ollaya command starts it, such as "
                f"`ollaya pull {self._model}`, which also downloads the model."
            )
        return f"Ollaya at {where} did not answer: {error}.{hint}"

    def _new_client(self) -> typesafe_sdk.TypeSafeClient:
        assert self._connection is not None  # only unset when a client was supplied
        reading, _ = _cost_reading_transports()
        client: typesafe_sdk.TypeSafeClient = self._sdk.TypeSafeClient(
            **self._connection, transport=reading()
        )
        return client

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """Answer every question in one request.

        Raises:
            BackendError: The request failed, or the response did not contain the answers asked
                for. The originating SDK exception is kept as the cause.
        """
        payload = self._payload(questions)
        slot: list[float | None] = [None]
        token = _cost_slot.set(slot)
        try:
            response = self.client.system_one(state=state, questions=payload, model=self._model)
        except self._sdk.TypeSafeError as error:
            raise BackendError(self._failed(error)) from error
        finally:
            _cost_slot.reset(token)
        return self._response(questions, response, slot[0])

    async def aask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """`ask`, awaitable, through the SDK's own async client: no thread involved."""
        client = self._async_client()
        if client is None:
            return await asyncio.to_thread(self.ask, state, questions)
        payload = self._payload(questions)
        slot: list[float | None] = [None]
        token = _cost_slot.set(slot)
        try:
            response = await client.system_one(state=state, questions=payload, model=self._model)
        except self._sdk.TypeSafeError as error:
            raise BackendError(self._failed(error)) from error
        finally:
            _cost_slot.reset(token)
        return self._response(questions, response, slot[0])

    def _payload(self, questions: Mapping[str, QuestionSpec]) -> dict[str, Any]:
        if not questions:
            raise BackendError("A backend call needs at least one question.")
        return {name: _to_sdk_question(spec, self._sdk) for name, spec in questions.items()}

    def _response(
        self, questions: Mapping[str, QuestionSpec], response: Any, cost: float | None = None
    ) -> BackendResponse:
        missing = set(questions) - set(response.answers)
        if missing:
            names = ", ".join(sorted(repr(name) for name in missing))
            raise BackendError(f"Jev answered without {names}.")

        return BackendResponse(
            answers={
                name: _from_sdk_answer(response.answers[name], name, self._sdk)
                for name in questions
            },
            # The resolved version, never the alias that was asked for.
            model=response.model,
            input_tokens=response.usage.input_tokens,
            cost=0.0 if self._local else cost,
        )

    def _async_client(self) -> typesafe_sdk.AsyncTypeSafeClient | None:
        """The SDK's async client for the running loop, or `None` to fall back to a thread."""
        if self._supplied_async is not None:
            return self._supplied_async
        if self._connection is None:
            return None
        loop = asyncio.get_running_loop()
        if self._async is None or self._async[0] is not loop:
            _, reading = _cost_reading_transports()
            self._async = (
                loop,
                self._sdk.AsyncTypeSafeClient(**self._connection, transport=reading()),
            )
        return self._async[1]

    def close(self) -> None:
        """Close the blocking client, if one was made."""
        if self._client is not None:
            self._client.close()

    async def aclose(self) -> None:
        """Close both clients, from inside the event loop."""
        self.close()
        if self._async is not None:
            await self._async[1].aclose()
            self._async = None

    def __enter__(self) -> JevBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> JevBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
