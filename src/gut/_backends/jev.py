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
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from gut._backends.base import (
    BackendResponse,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
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
            self._client = sdk.TypeSafeClient(**connection)
        self._supplied_async = async_client
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
            assert self._connection is not None  # only unset when an async client was supplied
            self._client = self._sdk.TypeSafeClient(**self._connection)
        return self._client

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """Answer every question in one request.

        Raises:
            BackendError: The request failed, or the response did not contain the answers asked
                for. The originating SDK exception is kept as the cause.
        """
        payload = self._payload(questions)
        try:
            response = self.client.system_one(state=state, questions=payload, model=self._model)
        except self._sdk.TypeSafeError as error:
            raise BackendError(f"Jev request failed: {error}") from error
        return self._response(questions, response)

    async def aask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """`ask`, awaitable, through the SDK's own async client: no thread involved."""
        client = self._async_client()
        if client is None:
            return await asyncio.to_thread(self.ask, state, questions)
        payload = self._payload(questions)
        try:
            response = await client.system_one(state=state, questions=payload, model=self._model)
        except self._sdk.TypeSafeError as error:
            raise BackendError(f"Jev request failed: {error}") from error
        return self._response(questions, response)

    def _payload(self, questions: Mapping[str, QuestionSpec]) -> dict[str, Any]:
        if not questions:
            raise BackendError("A backend call needs at least one question.")
        return {name: _to_sdk_question(spec, self._sdk) for name, spec in questions.items()}

    def _response(self, questions: Mapping[str, QuestionSpec], response: Any) -> BackendResponse:
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
        )

    def _async_client(self) -> typesafe_sdk.AsyncTypeSafeClient | None:
        """The SDK's async client for the running loop, or `None` to fall back to a thread."""
        if self._supplied_async is not None:
            return self._supplied_async
        if self._connection is None:
            return None
        loop = asyncio.get_running_loop()
        if self._async is None or self._async[0] is not loop:
            self._async = (loop, self._sdk.AsyncTypeSafeClient(**self._connection))
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
