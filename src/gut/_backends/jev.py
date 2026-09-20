"""The Jev backend, built on TypeSafe AI's Python SDK.

Imported lazily: `typesafe-sdk` is an optional extra (`gut[jev]`), and nothing in the core --
the cost rule, decision types, cache, batching, the whole test suite -- may depend on it.

What this module deliberately does **not** do is implement retries. The SDK already retries with
exponential backoff and honours both `retry-after` and `retry-after-ms`; wrapping it in a second
loop would stack two backoffs and double the delay on a 429. So the knobs here configure the SDK's
own `RetryPolicy` rather than replacing it. See D4 in DECISIONS.md.
"""

from __future__ import annotations

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
    'Install it with: pip install "gut[jev]"'
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
            whatever you name here (D12).
        api_key: Overrides `TYPESAFE_API_KEY`.
        base_url: Overrides `TYPESAFE_BASE_URL`.
        timeout: Seconds per HTTP operation.
        max_retries: Convenience for the SDK's retry policy. Mutually exclusive with `retry`.
        retry: A full SDK `RetryPolicy`, for anything `max_retries` does not cover.
        client: An already-configured `TypeSafeClient`, used as-is. Mutually exclusive with every
            other connection argument.

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
    ) -> None:
        sdk = _sdk()
        self._sdk = sdk

        if max_retries is not None and retry is not None:
            raise BackendError(
                "Pass max_retries or retry, not both: max_retries is shorthand for a RetryPolicy."
            )
        if client is not None and any(
            argument is not None for argument in (api_key, base_url, timeout, max_retries, retry)
        ):
            raise BackendError(
                "A supplied client is used as configured; configure it there rather than passing "
                "api_key, base_url, timeout, max_retries or retry alongside it."
            )

        self._model = model or sdk.constants.DEFAULT_MODEL
        if client is not None:
            self._client = client
        else:
            policy = retry
            if max_retries is not None:
                policy = sdk.RetryPolicy(max_retries=max_retries)
            self._client = sdk.TypeSafeClient(
                api_key=api_key,
                base_url=base_url,
                model=self._model,
                timeout=timeout,
                retry=policy,
            )

    @property
    def model_id(self) -> str:
        """The model this backend asks for -- an alias unless a version was pinned."""
        return self._model

    @property
    def client(self) -> typesafe_sdk.TypeSafeClient:
        """The underlying SDK client."""
        return self._client

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """Answer every question in one request.

        Raises:
            BackendError: The request failed, or the response did not contain the answers asked
                for. The originating SDK exception is kept as the cause.
        """
        if not questions:
            raise BackendError("A backend call needs at least one question.")

        payload = {name: _to_sdk_question(spec, self._sdk) for name, spec in questions.items()}
        try:
            response = self._client.system_one(state=state, questions=payload, model=self._model)
        except self._sdk.TypeSafeError as error:
            raise BackendError(f"Jev request failed: {error}") from error

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

    def close(self) -> None:
        """Close the underlying client."""
        self._client.close()

    def __enter__(self) -> JevBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
