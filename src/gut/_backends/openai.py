"""Any OpenAI-compatible chat server as a backend.

One protocol covers a lot of small, cheap models: OpenAI's own non-reasoning models
(`gpt-4.1-nano`, `gpt-4o-mini`), and anything served by Ollama, vLLM or llama.cpp -- Qwen, Llama,
Gemma, Phi, SmolLM and the rest.

Each question becomes a request for **a single token with its log-probabilities** -- two for a
yes/no question or a choice, which are read in both label orders and averaged. Nothing is
generated past that token and nothing is parsed: the answer is read from the probabilities the
server reports (see `_labels`). A batch goes out concurrently, and because the subject comes first
in every prompt, a server with prefix caching reads it once.

The server must return `logprobs`. OpenAI's reasoning models do not, and Ollama needs 0.12.11 or
later; a server that leaves them out gets a `BackendError` that says so, never a made-up answer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final

import httpx

from gut._backends._labels import Prompt, answer_from, build_prompts, mass_from_top_logprobs
from gut._backends._many import Item
from gut._backends.base import Answer, BackendResponse
from gut._errors import BackendError
from gut._questions import QuestionSpec, State

OPENAI_BASE_URL: Final = "https://api.openai.com/v1"

NO_LOGPROBS: Final = (
    "returned no log-probabilities, which this backend reads its answers from. OpenAI's reasoning "
    "models (the o-series, gpt-5) do not provide them; gpt-4.1-nano and gpt-4o-mini do. Ollama "
    "needs version 0.12.11 or later."
)


class OpenAICompatibleBackend:
    """Answers questions with any server that speaks the OpenAI chat completions protocol.

    ```python
    gut.OpenAICompatibleBackend("gpt-4.1-nano")                                    # OpenAI
    gut.OpenAICompatibleBackend("qwen2.5:0.5b", base_url="http://localhost:11434/v1")  # Ollama
    gut.OpenAICompatibleBackend("Qwen/Qwen2.5-1.5B-Instruct",
                                base_url="http://localhost:8000/v1")                # vLLM
    ```

    Args:
        model: The model name the server knows it by.
        base_url: The server's `/v1` root. Defaults to `OPENAI_BASE_URL`, then to OpenAI.
        api_key: Sent as a bearer token. Left unset, `OPENAI_API_KEY` is used -- but only for the
            server that key belongs to, never forwarded to an explicitly chosen `base_url`.
        timeout: Seconds per request.
        max_concurrency: How many questions from one batch may be in flight at once.
        top_logprobs: How many alternatives to ask for. Twenty is OpenAI's and Ollama's maximum.
        extra_body: Merged into every request, for server-specific switches.
        client: An `httpx.Client` to send requests with, instead of a new one. It is left open
            when this backend closes, since it is yours.
        balanced: Ask yes/no questions and choices in both label orders and average the two.
            Doubles their requests; turn it off for a model you know has no order bias.

    Raises:
        BackendError: An argument is out of range.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_concurrency: int = 8,
        top_logprobs: int = 20,
        extra_body: Mapping[str, Any] | None = None,
        client: httpx.Client | None = None,
        balanced: bool = True,
    ) -> None:
        if not model.strip():
            raise BackendError(f"model must be a non-empty string, got {model!r}.")
        if max_concurrency < 1:
            raise BackendError(f"max_concurrency must be at least 1, got {max_concurrency}.")
        if top_logprobs < 1:
            raise BackendError(f"top_logprobs must be at least 1, got {top_logprobs}.")

        environment_url = os.environ.get("OPENAI_BASE_URL", "").strip() or OPENAI_BASE_URL
        self.base_url = (base_url or environment_url).rstrip("/")
        if api_key is None and self.base_url == environment_url.rstrip("/"):
            api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None

        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._max_concurrency = max_concurrency
        self._top_logprobs = top_logprobs
        self._extra_body = dict(extra_body or {})
        self._balanced = balanced
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    @property
    def model_id(self) -> str:
        """The model name as configured."""
        return self._model

    def ask(self, state: State, questions: Mapping[str, QuestionSpec]) -> BackendResponse:
        """Answer every question with single-token requests, sent concurrently.

        Raises:
            BackendError: A request failed, or the server's answer could not be read.
            QuestionError: A choice has more options than this backend can label.
        """
        return self.ask_many([(state, questions)])[0]

    def ask_many(self, items: Sequence[Item]) -> list[BackendResponse]:
        """The same for many subjects, every request from all of them sharing one pool."""
        renderings: list[dict[str, tuple[Prompt, ...]]] = []
        for state, questions in items:
            if not questions:
                raise BackendError("A backend call needs at least one question.")
            renderings.append(
                {
                    name: build_prompts(state, spec, balanced=self._balanced)
                    for name, spec in questions.items()
                }
            )
        flat = [
            prompt for per_item in renderings for prompts in per_item.values() for prompt in prompts
        ]

        workers = min(self._max_concurrency, len(flat))
        if workers <= 1:
            results = [self._ask_one(prompt) for prompt in flat]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self._ask_one, flat))

        replies = iter(results)
        responses: list[BackendResponse] = []
        for (_, questions), per_item in zip(items, renderings, strict=True):
            answers: dict[str, Answer] = {}
            models: dict[str, str] = {}
            tokens = 0
            for name, prompts in per_item.items():
                readings = []
                for prompt in prompts:
                    mass, model, used = next(replies)
                    readings.append((prompt, mass))
                    models.setdefault(name, model)
                    tokens += used
                answers[name] = answer_from(questions[name], readings)
            first = next(iter(questions))
            responses.append(
                BackendResponse(
                    answers=answers,
                    model=models[first],
                    input_tokens=tokens or None,
                    models=models if len(set(models.values())) > 1 else None,
                )
            )
        return responses

    def _ask_one(self, prompt: Prompt) -> tuple[dict[str, float], str, int]:
        """One request: the label probabilities, the model that answered, and prompt tokens."""
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [dict(message) for message in prompt.messages],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self._top_logprobs,
            **self._extra_body,
        }
        where = f"{self._model} at {self.base_url}"
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions", json=body, headers=self._headers
            )
        except httpx.HTTPError as error:
            raise BackendError(f"{where} could not be reached: {error}") from error
        if response.status_code >= 400:
            raise BackendError(f"{where} answered {response.status_code}: {response.text[:300]}")

        try:
            data = response.json()
            first = data["choices"][0]["logprobs"]["content"][0]
            top = [
                (str(item["token"]), float(item["logprob"]))
                for item in first.get("top_logprobs") or ()
            ] or [(str(first["token"]), float(first["logprob"]))]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise BackendError(f"{where} {NO_LOGPROBS}") from error

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        return (
            mass_from_top_logprobs(prompt, top),
            str(data.get("model") or self._model),
            prompt_tokens if isinstance(prompt_tokens, int) else 0,
        )

    def close(self) -> None:
        """Close the HTTP client this backend created."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAICompatibleBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
