"""Models that run in this process: no API, no key, no per-call bill.

Two kinds, because "small model" covers two very different things:

- **`TransformersBackend`** runs a small instruction-tuned language model -- Qwen3-0.6B, Qwen2.5,
  SmolLM2, Llama 3.2 1B, Gemma 3 1B -- and reads its answer from the next-token distribution (see
  `_labels`). The subject is run through the model once; every question about it -- both readings
  of each -- then continues from that shared key-value cache, all of them in a single batched
  forward pass. Ten questions cost little more than one.
- **`ZeroShotBackend`** runs a natural-language-inference encoder such as DeBERTa-v3. It does not
  generate at all: it scores how strongly the subject *entails* a hypothesis. The default model has
  70M parameters and answers a question in about a tenth of a second on a laptop CPU.

Both need the optional `local` extra (`pip install "gutfeel[local]"`), imported only when one of
these classes is constructed, so `import gut` never pulls in `torch`.

The code that touches `torch` is excluded from the coverage floor: it cannot run in CI without
downloading models. Everything it feeds -- prompts, label ids, the arithmetic on the logits -- is
pure and tested, and `tests/test_local_models.py` runs these classes end to end against real models
wherever the extra is installed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from gut._backends._labels import (
    Prompt,
    answer_from,
    build_prompts,
    is_predicate,
    label_token_ids,
    render_subject,
    shared_prefix_length,
)
from gut._backends._many import Item
from gut._backends.base import Answer, BackendResponse, ChoiceAnswer, NoulAnswer, ScoreAnswer
from gut._errors import BackendError
from gut._questions import ChoiceSpec, NoulSpec, QuestionSpec, ScoreSpec, State

MISSING_EXTRA: Final = (
    "Local models need PyTorch and Hugging Face transformers, which are an optional extra. "
    'Install them with: pip install "gutfeel[local]"'
)

DEFAULT_CAUSAL_MODEL: Final = "Qwen/Qwen3-0.6B"

MAX_ROWS: Final = 32
"""Questions (counting both readings of each) run together in one forward pass."""

TEMPLATE_DEFAULTS: Final = {"enable_thinking": False}
"""Passed to every chat template unless overridden. The answer is read from the very first token,
so a model that opens with `<think>` has nothing to read; templates that do not know the switch
ignore it."""
DEFAULT_NLI_MODEL: Final = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"


def _import_local() -> tuple[Any, Any]:  # pragma: no cover - depends on the optional extra
    try:
        import torch
        import transformers
    except ImportError as error:
        raise BackendError(MISSING_EXTRA) from error
    return torch, transformers


def _pick_device(torch: Any, device: str | None) -> str:  # pragma: no cover - needs torch
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pinned(name: str, config: Any) -> str:  # pragma: no cover - needs a loaded model
    """`name@commit`, when the files came from the Hub and say which commit they are."""
    commit = getattr(config, "_commit_hash", None)
    return f"{name}@{commit[:12]}" if isinstance(commit, str) and commit else name


# --------------------------------------------------------------------------- causal language models


class TransformersBackend:
    """A small language model running locally, read through its next-token probabilities.

    ```python
    gut.TransformersBackend()                                   # Qwen/Qwen3-0.6B
    gut.TransformersBackend("Qwen/Qwen2.5-1.5B-Instruct")
    gut.TransformersBackend("HuggingFaceTB/SmolLM2-360M-Instruct")
    ```

    Args:
        model: A Hugging Face model id or a local path. It needs a chat template.
        device: `"cuda"`, `"mps"` or `"cpu"`. Defaults to the fastest one available.
        dtype: Passed to `from_pretrained`. Defaults to the model's own, except on a CPU, where
            it is `"float32"`: half precision is about twice as slow there, not faster.
        revision: A branch, tag or commit to pin, so the model cannot change under you.
        chat_template_kwargs: Passed to the chat template, over `enable_thinking=False`.
        balanced: Read yes/no questions and choices in both label orders and average, cancelling
            a small model's lean towards whichever label comes first.

    Raises:
        BackendError: The `local` extra is not installed.
    """

    def __init__(
        self,
        model: str = DEFAULT_CAUSAL_MODEL,
        *,
        device: str | None = None,
        dtype: str | None = None,
        revision: str | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        balanced: bool = True,
    ) -> None:  # pragma: no cover - loads a model
        torch, transformers = _import_local()
        self._torch = torch
        self._device = _pick_device(torch, device)
        options: dict[str, Any] = {"revision": revision}
        if dtype is None and self._device == "cpu":
            dtype = "float32"
        if dtype is not None:
            options["dtype"] = dtype
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model, revision=revision)
        loaded = transformers.AutoModelForCausalLM.from_pretrained(model, **options)
        self._model = loaded.to(self._device).eval()
        pad = self._tokenizer.pad_token_id
        self._pad_id = int(pad if pad is not None else self._tokenizer.eos_token_id or 0)
        self._model_id = model if revision is None else f"{model}@{revision}"
        self._resolved = _pinned(model, self._model.config)
        self._template_kwargs = {**TEMPLATE_DEFAULTS, **(chat_template_kwargs or {})}
        self._balanced = balanced
        self._label_ids: dict[tuple[tuple[str, ...], bool], dict[str, frozenset[int]]] = {}

    @property
    def model_id(self) -> str:  # pragma: no cover - needs a loaded model
        """The model as configured, with the revision when one was pinned."""
        return self._model_id

    def ask(
        self, state: State, questions: Mapping[str, QuestionSpec]
    ) -> BackendResponse:  # pragma: no cover - runs a model
        """Answer every question from one shared pass over the subject."""
        return self.ask_many([(state, questions)])[0]

    def ask_many(self, items: Sequence[Item]) -> list[BackendResponse]:  # pragma: no cover
        """Every question about every subject, in batched forward passes.

        One subject with many questions shares that subject's computation; many subjects share the
        system prompt, and run together `MAX_ROWS` at a time. The prefix is found on token ids, so
        both cases are the same code.
        """
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
        flat = [p for per_item in renderings for prompts in per_item.values() for p in prompts]
        distributions = iter(self._next_token_distributions([self._encode(p) for p in flat]))

        responses: list[BackendResponse] = []
        for (_, questions), per_item in zip(items, renderings, strict=True):
            answers: dict[str, Answer] = {}
            for name, prompts in per_item.items():
                readings = []
                for prompt in prompts:
                    distribution = next(distributions)
                    mass = {
                        label: float(distribution[sorted(ids)].sum())
                        for label, ids in self._ids_for(prompt).items()
                    }
                    readings.append((prompt, mass))
                answers[name] = answer_from(questions[name], readings)
            responses.append(BackendResponse(answers=answers, model=self._resolved, cost=0.0))
        return responses

    def _encode(self, prompt: Prompt) -> list[int]:  # pragma: no cover - needs a tokenizer
        messages = [dict(message) for message in prompt.messages]
        try:
            encoded = self._render(messages)
        except Exception:
            # Some chat templates refuse a system turn; fold it into the user's.
            system, user = messages
            merged = [{"role": "user", "content": f"{system['content']}\n\n{user['content']}"}]
            encoded = self._render(merged)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(token) for token in ids]

    def _render(self, messages: list[dict[str, str]]) -> Any:  # pragma: no cover
        return self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, **self._template_kwargs
        )

    def _ids_for(self, prompt: Prompt) -> dict[str, frozenset[int]]:  # pragma: no cover
        key = (tuple(prompt.labels), prompt.fold_case)
        if key not in self._label_ids:
            self._label_ids[key] = label_token_ids(
                prompt, lambda text: self._tokenizer.encode(text, add_special_tokens=False)
            )
        return self._label_ids[key]

    def _next_token_distributions(
        self, sequences: Sequence[Sequence[int]]
    ) -> list[Any]:  # pragma: no cover - runs a model
        """The next-token distribution after each sequence, computing their common prefix once.

        The prefix -- the system prompt and the subject -- runs alone. Its key-value cache is then
        repeated once per sequence and every remainder runs against it in one batch, right-padded.
        No attention mask is needed for that: under causal attention a real token never sees the
        padding after it, and every row's positions continue from the same prefix length.
        """
        torch = self._torch
        shared = shared_prefix_length(sequences) if len(sequences) > 1 else 0
        distributions: list[Any] = []
        with torch.inference_mode():
            prefix_cache = None
            if shared:
                prefix = torch.tensor([list(sequences[0][:shared])], device=self._device)
                prefix_cache = self._model(input_ids=prefix, use_cache=True).past_key_values
            for start in range(0, len(sequences), MAX_ROWS):
                rows = [list(sequence[shared:]) for sequence in sequences[start : start + MAX_ROWS]]
                cache = None
                if prefix_cache is not None:
                    cache = copy.deepcopy(prefix_cache)
                    cache.batch_repeat_interleave(len(rows))
                width = max(len(row) for row in rows)
                padded = [row + [self._pad_id] * (width - len(row)) for row in rows]
                input_ids = torch.tensor(padded, device=self._device)
                logits = self._model(
                    input_ids=input_ids, past_key_values=cache, use_cache=cache is not None
                ).logits
                ends = torch.tensor([len(row) - 1 for row in rows], device=self._device)
                last = logits[torch.arange(len(rows), device=self._device), ends]
                distributions.extend(torch.softmax(last.float(), dim=-1).cpu())
        return distributions


# --------------------------------------------------------------------------- NLI zero-shot encoders


def as_hypothesis(text: str) -> str:
    """Give a bare predicate a subject: `"is a bug report"` becomes `"This text is a bug report."`.

    Anything that already reads as a sentence -- `"the customer threatens to cancel"` -- is left
    alone.
    """
    stripped = text.strip()
    if is_predicate(stripped):
        return f"This text {stripped.rstrip('.')}."
    return stripped


def nli_labels(id2label: Mapping[int, str]) -> tuple[int, int]:
    """The indices of "entailment" and of whatever stands against it in this model's head.

    Two-way models say `not_entailment`; three-way MNLI models say `contradiction` and also have a
    `neutral` that is ignored, as the usual zero-shot recipe does.

    Raises:
        BackendError: This is not an NLI classifier.
    """
    by_name = {str(name).lower().replace(" ", "_"): int(index) for index, name in id2label.items()}
    entail = by_name.get("entailment")
    against = by_name.get("not_entailment", by_name.get("contradiction"))
    if entail is None or against is None:
        raise BackendError(
            f"ZeroShotBackend needs a natural-language-inference model, whose labels include "
            f"'entailment'; this one has {sorted(by_name)}."
        )
    return entail, against


def hypotheses_for(spec: QuestionSpec, template: str) -> list[str]:
    """What the subject is tested against, one hypothesis per answer the question can have."""
    match spec:
        case NoulSpec():
            return [as_hypothesis(spec.yes_means or spec.instructions)]
        case ChoiceSpec():
            return [
                template.format(description or name.replace("_", " ").lower())
                for name, description in spec.criteria.items()
            ]
        case ScoreSpec():  # pragma: no branch - exhaustive, proven by mypy
            return [as_hypothesis(level) for level in spec.criteria]


def _softmax(values: Sequence[float]) -> list[float]:
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def answer_from_entailment(spec: QuestionSpec, log_odds: Sequence[float]) -> Answer:
    """Turn entailment log-odds, one per hypothesis, into a typed answer.

    A yes/no question has one hypothesis, and its log-odds are the answer. A choice or a rating is
    single-label: the options compete, so their log-odds go through one softmax.
    """
    match spec:
        case NoulSpec():
            (single,) = log_odds
            return NoulAnswer(p=_softmax([single, 0.0])[0])
        case ChoiceSpec():
            names = list(spec.criteria)
            shares = _softmax(log_odds)
            probabilities = dict(zip(names, shares, strict=True))
            best = max(probabilities, key=probabilities.__getitem__)
            return ChoiceAnswer(
                choice=best, confidence=probabilities[best], probabilities=probabilities
            )
        case ScoreSpec():  # pragma: no branch - exhaustive, proven by mypy
            shares = _softmax(log_odds)
            by_level = dict(enumerate(shares))
            return ScoreAnswer(
                score=sum(level * p for level, p in by_level.items()),
                confidence=max(shares),
                probabilities=by_level,
                legend=dict(enumerate(spec.criteria)),
            )


class ZeroShotBackend:
    """A natural-language-inference model as a backend: no prompt, no generation, no API.

    `likely` asks whether the subject entails the question. `classify` asks it once per option,
    phrased through `hypothesis_template`, and lets the options compete. `rate` does the same with
    each level, so write levels as statements: `"someone is blocked right now"`, not `"today"`.

    ```python
    gut.ZeroShotBackend()                                    # 70M parameters, runs on a CPU
    gut.ZeroShotBackend("MoritzLaurer/deberta-v3-base-zeroshot-v2.0")   # larger, more accurate
    gut.ZeroShotBackend("facebook/bart-large-mnli")
    ```

    Args:
        model: A Hugging Face NLI classifier: its labels must include `entailment`.
        device: `"cuda"`, `"mps"` or `"cpu"`. Defaults to the fastest one available.
        revision: A branch, tag or commit to pin.
        hypothesis_template: How an option becomes a hypothesis for `classify`.
        max_length: Tokens per subject-hypothesis pair; longer subjects are truncated.
        batch_size: Pairs per forward pass.

    Raises:
        BackendError: The `local` extra is not installed, or the model is not an NLI classifier.
    """

    def __init__(
        self,
        model: str = DEFAULT_NLI_MODEL,
        *,
        device: str | None = None,
        revision: str | None = None,
        hypothesis_template: str = "This text is about {}.",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:  # pragma: no cover - loads a model
        torch, transformers = _import_local()
        self._torch = torch
        self._device = _pick_device(torch, device)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model, revision=revision)
        loaded = transformers.AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision
        )
        self._model = loaded.to(self._device).eval()
        self._entail, self._against = nli_labels(self._model.config.id2label)
        self._model_id = model if revision is None else f"{model}@{revision}"
        self._resolved = _pinned(model, self._model.config)
        self._template = hypothesis_template
        self._max_length = max_length
        self._batch_size = batch_size

    @property
    def model_id(self) -> str:  # pragma: no cover - needs a loaded model
        """The model as configured, with the revision when one was pinned."""
        return self._model_id

    def ask(
        self, state: State, questions: Mapping[str, QuestionSpec]
    ) -> BackendResponse:  # pragma: no cover - runs a model
        """Score every hypothesis of every question against the subject, in batches."""
        return self.ask_many([(state, questions)])[0]

    def ask_many(self, items: Sequence[Item]) -> list[BackendResponse]:  # pragma: no cover
        """Every subject-hypothesis pair from every item, `batch_size` pairs per forward pass."""
        pairs: list[tuple[str, str]] = []
        layout: list[dict[str, int]] = []
        for state, questions in items:
            if not questions:
                raise BackendError("A backend call needs at least one question.")
            premise = render_subject(state)
            counts: dict[str, int] = {}
            for name, spec in questions.items():
                hypotheses = hypotheses_for(spec, self._template)
                pairs.extend((premise, hypothesis) for hypothesis in hypotheses)
                counts[name] = len(hypotheses)
            layout.append(counts)
        scores = self._log_odds(pairs)

        responses: list[BackendResponse] = []
        start = 0
        for (_, questions), counts in zip(items, layout, strict=True):
            answers: dict[str, Answer] = {}
            for name, count in counts.items():
                answers[name] = answer_from_entailment(
                    questions[name], scores[start : start + count]
                )
                start += count
            responses.append(BackendResponse(answers=answers, model=self._resolved, cost=0.0))
        return responses

    def _log_odds(self, pairs: Sequence[tuple[str, str]]) -> list[float]:  # pragma: no cover
        torch = self._torch
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self._batch_size):
                chunk = list(pairs[start : start + self._batch_size])
                inputs = self._tokenizer(
                    [premise for premise, _ in chunk],
                    [hypothesis for _, hypothesis in chunk],
                    truncation="only_first",
                    max_length=self._max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**inputs).logits.float()
                odds = logits[:, self._entail] - logits[:, self._against]
                scores.extend(float(value) for value in odds.cpu().tolist())
        return scores
