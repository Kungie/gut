# Backends

[← docs index](README.md)

A backend is whatever actually answers. `gut` asks every backend the same three kinds of question --
yes/no, which one, how much -- and gets back the same typed answers, so the code above it never
changes when the model does:

```python
import gut

gut.configure(backend=gut.ZeroShotBackend())
```

| backend | runs | install | reach for it when |
|---|---|---|---|
| [`ZeroShotBackend`](#zeroshotbackend) | in your process, CPU is fine | `gut[local]` | yes/no and routing on short text, for free |
| [`TransformersBackend`](#transformersbackend) | in your process, GPU helps | `gut[local]` | you want a small language model and no server |
| [`OpenAICompatibleBackend`](#openaicompatiblebackend) | Ollama, vLLM, llama.cpp, OpenAI | core | a model is already served somewhere |
| [`JevBackend`](#jevbackend) | TypeSafe AI's API | `gut[jev]` | a hosted model built for exactly these questions |
| [`Cascade`](#cascade) | wherever its stages run | core | cheap model first, bigger only when unsure |
| [`FakeBackend`](#fakebackend) | nowhere | core | tests and offline work |
| [your own](#writing-your-own) | anywhere | -- | you have a model, or a rule, `gut` does not know |

Every real backend is imported on first use, so `import gut` never loads PyTorch, an HTTP client or
a vendor SDK you did not ask for.

## `ZeroShotBackend`

A natural-language-inference model: it reads the subject and a *hypothesis* and scores how strongly
one entails the other. It never generates text, so there is no prompt to tune and nothing to parse.

```python
import gut

gut.ZeroShotBackend()                                             # 70M params, ~150 MB, CPU
gut.ZeroShotBackend("MoritzLaurer/deberta-v3-base-zeroshot-v2.0") # larger, more accurate
gut.ZeroShotBackend("facebook/bart-large-mnli")                   # the classic, 400M params
```

`likely` asks whether the subject entails the question -- a bare predicate like `"is spam"` is read
as *"This text is spam."* `classify` asks it once per option, through `hypothesis_template`
(`"This text is about {}."`), and lets the options compete. `rate` does the same per level, so
write levels as statements: `"someone is blocked right now"`, not `"today"`.

It is fast and free, and it is literal: "is spam or abusive" is one hypothesis it reads poorly, so
ask two questions. Any Hugging Face NLI model whose labels include `entailment` works.

## `TransformersBackend`

A small instruction-tuned language model, in your process.

```python
import gut

gut.TransformersBackend()                                         # Qwen/Qwen3-0.6B
gut.TransformersBackend("Qwen/Qwen2.5-1.5B-Instruct")
gut.TransformersBackend("HuggingFaceTB/SmolLM2-1.7B-Instruct", device="cpu")
```

Any model with a chat template will run; Qwen3-0.6B and Qwen2.5-0.5B are the ones this project has
tried. Gated models such as Llama 3.2 and Gemma 3 need their licence accepted on Hugging Face and a
login first.

The answer is **read, not generated**: the question asks for a single label -- `Yes`/`No`, a letter
per option, a digit per level -- and the probability of each label is taken from the model's
next-token distribution. Two details make those probabilities worth deciding on:

- **One pass over the subject.** The subject is computed once and every question about it continues
  from that shared state, all in one batched forward pass. Ten questions cost little more than one.
- **Both orders.** Small models lean towards whichever label they are offered first, so a yes/no
  question is asked as "Yes or No" *and* "No or Yes", a choice with its options both ways round,
  and the readings averaged. It costs a few dozen tokens. [D40](../DECISIONS.md) has the numbers.

It picks CUDA, then Apple's GPU, then the CPU, and on a CPU uses float32, which there is about twice
as fast as half precision. Thinking modes are switched off: the answer is the first token.

## `OpenAICompatibleBackend`

Anything that speaks the OpenAI chat completions protocol **and returns log-probabilities**.

```python
import gut

gut.OpenAICompatibleBackend("gpt-4.1-nano")                                  # OpenAI
gut.OpenAICompatibleBackend("qwen2.5:1.5b", base_url="http://localhost:11434/v1")  # Ollama
gut.OpenAICompatibleBackend("Qwen/Qwen2.5-1.5B-Instruct",
                            base_url="http://localhost:8000/v1")               # vLLM
gut.OpenAICompatibleBackend("local", base_url="http://localhost:8080/v1")      # llama.cpp
```

Each question is a request for one token and its top log-probabilities; the questions in a batch go
out concurrently, subject first, so a server with prefix caching reads it once. Yes/no questions and
choices are asked both ways round, as above -- two requests each. Pass `balanced=False` for a model
you know has no lean.

Log-probabilities are the one hard requirement. OpenAI's reasoning models (the o-series and
`gpt-5`) do not return them, and Ollama needs 0.12.11 or later. A server that leaves them out gets a
`BackendError` saying so, never a guessed answer. Choices are labelled A to Z on this backend, so
they are capped at 26 options.

`OPENAI_API_KEY` is sent only to OpenAI, or to `OPENAI_BASE_URL` if you set one. Point the backend
anywhere else and it sends no key unless you pass `api_key=`.

## `JevBackend`

TypeSafe AI's Jev answers typed questions natively -- no labels, no prompt -- and bills on input
only, so batching is close to free.

```python
import gut

gut.JevBackend(model="jev-1.13.0")        # pin a version; aliases move
```

`pip install "gut[jev]"` and set `TYPESAFE_API_KEY`. It is the one backend `gut` will build without
being told to, because that variable exists for nothing else.

## `Cascade`

Ask the cheapest model first, and pass on only what it is unsure of:

```python
import gut

gut.configure(backend=gut.Cascade(
    gut.ZeroShotBackend(),                      # free, local, settles the obvious
    gut.TransformersBackend(),                  # reads what the first could not
    gut.OpenAICompatibleBackend("gpt-4.1-nano"),
    unsure_band=(0.2, 0.8),                     # a yes/no answer in here goes on
    min_confidence=0.7,                         # a choice or rating below this goes on
))
```

Whatever reaches the last stage is kept. A stage that fails -- a server that is down, a choice
with more options than a model can label -- hands its questions on too, so a cascade doubles as a
fallback chain. It is an ordinary backend: `@semantic` hands it a whole batch, the cache stores its
answers, and `decision.model` names the stage that actually answered. `cascade.answered_by` counts
answers per model -- the number that shows what the cheap stage saved.

The band is the cascade's own. Your `stakes` and `ask_human` still apply to the final answer, so
what the last model is unsure of can still reach a person.

## `FakeBackend`

Answers from fixtures, counts its calls, and never touches a model. It is strict: a question with no
fixture raises rather than inventing a probability.

```python
import gut

backend = gut.FakeBackend(answers={
    "is a bug report": 0.91,          # a probability for likely()
    "which team owns this": "PLATFORM",  # an option name for classify()
    "how urgent is this": 2,          # a level for rate()
})
gut.configure(backend=backend)

gut.configure(backend=gut.FakeBackend(rule=gut.deterministic_rule))  # any question, stable answers
```

## Writing your own

A backend is two members: a `model_id`, and an `ask()` that takes one subject and a batch of named
questions and returns one answer per name. Here is the keyword rule you already had, as the free
first stage of a cascade:

```python
import re

import gut


class KeywordBackend:
    model_id = "keywords-1"

    def __init__(self, rules):
        self.rules = {q: [re.compile(p, re.I) for p in ps] for q, ps in rules.items()}

    def ask(self, state, questions):
        answers = {}
        for name, spec in questions.items():
            hit = any(p.search(str(state)) for p in self.rules.get(spec.instructions, []))
            answers[name] = gut.NoulAnswer(p=0.99 if hit else 0.5)   # 0.5: "no idea"
        return gut.BackendResponse(answers=answers, model=self.model_id)


keywords = KeywordBackend({"asks for a refund": [r"\brefund\b", r"money back"]})
gut.configure(backend=gut.Cascade(keywords, gut.FakeBackend(default=0.1)))

assert gut.likely("I want my money back", "asks for a refund").model == "keywords-1"
assert gut.likely("Where is my parcel?", "asks for a refund").model == "fake-1.0"
```

[`examples/custom_backend.py`](../examples/custom_backend.py) is the complete version, handling
every question shape. The rest of the contract:

- **Answer every name you were asked**, with the answer type matching the question: `NoulAnswer` for
  a `NoulSpec`, `ChoiceAnswer` for a `ChoiceSpec`, `ScoreAnswer` for a `ScoreSpec`.
- **Probabilities, not verdicts.** `gut` applies the caller's posture to them; a backend that only
  ever says 0 or 1 disables `ask_human` for everyone who uses it.
- **`model` is the exact version that answered**; `model_id` is what you were configured with and
  goes into the cache key. If different questions were answered by different models, say which in
  `BackendResponse.models`.
- **Raise `BackendError`** for anything that went wrong, so a `Cascade` can move on.
