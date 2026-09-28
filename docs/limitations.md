# Honest limitations

[← docs index](README.md)

**Small models are small.** That is the point, and it has a price. On one laptop run of the
[examples](../examples/), the 70M-parameter NLI model routed support tickets to the right team and
missed a customer saying they would move to another vendor; Qwen3-0.6B caught that and was unsure
about much else. Neither is wrong to exist -- they are cheap enough to ask about everything -- but
check your own questions on your own data before trusting a model with anything that matters, and
let `ask_human=True` and [`Cascade`](backends.md#cascade) catch what the cheap model cannot settle.

**Each backend has its own blind spots.**

- NLI models read one claim at a time. "Is spam or abusive" is two questions.
- Language models under a billion parameters lean towards whichever label they are offered first.
  `gut` reads every yes/no question and choice in both orders and averages them, which removes most
  of the lean and doubles the requests on a hosted server.
- Text-model backends label a choice's options A to Z, so they take at most 26 options. The NLI
  backend and Jev have no such limit.
- OpenAI-compatible servers report their top 20 tokens. An option that does not appear there is given
  the most probability it could have had, which makes the answer look less certain, never more.

**The same number means different things on different backends.** A probability of 0.8 from an NLI
model and 0.8 from a language model are not the same claim. `stakes` and `lean` sit on top of
whatever the backend reports, so switching backends can move where your boundaries effectively
fall. Look at `d.p` for a handful of subjects you know the answer to after switching.

**`confidence` is not an accuracy estimate.** For `classify` and `rate` it is how peaked the answer's
own distribution is -- on the local and OpenAI-compatible backends, the probability of the top
option. A model can be confidently wrong.

**Probabilities across questions are not comparable.** `gut` never synthesises `P(no)` from a
separately asked negated question, and neither should you.

**Text in the subject can influence the answer**, and `gut` does not do taint tracking. Treat a
decision over user-controlled text as advisory in security contexts.

**Do the arithmetic in Python.** Small models are weak at counting, arithmetic and date comparison.
Compute those, and let `gut` judge the rest.

**Local models need memory and time.** Qwen3-0.6B takes about 1.2 GB of memory in half precision and
twice that in float32, which is what it uses on a CPU. Measured on one 8 GB M-series laptop, it
answered one question in about 2 s on the CPU and 0.4 s on the GPU, and five questions about the
same ticket in 2.6 s and 0.9 s; the NLI model took about 0.1 s per question on the CPU.

**Decision ids survive edits, not moves.** An id is the question plus the module and function it is
asked in, so inserting lines above a call doesn't change it -- moving it to another function does.

**`@semantic` is speculative.** It asks questions behind branches that never run.

**Reading a `judge()` handle is synchronous.** In async code, `await j.aresolve()` before reading, or
the first read asks the model on the spot and blocks the event loop while it does.
