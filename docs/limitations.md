# Honest limitations

[← docs index](README.md)


**About the model.** Jev reads instructions literally and is weak at counting, arithmetic and date
comparison — compute those in Python and let `gut` judge the rest. Irrelevant state hurts accuracy,
so pass the narrowest subject that contains the answer. Multi-hop reasoning degrades.

**Text in the subject can influence the answer**, and `gut` does not do taint tracking. Treat a
decision over user-controlled text as advisory in security contexts.

**"Calibrated" is a claim, not a guarantee, and the measurements above show where it fails.** The
cost rule and the postures are only as good as the probabilities feeding them. Measure yours with
`gut eval` before trusting a preset band to sit where this page says it does, and fix what you can
with `gut calibrate`. A correction is fitted on your data, for your model — nothing ships
pre-calibrated, because nothing could be.

**`confidence` is not an accuracy estimate.** For `classify` and `rate` it is a statistic over how
peaked the answer's own distribution is. It held up for `classify` on our data and inverted for
`rate`. Measure it.

**Costs do not transfer across backends.** A cost model is a claim about *these* probabilities. A
different model, or an unpinned version that moves, can be sharper or flatter in the middle and put
the implied boundaries somewhere else on its distribution.

**Probabilities across questions are not comparable.** The vendor's docs warn that negated questions
need not sum to 1 and that different primitives yield non-comparable numbers. `gut` never
synthesises `P(no)` from a separately asked negated question, and neither should you.

**Decision ids survive edits, not moves.** An id is the question plus the module and function it is
asked in, so inserting lines above a call doesn't reset its history — moving it to another function
does.

**`@semantic` is speculative.** It asks questions behind branches that never run. It handles
`async` functions by running the prefetch in a worker thread -- correct and non-blocking, but a
backend that spoke `async` natively would not need the thread. `judge()` is still synchronous.

**Context limits.** 64k tokens for subject plus every question, 32k for subject plus the longest
one. Large batches split automatically, using a character-count estimate rather than a real
tokeniser — a wrong estimate costs an extra request, never a wrong answer.
