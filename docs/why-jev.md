# Why Jev makes this worth doing

[← docs index](README.md)

`gut` would work on an LLM. It would just be too slow and too expensive to use the way it is meant
to be used — a judgment inside an `if`, several per request, everywhere.

TypeSafe's own published figures for Jev, which are **self-reported vendor benchmarks and should be
read as such**:

| | Jev | frontier LLMs |
|---|---|---|
| end-to-end response | 70–500 ms | 3–329 s |
| speed, like for like | “40x-200x faster for the same levels of frontier intelligence for System One shaped queries” | |
| input tokens | $0.042 / MTok | $0.20 – $10 / MTok |
| output tokens | free | roughly 5x input |

Their homepage quotes `193.6x faster` and `444.6x cheaper` from workflow evaluations, with the
caveat — theirs, not ours — that those are "on the higher end of real world gains".

Two structural differences matter more than the multipliers. Jev emits all its probabilities in
parallel rather than generating tokens one at a time, which is where the latency goes; and it
returns typed answers, so there is nothing to parse and nothing to hallucinate. Billing on input
only is what makes [batching](batching.md) nearly free.

None of that is `gut`'s claim to verify, and none of it is load-bearing for the design. Jev is why
this is practical now, not what the library is.