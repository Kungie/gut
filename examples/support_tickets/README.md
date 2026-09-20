# Demo: support ticket triage

101 synthetic support tickets with ground-truth labels, and the same triage handler written twice.

```bash
python report.py                          # stand-in model, no key needed
TYPESAFE_API_KEY=... python report.py     # the real thing
```

## The files

| | |
|---|---|
| `tickets.json` | 101 tickets. 26 are marked `hard` with a note saying why. |
| `before.py` | The handler as you would write it without `gut`: five calls, a made-up `0.7`, no third branch. |
| `after.py` | The same handler with `gut`: one call, costs instead of thresholds, `UNSURE` routed to a person. |
| `report.py` | Runs `after.py` over the dataset and prints what it cost. |
| `_simulate.py` | A stand-in model so the demo runs with no API key. A prop, not a model. |

## The dataset

Written to be awkward on purpose, because a dataset of obvious cases measures nothing. The 26 hard
ones include:

- **"How do I cancel?"** — every churn keyword, zero churn risk. It is an upgrade.
- **"Cancellation policy?"** — a *pre-sales* question that reads as a threat.
- **"I'm not threatening anything, but I do have to justify this spend."** — explicitly denies the
  thing it is doing.
- **"cancel"** — one word, no context. Genuinely ambiguous, and labelled as a threat because the
  downside is asymmetric.
- **"Nothing works and nobody answers"** — a real churn risk with no cancellation word anywhere.
- **"(no subject) / see attached"** — nothing to judge. Any confident answer is wrong.
- Tickets that are genuinely two tickets, where any routing decision is half wrong.

## What the report showed

Against the real model, all 101 tickets:

```
queue
  resolved automatically               40  40%
  sent to a human                      61  60%

automatic decisions
  wrong                                 0  0% of automatic
    churn risk missed                   0  at 20 each
    escalated needlessly                0  at 2 each

cost
  total                                61  on after.py's scale

model
  requests                            101  1.0 per ticket
  judgments                           505  5.0 per request
  wall clock                       38.90s  385 ms per ticket

the ambiguous ones
  marked hard                          26  26% of the dataset
    sent to a human                    22  85% of them
    wrong automatically                 0

same answers, before.py's rule
  a hard threshold of 0.7, and no third branch
  wrong                                 8  vs 0 with the cost rule
    churn risk missed                   8  vs 0
  total cost                          160  vs 61
  requests                            505  vs 101, one per judgment
```

Three things in there are worth more than the rest.

**Nothing was auto-decided wrongly, and 85% of the hard tickets reached a person.** The model is
uncertain in almost exactly the places the dataset says are ambiguous. Without a third branch that
uncertainty has nowhere to go and becomes a confident guess.

**505 judgments in 101 requests.** One per ticket, five judgments each. `before.py` cannot do that
without restructuring the function, because nothing in it knows the five questions are about the
same thing.

**The same answers under a hard `0.7` miss eight churn risks.** Not because the model was wrong —
it is the same model and the same probabilities — but because `0.7` encodes a claim nobody checked:
that escalating a calm customer is roughly as costly as losing an angry one. The costs say it is ten
times cheaper, and the threshold that follows is `0.09`.

The stand-in model gives a different and more interesting shape, because it is deliberately worse:

```
  resolved automatically               62  61%
  sent to a human                      39  39%
  wrong                                 8  13% of automatic
    churn risk missed                   0
    escalated needlessly                8
  total cost                           55  vs 120 under before.py's rule
```

Read that comparison carefully, because it is not flattering in the obvious way. **The cost rule
makes *more* wrong automatic decisions than the hard threshold** — eight against six. It is cheaper
anyway, by more than half, because the eight are all needless escalations at `2` while the six
include every missed churn risk at `20`. Optimising for accuracy and optimising for cost are
different things, and only one of them is what the business is paying for.

## The bug this demo found

The first version of `after.py` used the handoff's own cost example — `cost_false_yes=2`,
`cost_false_no=50`, `cost_human=5` — and the report came back with **zero tickets sent to a human**.

Not a bug in the routing. The expected cost of asking a person is flat, while the cheaper of yes and
no peaks where those two cross, at `2 × 50 / 52 = 1.92`. A `cost_human` of `5` sits above that peak,
so at every probability either yes or no is cheaper and the `UNSURE` branch is unreachable. The code
was fine. The economics were wrong, and nothing said so.

`gut` warns about this now, and `Policy.max_useful_cost_human` will tell you the ceiling. See D19 in
[DECISIONS.md](../../DECISIONS.md).
