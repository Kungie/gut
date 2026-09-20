# Demo: support ticket triage

101 synthetic support tickets with ground-truth labels, the same triage handler written twice, and
a sweep showing what each risk posture actually does to the queue.

```bash
python report.py                                     # offline, from the committed cassette
gut eval predicates/ --cassette cassettes/predicates.json --model jev-1.13.0
```

Both run with no API key. The answers are real — recorded from `jev-1.13.0` and committed — so
nothing here is simulated.

## The files

| | |
|---|---|
| `tickets.json` | 101 tickets. 26 are marked `hard`, each with a note saying why. |
| `before.py` | The handler as you would write it without `gut`: five calls, a made-up `0.7`, no third branch. |
| `after.py` | The same handler with `gut`: one call, a posture in words, `UNSURE` routed to a person. |
| `report.py` | Runs `after.py` at every posture and prints what each costs. |
| `predicates/` | The five judgments `after.py` makes, as eval files, with the dataset labels as the answer key. |
| `cassettes/` | Those 505 judgments as recorded from the real model. Safe to commit: the tickets are invented. |

## The dataset

Written to be awkward on purpose. The 26 hard ones include:

- **"How do I cancel?"** — every churn keyword, zero churn risk. It is an upgrade.
- **"Cancellation policy?"** — a *pre-sales* question that reads as a threat.
- **"I'm not threatening anything, but I do have to justify this spend."** — explicitly denies the
  thing it is doing.
- **"cancel"** — one word, no context.
- **"Nothing works and nobody answers"** — a real churn risk with no cancellation word anywhere.
- **"(no subject) / see attached"** — nothing to judge. Any confident answer is wrong.
- Tickets that are genuinely two tickets, where any routing decision is half wrong.

## The dial

Every row asks the same five questions about the same 101 tickets. Only what happens to the answers
changes.

```
posture                     auto   human  false yes    missed    cost    team
--------------------------------------------------------------------------
no arguments                100%      0%          0         4      80     84%
stakes=low    lean=none      94%      6%          0         2      46     79%
stakes=low    lean=yes       97%      3%          0         0       3     79%
stakes=low    lean=no       100%      0%          0         8     160     79%
stakes=medium lean=none      92%      8%          0         0       8     76%
stakes=medium lean=yes       80%     20%          0         0      20     76%
stakes=medium lean=no        93%      7%          0         1      27     76%
stakes=high   lean=none      72%     28%          0         0      28     68%
stakes=high   lean=yes       40%     60%          0         0      61     67%
stakes=high   lean=no        83%     17%          0         0      17     68%

before.py: a hard 0.7 threshold, no third branch
threshold=0.7               100%      0%          -         8     160       -
```

`missed` is a churn risk decided as "no". Cost is `2` per needless escalation, `20` per miss, `1`
per human review — the same scale for every row.

Four things worth reading off it.

**`stakes="low", lean="yes"` costs `3`. The hard threshold costs `160`.** Both keep almost the whole
queue automatic; one of them misses eight churn risks and the other misses none, at the price of
sending three tickets out of a hundred to a person. The model is identical. Only the decision rule
differs.

**More caution is not better.** `stakes="high", lean="yes"` also misses nothing, and costs `61` —
twenty times the best row — because it sends 60% of the queue to a person at `1` each. The dial has
a minimum and it is not at either end, which is exactly why it is a dial and not a switch.

**Leaning the wrong way is worse than having no opinion.** `lean="no"` at low stakes reproduces the
hard threshold's result exactly: 8 missed, cost 160. A posture is a claim about which mistake hurts,
and getting that backwards costs more than saying nothing.

**The team column drops as stakes rise** — from 84% to 67%. That is `ask_human` doing its job on the
classification too: at higher stakes an unpeaked answer becomes `UNSURE` rather than a guess, and
`UNSURE` is counted here as "not the right team". Caution has a price on every judgment, not just
the headline one.

## Is any of this trustworthy?

The cost rule and the postures both take the model's probabilities at face value. `gut eval`
measures whether that is earned, using the dataset labels as ground truth:

```
predicate        kind     accuracy   Brier     ECE
refund_request   noul       97%      0.028    0.049    pass
cancel_threat    noul       96%      0.029    0.100    pass
owning_team      choice     84%      0.110    0.037    fail (accuracy)
bug_report       noul       83%      0.126    0.156    fail
urgency          score      74%      0.255    0.248    fail
```

Two of five pass. The honest reading:

**`urgency`'s confidence is worse than useless here.** A Brier score of `0.255` is worse than
answering `0.5` to everything, and the reliability table is close to inverted:

```
p 0.2-0.4   n=18   said 0.31   happened 0.89
p 0.4-0.6   n=24   said 0.51   happened 0.79
p 0.6-0.8   n=28   said 0.70   happened 0.46
p 0.8-1.0   n=31   said 0.90   happened 0.87
```

The bucket where the model was *least* sure was its most accurate. A `min_confidence` floor — which
is what `stakes` maps to for `rate` — would route away the answers most likely to be right. On this
task, do not use it.

**`owning_team`'s confidence holds up.** The lowest calibration error of the five (`0.037`) despite
the lowest accuracy: the model knows when it is guessing. The same mechanism, sound on one task and
unsound on another, which is the whole argument for measuring instead of assuming.

**The yes/no probabilities rank well but are not centred.** `cancel_threat` is under-confident in
the middle (said `0.28`, happened `0.67`; said `0.52`, happened `1.00`); `bug_report` is
over-confident at the top (said `0.92`, happened `0.75`). Good enough to act on, not good enough to
read literally.

## Re-recording

```bash
TYPESAFE_API_KEY=... GUT_RECORD=1 python report.py
TYPESAFE_API_KEY=... GUT_RECORD=1 gut eval predicates/ \
    --cassette cassettes/predicates.json --model jev-1.13.0
```

505 live judgments, a few minutes. Replaying them takes 0.63 seconds.

## The bug this demo found

The first version of `after.py` used the handoff's own cost example — `cost_false_yes=2`,
`cost_false_no=50`, `cost_human=5` — and the report came back with **zero tickets sent to a human**.

Not a routing bug. The expected cost of asking a person is flat, while the cheaper of yes and no
peaks where those two cross, at `2 × 50 / 52 = 1.92`. A `cost_human` of `5` sits above that peak, so
at every probability either yes or no is cheaper and the `UNSURE` branch is unreachable. The code
was fine. The economics were wrong, and nothing said so.

`gut` warns about this now, and the posture presets make it structurally impossible — they are
defined as bands, and a band is reachable whenever its ends are in the right order. See D19 and D20
in [DECISIONS.md](../../DECISIONS.md).
