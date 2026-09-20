# Benchmarks

[← docs index](README.md)

Everything else in this repository is measured against 101 tickets written for it. This page is
not. Four public datasets with published labels, sampled once with a fixed seed, asked once.

**These are not `gut`'s accuracy figures.** Accuracy, Brier score and calibration error are the
model's — they belong here as the baseline, and reporting them as `gut`'s result would be taking
credit for someone else's work. What `gut` contributes is what happens to those probabilities
afterwards: how much error it removes from the decisions it makes on its own, at what cost in
coverage, and what it does with an input that fits nothing.

```bash
python -m benchmarks                 # all five, offline from the committed recordings
python -m benchmarks clinc           # one
python -m benchmarks --record        # against the live model; needs TYPESAFE_API_KEY
```

---

## Method

The protocol is ordinary. The discipline is the point.

- **One split per dataset, seed `20260920`,** stratified by label. Dev ≈ 200, test ≈ 500, drawn so
  they cannot overlap.
- **Question wording was written against dev and frozen** before any test set was scored. It was
  revised once, on dev evidence, and that revision is recorded in
  [D33](../DECISIONS.md). `python -m benchmarks --dev-only` exists so that judging the wording
  cannot accidentally show a test result.
- **Calibrators were fitted on dev and applied to test.** Never fitted on what they are scored on.
- **Model pinned** to `jev-1.13.0`, recorded in every result.
- **Every source is pinned and checksummed.** A benchmark whose inputs can change underneath it is
  not a benchmark.
- **Each example is asked exactly once.** A posture changes how an answer is acted on, never what
  was asked, so the whole sweep is computed afterwards from those probabilities — which also means
  every posture is scored on identical model answers.

Total: **3,544 requests, $0.060**, median 292–312 ms per request, p95 698–1348 ms.

> **Read this before the numbers.** CLINC150, the SMS Spam Collection, the NLBSE issue corpus
> and the TweetEval irony set are well known and public. They may be in the model's training data, and if they are, everything
> below is optimistic. That is an argument for measuring your own task with
> [`gut eval`](trusting-it.md), not for trusting these.

| dataset | licence | recording committed? |
|---|---|---|
| [CLINC150](https://github.com/clinc/oos-eval) — 151-way intent routing with out-of-scope | CC BY 3.0 per the dataset card | yes |
| [NLBSE'24 issues](https://github.com/nlbse2024/issue-report-classification) — real GitHub issues | none declared | **no** — re-record it |
| [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | CC BY 4.0 | yes |
| [TweetEval irony](https://github.com/cardiffnlp/tweeteval) — SemEval-2018 Task 3 | none declared | **no** — re-record it |

Banking77 and GoEmotions were skipped. See [D31](../DECISIONS.md) for how each licence
was checked and what follows from it.

---

## CLINC150 — knowing when it doesn't know

550 test queries: 450 across 150 intents, and **100 deliberately out of scope** (18.2%, matching
the canonical split). The enum has 150 members plus `OTHER`, described as "none of the above: the
request is outside what this assistant handles".

This is the clearest result on the page.

| | coverage | error on what it decided |
|---|---|---|
| **forced to pick an intent** (no `OTHER`) | 100% | **26.2%** |
| `classify(query, Intent)` — `OTHER` available | 83% | 11.5% |
| `ask_human=True, stakes="medium"` | 76% | **7.6%** |
| the same, calibrated | 63% | **3.8%** |
| `ask_human=True, stakes="high"`, calibrated | 35% | 1.6% |

*Jev's own numbers on this task: Brier 0.183, ECE 0.166; 0.152 and 0.061 after calibration.*
*For context, Larson et al. report 96.9% in-scope accuracy for a fine-tuned BERT.*

**A quarter of the forced answers are wrong, and most of that is recoverable by letting the model
decline.** Adding a catch-all member costs nothing and removes more than half the errors. Adding a
confidence floor removes half of what is left.

### The out-of-scope numbers

At `stakes="medium"`, of the 100 queries that fit no intent:

| | declined | via `OTHER` | via `UNSURE` | confidently wrong | in-scope wrongly declined |
|---|---|---|---|---|---|
| raw | **89** | 68 | 21 | 11 | 42 / 450 (9%) |
| calibrated | **98** | 43 | 55 | 2 | 107 / 450 (24%) |

Raw, it declines 89% of the out-of-scope traffic while losing 9% of the real traffic to a person.
Calibrated, it catches almost everything — at nearly three times the cost in false abstentions.
Neither is the right answer; which one you want is what `stakes` is for.

Two of the eleven "confidently wrong" cases are not really wrong. CLINC's out-of-scope set overlaps
its in-scope intents: `"give me the weather forecast for today"` is labelled out-of-scope while
`"give me the 7 day forecast"` is labelled `weather`. Measured out-of-scope recall is a floor
([D34](../DECISIONS.md)).

---

## SMS spam — the cleanest calibration check

500 messages, 13.4% spam. One `likely`, nothing else.

| | coverage | error |
|---|---|---|
| keyword filter (`free`, `win`, `prize`, premium numbers…) | 100% | **7.0%** |
| hard threshold at `0.7` | 100% | 2.4% |
| `likely(sms, "…")` with no arguments | 100% | 1.4% |
| `ask_human=True, stakes="medium"` | 95% | **0.4%** |
| `ask_human=True, stakes="high", lean="yes"` | 77% | **0.0%** |

*Jev: Brier 0.013, ECE 0.038. Almeida et al. report ~97.6% for an SVM trained on this corpus.*

The keyword filter is the README's "brittle rules" option, written in good faith, and it is **five
times worse** than the same model with no arguments at all. The posture then takes 1.4% down to
0.4% for five messages in a hundred going to a person, and to zero for twenty-three.

### Where calibration made things worse

SMS is nearly separable, so the isotonic fit learned a step and mapped **every test message to
exactly 0.0 or 1.0**. Average calibration improved — ECE 0.038 → 0.016 — and the third branch
disappeared, because no posture band contains a probability that is exactly 0 or 1. Every posture
went to 100% coverage and the error rate rose slightly, to 1.6%.

That is the two objectives diverging: a calibrator minimises average error, which rewards
confidence wherever the model is usually right, and `gut`'s value is concentrated where it is not.
`gut calibrate` now warns when a fit does this ([D35](../DECISIONS.md)). **Calibrate for the
decision you are making, not for the average.**

---

## Irony — where the model is weakest, and most useful anyway

This one was added to answer a specific question: *should `gut` have a `tone()` function?* The
answer turned on whether the model can read tone at all, so it was measured rather than argued
about. 500 tweets from SemEval-2018 Task 3, 48% ironic — near-balanced, so a coin flip errs 50%.

Sarcasm is the hardest thing on this page for a literal reader, and Jev's documented weakness is
that it reads instructions literally. Note the base rate: at 48% ironic, guessing errs 50%, so
there is nowhere for a weak result to hide.

| | coverage | error |
|---|---|---|
| the `#irony` / `#sarcasm` hashtag itself | 100% | **42.0%** |
| hard threshold at `0.7` | 100% | 28.4% |
| `likely(tweet, "…")` with no arguments | 100% | 28.4% |
| `ask_human=True, stakes="medium"` | 48% | 13.9% |
| `ask_human=True, stakes="high"` | 15% | **6.8%** |

*Jev: Brier 0.182, ECE 0.036. TweetEval reports 82.1 irony-class F1 for a fine-tuned BERTweet.*

**The model is bad at this and honest about it, and those are different things.** 28.4% error is
weak — a fine-tuned BERTweet is far better. But ECE 0.036 is the second-best calibration on this
page, better than the CLINC router that is three times more accurate. The model does not know
which tweets are ironic; it *does* know which ones it cannot read.

That is the whole mechanism in one result. `gut` cannot make a model accurate. It can spend the
model's honesty: declining 85% of the queue turns 28.4% error into **6.8%**, a 4× reduction. And
it does it on the task where the model comes closest to guessing — 28.4% against a chance rate of
50% is a narrower margin over a coin than anything else here, including `nlbse-kind`'s nominally
higher 30.7% on a three-way choice.

`nlbse-kind` is the control, and the comparison is the point: the model is about as inaccurate
there, but *over*confident (ECE 0.174 against 0.036), and the same move only reaches 18.8%.
Accuracy is not what `gut` spends. Honesty is.

Calibration added nothing here (Brier 0.182 → 0.186) for the same reason: there was little
dishonesty left to correct.

### The hashtag check

SemEval built this corpus by searching for `#irony` and `#sarcasm`, so before trusting any of the
above: 14.1% of the ironic tweets still carry one of those tags, against 5.0% of the plain ones.
The leak is real but small, and the baseline row measures it — reading the tag off the text errs
**42%**, worse than answering at random on a balanced set. Whatever the model is doing, it is not
that.

### So, `tone()`?

No. `tone(msg, "sarcastic")` is `likely(msg, "…is sarcastic")` with a narrower vocabulary, and
adding it invites `sentiment()`, `urgency()` and `intent()` behind it. Tone is a *subject*, not a
shape of question — and this result is the argument for keeping it that way: the thing worth
knowing is not a helper name, it is that this question needs `stakes="high"` to be usable at all.
A `tone()` returning a bare label would have hidden exactly that. See
[D36](../DECISIONS.md).

---

## NLBSE'24 GitHub issues

501 real issues from five projects, balanced across bug / feature / question. Two questions about
the same issue: is it a bug report, and which kind is it.

### `likely(issue, "…reports a defect…")`

| | coverage | error | false yes | false no |
|---|---|---|---|---|
| regex (`bug`, `error`, `crash`, `exception`…) | 100% | **44.1%** | 177 | 44 |
| hard threshold at `0.7` | 100% | 21.2% | 84 | 22 |
| no arguments | 100% | 26.3% | 120 | 12 |
| `ask_human=True, stakes="medium", lean="no"` | 69% | 14.4% | 44 | 6 |
| `ask_human=True, stakes="high", lean="no"` | 36% | **7.1%** | 12 | 1 |
| calibrated, `stakes="high", lean="yes"` | 52% | **7.3%** | 12 | 7 |

*Jev: Brier 0.190, ECE 0.211; 0.138 and 0.053 calibrated.*

**`lean` can hurt.** On this task the model already over-predicts "bug" — 120 false positives
against 12 false negatives at no arguments — so `lean="yes"` makes it worse (32.1% error at low
stakes against 24.2% with no lean) and `lean="no"` helps. A posture is a claim about which mistake
is worse *for you*; it is not a dial that only goes one way.

### `classify(issue, IssueKind)`

| | coverage | error |
|---|---|---|
| always answer | 100% | 30.7% |
| `ask_human=True, stakes="high"` | 76% | 23.0% |
| the same, calibrated | 55% | **18.8%** |

*Jev: Brier 0.222, ECE 0.174; 0.198 and 0.076 calibrated. NLBSE'24 baselines report ~0.87 F1 for a
fine-tuned RoBERTa.*

**This is the weakest result on the page** and worth stating plainly: three-way issue routing is
genuinely hard, the model is wrong about 30% of the time, and abstention improves that to 19% only
by declining nearly half the queue. A trained RoBERTa does much better. If you have labels and the
volume to justify training, train something.

### Batching, on real data

The two judgments above are about the *same* issue, so they are a single request:

```
1002 judgments about 501 issues
  asked separately   1002 requests
  asked together      501 requests   (50% fewer)
```

At the measured median of 295 ms, that is 148 seconds of serial latency not spent, and the issue
text billed once instead of twice.

---

## Does a posture mean the same thing everywhere?

This is `gut`'s most distinctive claim, and the honest answer is *it depends what you thought it
promised*.

`stakes="medium"` puts a person in the loop for `p` between 0.25 and 0.75, so anything it decides
automatically has at least 0.25 of margin. If the probabilities are honest, that **bounds** the
error rate at 25%. It cannot equalise it: how much error you actually see depends on where a task's
probability mass sits, and a spam filter and a 151-way router do not have their mass in the same
places.

| dataset | raw coverage | raw error | calibrated coverage | calibrated error | bound held? |
|---|---|---|---|---|---|
| clinc | 76% | 7.6% | 63% | 3.8% | yes |
| nlbse-bug | 75% | 20.2% | 71% | 10.6% | yes |
| nlbse-kind | 83% | 24.9% | 74% | 22.3% | yes |
| irony | 48% | 13.9% | 39% | 12.2% | yes |
| sms | 95% | 0.4% | 100% | 1.6% | yes |

**The bound held everywhere.** The spread did not narrow much — 24.5% raw, 20.7% calibrated — and
it was never going to. `stakes="medium"` is a promise about margin, not about outcomes.

---

## What this does and does not show

**Where `gut` wins.**

- Against what a developer actually writes: the keyword filters lose by 5× (SMS) and 2× (issues),
  and a hard `0.7` threshold is beaten on every dataset by a posture that declines a slice of the
  queue.
- Letting a classifier say "none of these" is worth more than any tuning: CLINC's error falls from
  26.2% to 11.5% for the price of one extra enum member.
- Most clearly where the model is *weakest*: irony goes 28.4% → 6.8%, because being wrong often
  and being overconfident are independent, and `gut` only needs the second one to be false.
- No training, no labelled data, no model to host. 3,544 judgments for 6.0 cents.
- Two judgments about one subject cost one request.

**Where it loses.**

- To trained models on accuracy, and not narrowly. A fine-tuned BERT gets 96.9% in-scope on CLINC
  against roughly 92% here; RoBERTa gets ~0.87 F1 on the issues against ~0.81. If you have labelled
  data and volume, train something.
- On genuinely hard multiway routing, abstention buys less than you would hope: NLBSE's three-way
  task still errs 19% of the time after declining 45% of the queue.
- Calibration is not a free improvement. On SMS it removed the third branch entirely, and on irony
  it did nothing at all.
- Coverage can collapse. Irony's 6.8% is real, but it is 6.8% of the 15% of tweets the model was
  willing to call. If you need an answer for every input, this page offers you 28.4%.

**What has not been tested.** Every number here comes from one model. The strongest evidence that
`gut`'s value is in `gut` rather than in Jev would be repeating this against a second backend — an
LLM exposing token log-probabilities, or a small local model reading option probabilities. That is
a real piece of work and it has not been done; it is the next step, recorded as such in
[DECISIONS.md](../DECISIONS.md).

---

## Reproducing

CLINC and SMS replay from committed recordings with no API key:

```bash
python -m benchmarks clinc sms
```

The NLBSE and irony recordings are not in the repository — the issue text and the tweets are
third-party content under no declared licence. Regenerate them with a key:

```bash
TYPESAFE_API_KEY=... python -m benchmarks nlbse-bug nlbse-kind irony --record
```

Machine-readable results, including the full posture sweeps and reliability tables, are in
[`benchmarks/results.json`](../benchmarks/results.json).
