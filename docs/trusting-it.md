# Knowing whether to trust it

[← docs index](README.md)


None of the above is worth anything if the probabilities are not. Write down the cases you care
about:

```yaml
# predicates/cancel_threat.yaml
question: "the customer threatens to cancel"
min_accuracy: 0.9
examples:
  - text: "If this happens again I'm cancelling my subscription."
    expected: yes
  - text: "How do I cancel my subscription? I want to downgrade."
    expected: no
  - text: "Third outage this month. We're evaluating alternatives."
    expected: yes
```

```bash
gut eval predicates/          # or: pytest --gut-evals predicates/
```

```
PASS  cancel_threat  (noul)
        accuracy   96%  (97/101)   min 85%
        Brier      0.029   0 is perfect, 0.25 is a coin flip
        cal. error 0.100
        p 0.0-0.2   n=85   said 0.07   happened 0.00
        p 0.2-0.4   n=3    said 0.28   happened 0.67
        p 0.4-0.6   n=6    said 0.52   happened 1.00
        p 0.8-1.0   n=7    said 0.94   happened 1.00
```

Accuracy says it picked the right side. **Brier score** and **calibration error** say whether the
number attached to that answer was worth anything, and the table says where it wasn't. Below 30
examples the report says the numbers are noise rather than pretending otherwise.

`--json` for CI, `--plot out.png` for a reliability diagram (`pip install "gut[plot]"`), and
`max_ece:` in a file to fail on calibration as well as accuracy. Pass and fail still hang on
`min_accuracy` alone unless you opt in.

### What it found

Five predicates, 101 labelled tickets, `jev-1.13.0`. Two of five pass, and the failures are the
interesting part.

> **What these numbers are.** The 101 tickets and every label on them — including which 26 are
> marked ambiguous — were written synthetically for this repository. So these measure **agreement
> with those labels**, not real-world performance, and the difficulty of the set is a judgement
> the author made rather than one the world imposed. They are useful for showing what the tools
> surface and for catching regressions; they are not evidence about how Jev behaves on your data.
> Running this against a real public dataset is a next step, noted in
> [DECISIONS.md](../DECISIONS.md).

| predicate | kind | accuracy | Brier | ECE | |
|---|---|---|---|---|---|
| refund_request | noul | 97% | 0.028 | 0.049 | pass |
| cancel_threat | noul | 96% | 0.029 | 0.100 | pass |
| owning_team | choice | 84% | 0.110 | 0.037 | fail (accuracy) |
| bug_report | noul | 83% | 0.126 | 0.156 | fail |
| urgency | score | 74% | 0.255 | 0.248 | fail |

**`urgency`'s confidence is worse than useless on this task.** Brier `0.255` is worse than answering
`0.5` to everything, and its table is close to inverted — the bucket where the model was least sure
was its most accurate. A `min_confidence` floor, which is what `stakes` maps to for `rate`, would
route away the answers most likely to be right. **Do not use `stakes` on `rate` for a task like this
without measuring first.**

**`owning_team`'s confidence holds up** — the lowest calibration error of the five despite the
lowest accuracy. Same mechanism, sound on one task, unsound on another. That is the argument for
measuring rather than assuming, and it is why `gut eval` exists.

**The yes/no probabilities rank well but are not centred.** `cancel_threat` is under-confident in
the middle, `bug_report` over-confident at the top. Good enough to act on, not good enough to read
literally — which means a preset band drawn at `0.25–0.75` does not sit where you would assume on
either distribution.

### Fixing it

A measurement you can't act on is just bad news. `gut calibrate` fits a correction per predicate
from the same files, and reports what it bought **out of fold** — each point corrected by a
calibrator that never saw it — so the improvement is not self-graded:

```bash
gut calibrate predicates/ --cassette tape.json --model jev-1.13.0 --out calibration.json
```

```
  cancel_threat      brier 0.029 -> 0.006   ece 0.100 -> 0.004   (out-of-fold)
                     isotonic, 30 breakpoints
  urgency            brier 0.255 -> 0.197   ece 0.248 -> 0.061   (out-of-fold)
                     isotonic, 4 breakpoints
  bug_report         brier 0.126 -> 0.095   ece 0.156 -> 0.082   (out-of-fold)
                     in-sample ece would have read 0.007; that gap is the overfit
  owning_team        brier 0.110 -> 0.124   ece 0.037 -> 0.058   (out-of-fold)
                     DROPPED: this makes calibration worse out of fold (0.037 -> 0.058).

4 correction(s) written to calibration.json
dropped for making things worse: owning_team
```

```python
import gut
from gut import CalibrationSet

gut.configure(calibration=CalibrationSet.load("calibration.json"))
```

Four things that file will not let you get wrong.

**A correction that loses is not shipped.** `owning_team` was already the best-calibrated of the
five; fitting on a hundred examples added noise and nothing else. Shipping that is strictly worse
than shipping nothing, so it is dropped unless you pass `--keep-all`.

**The in-sample number is a lie and the tool says so.** `bug_report` reads `0.007` in sample against
`0.082` out of fold. That gap is isotonic fitting noise, and it is printed rather than averaged
away.

**A correction cannot fix a ranking problem.** `urgency`'s calibration error falls from `0.248` to
`0.061` and its accuracy does not move, because the fit flattens a signal that does not rank. That
is the right outcome — the number becomes honestly uninformative instead of confidently wrong — and
it is why isotonic is the default: given an inverted relationship, the best non-decreasing fit is a
constant, which says "I have no information" rather than dressing noise up as a probability.

**Corrections are per question, and per model.** A fit for "is this a churn threat" says nothing
about "is this a bug report". A question with no entry is left alone, and using a correction against
a model it was not fitted on warns once.

Decisions keep both numbers: `decision.p` is what the rule acted on, `decision.raw_p` is what the
model said. The cache stores the raw answer, so refitting invalidates nothing.

### Record once, replay forever

```bash
GUT_RECORD=1 gut eval predicates/ --cassette tape.json --model jev-1.13.0
gut eval predicates/ --cassette tape.json --model jev-1.13.0     # 0.63s, offline
```

Entries are keyed by the subject and the question, not by the request they travelled in, so
regrouping questions — adding `@semantic`, say — does not invalidate a recording. In replay an
unrecorded question is an error that names itself; quietly reaching for the network would turn one
forgotten re-record into a suite that passes on your laptop, fails in CI, and bills you either way.

> ⚠️ **A cassette holds whatever you asked about, verbatim.** Deliberate: a reviewer should be able
> to read a diff and see what changed about the model's behaviour. But recording against real
> customer tickets and committing the file commits customer text to your repository. Record against
> fixtures you are happy to publish, or keep the cassette out of version control.

`FakeBackend` covers development with no key at all, and this project's own suite passes without one.
