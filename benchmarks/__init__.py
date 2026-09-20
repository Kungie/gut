"""Measuring `gut` on real, public, labelled data.

Everything else in this repository is measured against 101 tickets written for it. These are not.
Four public datasets with known labels, sampled once with a fixed seed, asked once, and scored
against what a developer would otherwise have written.

The point is not the model's accuracy -- that is Jev's number, and it belongs here only as the
baseline. The point is what `gut` does with those probabilities afterwards: how much error it
removes from the decisions it makes automatically, at what cost in coverage, and whether the same
posture means the same thing on three different tasks.

```bash
python -m benchmarks              # every dataset, offline from cassettes
python -m benchmarks clinc        # one
python -m benchmarks --record     # against the live model, re-recording
```
"""
