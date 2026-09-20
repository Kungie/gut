# Knowing when it doesn't know

[← docs index](README.md)


Three words say how careful to be. They are not thresholds — you never name a number.

```python
from gut import likely

likely(email, "the customer threatens to cancel", lean="yes")
likely(email, "the customer threatens to cancel", ask_human=True)
likely(email, "the customer threatens to cancel", stakes="high", lean="yes", ask_human=True)
```

- **`lean`** — `"yes"`, `"no"`, or nothing. Which mistake is worse, so which way to err.
- **`ask_human`** — whether `UNSURE` is possible at all. Off by default; nothing becomes
  three-valued behind your back.
- **`stakes`** — `"low"`, `"medium"`, `"high"`. How bad an automatic mistake is next to a person
  looking instead. Only widens the range where a person is asked, so it needs `ask_human=True`;
  `gut` warns if you pass it without.

The two knobs are independent on purpose: being more careful must never quietly change which way you
err. `lean` fixes where yes overtakes no; `stakes` only widens the band around it.

| `stakes` | `lean=None` | `lean="yes"` | `lean="no"` |
|---|---|---|---|
| **low** | ask 0.40 – 0.60 | ask 0.20 – 0.40 | ask 0.60 – 0.80 |
| **medium** | ask 0.25 – 0.75 | ask 0.125 – 0.625 | ask 0.375 – 0.875 |
| **high** | ask 0.10 – 0.90 | ask 0.05 – 0.85 | ask 0.15 – 0.95 |

Below the range it answers no, above it answers yes, inside it asks. Print the table yourself, so
you never have to take this page's word for it:

```python
import gut

for preset in gut.presets():
    print(preset)
```

`classify` and `rate` take `stakes` and `ask_human` too, where they become a confidence floor. They
do not take `lean`: there is no safer side of a four-way choice.

### `if` and `match`

`if decision:` works. `YES` is truthy, `NO` is falsy, and `UNSURE` is an explicit choice rather than
a silent one:

```python
import gut

gut.configure(on_unsure="raise")   # default: raises UnsureDecision
gut.configure(on_unsure="false")   # or "true", or a callback

with gut.on_unsure("false"):       # scoped; follows async tasks, never leaks across threads
    ...
```

The default raises because coercing `UNSURE` to `False` is the most dangerous option available: it
is what your existing `if` already does, so the third branch would vanish down the "no" path — the
exact bug this library exists to prevent, reintroduced as a default.

In a `match`, the outcomes must be **dotted**. A bare `case YES:` is not a value pattern in Python,
it is a capture pattern that matches anything, and the compiler rejects it:

```
SyntaxError: name capture 'YES' makes remaining patterns unreachable
```

```python
import gut
from gut import Outcome

match decision:
    case gut.YES: ...
    case gut.NO: ...

match decision:
    case Outcome.YES: ...
```
