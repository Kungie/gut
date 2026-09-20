# Exact costs, when a mistake has a price tag

[← docs index](README.md)


The postures are shorthand for a cost model. If you know the actual numbers, give them instead:

```python
import gut

gut.likely(email, "the customer threatens to cancel",
           cost_false_yes=2,     # a CSM spends twenty minutes on a calm customer
           cost_false_no=50,     # we lose the account
           cost_human=1)         # someone reads the ticket and decides
```

Given a probability `p`:

```
expected cost of saying YES        = (1 - p) · cost_false_yes
expected cost of saying NO         =      p  · cost_false_no
expected cost of asking a human    =         cost_human
```

`gut` takes the cheapest. Ties prefer `UNSURE`, then `NO`. With no human in the loop this reduces to

```
YES  ⟺  p > cost_false_yes / (cost_false_yes + cost_false_no)
```

which for `2` and `50` is `0.038`. Nobody guesses `0.038`.

`threshold=` and `unsure_band=(lo, hi)` are there if you already know the number you want. Mixing
exact costs with posture words in one call is an error — one of them would have to win silently.

### The trap in that formula

The expected cost of asking a person is **flat** in `p`, while the cheaper of yes and no *peaks*
where those two lines cross. Put `cost_human` above that peak and there is no probability at all
where a person is worth asking: the third branch you carefully wrote is unreachable, silently. The
ceiling is

```
cost_false_yes · cost_false_no / (cost_false_yes + cost_false_no)
```

`1.92` for `2` and `50`. A `cost_human` of `5` never fires; `1` does. `gut` warns when you cross it,
and `Policy.max_useful_cost_human` tells you where it is. This is not theoretical — it is the
mistake the demo shipped with, and the report showed it as *zero tickets sent to a human* with no
error anywhere.

The postures cannot do this to you. They are defined as bands and the costs derived, and a band is
reachable whenever its ends are in the right order.

Every policy will tell you where its boundaries are, rather than leaving you to work them out from
the costs:

```python
import gut

for preset in gut.presets():
    print(preset)

print(gut.policy(cost_false_yes=2, cost_false_no=50, cost_human=1).describe())
# no below 0.02, ask a person from 0.02 to 0.5, yes above 0.5
```
