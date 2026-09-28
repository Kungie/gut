# Command line

[← docs index](README.md)

`gut` also runs from a shell, over lines of text: grep, but for meaning. It is the quickest way to
try a question on real data, and the simplest way for a coding agent to judge a thousand things
without reading them all itself.

```bash
git log --format=%s | gut filter "adds a new feature"
```

```text
Add gutfeel-mcp, so an agent can hand its judgment calls to a small model
Add the website: the Claude Design page, made to work
gut: 30 records · 6 yes, 24 no · 29 calls, $0 · 2.3s · MoritzLaurer/deberta-v3-xsmall-zeroshot-…
```

The lines the claim is true of go to standard output; the summary -- how many, what it cost, how
long, which model -- goes to standard error, so it never ends up in a pipe. `gut` comes with the
package, and `uvx` runs it without installing anything:

```bash
pip install gutfeel               # then: gut filter ...
uvx gutfeel filter "is spam" comments.txt
```

## Pick the model

The same environment variables as the [MCP server](mcp.md): Jev whenever `TYPESAFE_API_KEY` is set,
or `GUT_BACKEND` for another model, with `GUT_MODEL` where it takes one.

```bash
export TYPESAFE_API_KEY=...                                # Jev
export GUT_BACKEND=openrouter OPENROUTER_API_KEY=...       # Jev, through OpenRouter
export GUT_BACKEND=zeroshot                                # on your machine, free
export GUT_BACKEND=ollama GUT_MODEL=qwen3:0.6b             # a local Ollama server
```

`--backend` and `--model` override them for one run. A local model needs the `local` extra:
`uvx --from "gutfeel[local]" gut ...`.

## filter: the lines a claim is true of

```bash
gut filter "is an error worth a closer look" app.log
git ls-files | gut filter "retries failed requests" --read-files
gut filter "asks for a refund" inbox.txt --show no          # the ones that do not
```

It reads the files named, or standard input, one record per line; blank lines are skipped. With
`--read-files`, each line is a path and the file it names is judged instead, and the path is what
gets printed. Binary files and files over 200,000 characters are skipped, and listed.

Like grep, it exits with 1 when nothing matched.

## map: every answer, as JSON

Several questions about each line, asked together -- one request per line, whatever their number.
One JSON object comes out per line, shaped like the one below:

```bash
gut map tickets.txt \
  --likely  refund="asks for a refund" \
  --classify team=billing,platform,other \
  --rate    urgency="can wait,this week,right now"
```

```json
{"line": 1, "text": "I was charged twice…", "refund": {"outcome": "yes", "p": 0.97}, "team": {"outcome": "yes", "value": "billing", "confidence": 0.99}, "urgency": {"outcome": "yes", "score": 1.2, "level": "this week", "confidence": 0.71}}
```

Options can carry descriptions, which is what the model reads:
`--classify 'team={"billing": "invoices and refunds", "platform": "outages and bugs", "other": "anything else"}'`.
Each answer is a compact `Decision.to_dict()`, and the output is ready for `jq`.

## Knowing when it doesn't know

`--ask-human` makes `unsure` a possible answer, exactly as `ask_human=True` does in code, and
`--stakes` and `--lean` are the same words as in
[Knowing when it doesn't know](knowing-when-it-doesnt-know.md). With `filter`, `--show` picks which
answers are printed, so the lines a person should read are one command away:

```bash
gut filter "threatens to cancel" emails.txt --ask-human --show unsure
```

## A budget

`--max-cost` stops sending requests once a run has spent that many dollars, and says where to pick
up again, along these lines:

```bash
gut filter "is spam" comments.txt --max-cost 0.50
```

```text
gut: Stopped before the next call: $0.5003 spent of a $0.5 budget, over 41,702 calls.
gut: resume from line 41703: tail -n +41703 FILE | gut ...
```

It needs a backend that says what its calls cost: Jev through OpenRouter does, and a model on your
machine costs nothing, so `--max-cost 0` allows it and nothing else. TypeSafe's own API reports
tokens rather than dollars; with it, `--max-cost` stops after the first line and says why rather
than guess. In Python, the same budget is `gut.usage(max_cost=0.50)` -- see
[Caching and observability](caching-and-observability.md).

## Exit status

| status | meaning |
|---|---|
| 0 | done, and something was printed |
| 1 | `filter` matched nothing |
| 2 | a mistake on the command line, or no model configured |
| 3 | the budget stopped the run |

## For coding agents

An agent with a shell needs nothing else: `gut filter` over `git ls-files`, a log, or search results
settles in one command what would otherwise mean reading every item. Give it the skill, and it knows
when to reach for this:

```bash
npx skills add Kungie/gut --skill gut
```

With an MCP client, the [MCP server](mcp.md) offers the same calls as tools.
