# MCP server

[← docs index](README.md)

`gutfeel-mcp` gives an agent gut's judgments as tools. Claude Code, Claude Desktop, Cursor or any
other MCP client can then hand its judgment calls -- *is this spam, which team owns this, how
urgent is it* -- to a small, fast, cheap model, and get back an outcome and a probability.

It runs over stdio, and `uvx` fetches it on first use, so there is nothing to install by hand.

## Add it to your agent

With Jev, the model gut is built around, the only setting is your TypeSafe key. In Claude Code:

```bash
claude mcp add gut --env TYPESAFE_API_KEY=your-key -- uvx --from "gutfeel[mcp]" gutfeel-mcp
```

Every other client takes the same thing as JSON -- Claude Desktop in
`claude_desktop_config.json`, Cursor in `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "gut": {
      "command": "uvx",
      "args": ["--from", "gutfeel[mcp]", "gutfeel-mcp"],
      "env": { "TYPESAFE_API_KEY": "your-key" }
    }
  }
}
```

The key stays in your client's configuration and goes to TypeSafe's API, nowhere else.

## Pick the model

The server reads its model from the environment. With none of these set it uses Jev whenever
`TYPESAFE_API_KEY` is there, and otherwise every tool call answers with how to set one.

| variable | what it does |
|---|---|
| `TYPESAFE_API_KEY` | TypeSafe's Jev: the default whenever it is set |
| `GUT_BACKEND` | `jev`, `openai`, `ollama`, `zeroshot`, `transformers`, or `fake` for trying the tools without a model |
| `GUT_MODEL` | the model to ask; required for `openai` and `ollama` |
| `GUT_BASE_URL` | an OpenAI-compatible server's URL; `ollama` defaults to `http://localhost:11434/v1` |

A local Ollama model, with nothing leaving your machine:

```json
{
  "mcpServers": {
    "gut": {
      "command": "uvx",
      "args": ["--from", "gutfeel[mcp]", "gutfeel-mcp"],
      "env": { "GUT_BACKEND": "ollama", "GUT_MODEL": "qwen3:0.6b" }
    }
  }
}
```

The local NLI encoder, which needs PyTorch and so the `local` extra too:

```bash
claude mcp add gut --env GUT_BACKEND=zeroshot -- uvx --from "gutfeel[mcp,local]" gutfeel-mcp
```

The model is loaded on the first call rather than at startup, so the client's handshake is instant
even when the first answer takes a few seconds. [Backends](backends.md) says what each one is good at.

## The tools

| tool | asks | answers with |
|---|---|---|
| `likely` | is this claim true of the text? | `outcome` (yes, no, or unsure) and `p`, the probability of yes |
| `classify` | which of these options fits? | `value`, the chosen label, every option's probability and a `confidence` |
| `rate` | where on this scale is it? | `score`, which can fall between levels, the nearest `level` and a `confidence` |
| `each` | the same question about up to 1000 texts | `results`, one answer per text, in order |

They take the same arguments as the Python functions. `classify` takes its options either as a
list of labels or as `{label: description}`, since a description is what the model reads.
`ask_human`, `stakes` and `lean` are the words from
[Knowing when it doesn't know](knowing-when-it-doesnt-know.md), and are optional.

A `likely` answer, as the agent receives it:

```json
{
  "kind": "likely",
  "outcome": "yes",
  "p": 0.93,
  "question": "asks for a refund",
  "model": "jev-1.13.0",
  "policy": "yes above 0.5, no at or below it",
  "latency_ms": 41.0
}
```

`unsure` is only possible when the agent passes `ask_human: true`, and means the model could not
tell: the agent should read the text itself, or ask you.

## When an agent should reach for it

The agent calling these tools is a far larger model than the one answering them. For one subtle
judgment it can simply read the text itself. gut is worth it when the judgment is simple and
there is a lot of it: sorting an inbox, filtering search results, labelling every file in a diff,
routing a queue. There, one `each` call answers hundreds of texts at a small model's speed and
price, with the same model and the same question every time.

The server tells the agent this in its instructions, along with the one habit that matters most
for small models: phrase a question as a short, concrete claim about the text ("asks for a
refund"), never as an open question or a judgment of quality.

## What it can and cannot do

Every tool is read-only: asking a model changes nothing, and the same question about the same
text gets the same answer, from the cache after the first time. The text is sent to whichever model
you configured, and nowhere else.

Mistakes in a call come back as errors the agent can read and fix: an option list that would
collapse two options into one, a rubric of one level, `stakes` without `ask_human`. A program gets
a warning for that last one; an agent would never see a warning, so here it is an error.
