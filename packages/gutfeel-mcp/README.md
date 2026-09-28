# gutfeel-mcp

gut's judgment calls as MCP tools, for Claude Code, Claude Desktop, Cursor and any other MCP client:
`likely`, `classify`, `rate` and `each`, answered by a small, fast, cheap model, with YES, NO or
UNSURE.

```bash
claude mcp add gut --env TYPESAFE_API_KEY=your-key -- uvx gutfeel-mcp
```

Or, in any client's JSON configuration:

```json
{
  "mcpServers": {
    "gut": {
      "command": "uvx",
      "args": ["gutfeel-mcp"],
      "env": { "TYPESAFE_API_KEY": "your-key" }
    }
  }
}
```

The model comes from the environment: TypeSafe's Jev whenever `TYPESAFE_API_KEY` is set, or
`GUT_BACKEND` for another -- `openrouter`, `ollaya`, `ollama`, `openai`, or `zeroshot` for a local
model (`uvx --with "gutfeel[local]" gutfeel-mcp`).

Documentation: https://gutpy.dev/docs/mcp.html · The library: https://pypi.org/project/gutfeel/

<!-- mcp-name: io.github.Kungie/gut -->
