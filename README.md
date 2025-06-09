# tclaude — Claude in the terminal

A complete implementation of Claude in the terminal.

Unlike other tools that aim to support all kinds of LLMs, **tclaude** is designed specifically for Claude.
As such, Claude-specific features like caching, Claude-native web search or code execution are implemented correctly and fully.

### Highlights

- Interactive chat with resumable sessions, extended thinking, and tool use
    - Built-in gounded web search, code execution, and file analysis
    - Remote [MCP server](https://mcpservers.org/) support (local server support coming soon)
- Implement any custom tool in just a few lines of Python
- Automatic caching (makes Claude up to 10x cheaper!)

## Installation

```bash
git clone https://github.com/tom94/tclaude
pip install . --user
```

Then set the `ANTHROPIC_API_KEY` environment variable to your [Claude API key](https://console.anthropic.com/settings/keys) and you are good to go.

## Usage

Running `tclaude` opens a new chat session. You can also directly pass a prompt to start a session.

```bash
tclaude "How do I make great pasta?"
# or: echo "How do I make great pasta?" | tclaude
> Great pasta starts with quality ingredients and proper technique. ...
```

Or use an outward pipe to integrate `tclaude` into unix workflows

```bash
git diff --staged | tclaude "Write a commit message for this diff." | xargs -0 git commit -m
```

Upload files with `-f`

```bash
tclaude -f paper.pdf "Summarize this paper."
tclaude -f cat.png "Is this a dog?"
```

Claude will use web search and server-side code execution when the request demands it:

```bash
tclaude "Tell me the factorials from 1 through 20."
> [Uses Python to compute the answer.]

tclaude "What is the state of the art in physically based rendering?"
> [Uses web search and responds with citations.]
```

### Sessions

Once you're done chatting, the session will be automatically named and saved as `<session-name>.json` in the working directory.

You can resume the session with `tclaude -s <session-name>.json`.

Customize where sessions are saved by passing `--sessions-dir <dir>` or by setting the `TCLAUDE_SESSIONS_DIR` environment variable.

### Extended thinking

Enable thinking with `--thinking`

```bash
tclaude --thinking "Write a quine in C++."
> [Claude thinks about how to write a quine before responding.]
```

### Custom system prompt

If you'd like to customize the behavior of Claude (e.g. tell it to be brief, or give it background information), create `~/.configs/tclaude/roles/default.md`.
The content of this file will be prepended as system prompt to all conversations.

If you'd like to load different system prompts on a case-by-case basis, you can pass them as

```bash
tclaude --role pirate.md "How do I make great pasta?"
> Ahoy there, matey! Ye be seekin' the secrets of craftin' the finest pasta this side of the Mediterranean, eh? ...
```

### Custom tools

Simply implement your tool as a function in `src/tclaude/tools.py` and it will be callable by Claude.
Make sure to [document](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use#best-practices-for-tool-definitions) the tools' function thoroughly such that Claude uses it optimally.

### MCP server support

To connect **tclaude** to remote [MCP servers](https://mcpservers.org/), create `~/.configs/tclaude/tclaude.toml` with the servers' address and authorization tokens:

```toml
[[mcp.remote_servers]]
name = "example-mcp"
url = "https://example-server.modelcontextprotocol.io/sse"
authorization_token = "<your-authorization-token>"
# Optional: restrict the tools that can be used with this MCP server
# tool_configuration.enabled = true
# tool_configuration.allowed_tools = [
#   "example_tool_1",
#   "example_tool_2",
# ]

[[mcp.remote_servers]]
name = "another-mcp-server"
url = "..."
```

## License

GPLv3; see [LICENSE](LICENSE.txt) for details.
