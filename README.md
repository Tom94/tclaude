# tai — Terminal AI

**tai** aims to be a complete implementation of Claude in the terminal.

- Interactive chat with resumable sessions and auto-naming
- Extended thinking
- Web search and server-side code execution
- Files (upload, analyze, summarize w/ citations, etc.)
  - Images and PDFs are natively supported
  - Other files are analyzed by code execution
- Custom tools
- Automated caching (much lower cost!)

Coming soon:
- Server- and client-side MCP
- Download of code-generated files
- Claude-native client tools (text editor & computer use)


## Why use tai?

Unlike other tools that aim to support all kinds of LLMs, **tai** is designed specifically for Claude in the terminal.
As such, Claude-specific features like caching, Claude-native web search or code execution are implemented simply, correctly, and fully.

## Installation

```bash
git clone https://github.com/tom94/tai
pip install . --user
```

## Usage

Running `tai` opens a new chat session. Once you're done chatting, the session will be automatically named and saved as a .json file in the working directory.
Customize the sessions directory by passing `--sessions-dir <dir>` or by setting the `TAI_SESSIONS_DIR` environment variable.
Resume previous sessions with `-s <session name>.json`.

You can also prompt via stdin / pipes or with CLI arguments.

```bash
tai "How do I make great pasta?"
# or: echo "How do I make great pasta?" | tai
> Great pasta starts with quality ingredients and proper technique. ...
```

Or use an outward pipe to integrate `tai` into unix workflows
```bash
git diff --staged | tai "Write a commit message for this diff." | xargs -0 git commit -m
```

Upload files with `-f`
```bash
tai -f paper.pdf "Summarize this paper."
tai -f cat.png "Is this a dog?"
```

Claude will use web search and server-side code execution when the request demands it:
```bash
tai "Tell me the factorials from 1 through 20."
> [Uses Python to compute the answer.]

tai "What is the state of the art in physically based rendering?"
> [Uses web search and responds with citations.]
```

### Extended thinking

Enable thinking with `--thinking`
```bash
tai --thinking "Write a quine in C++."
> [Claude thinks about how to write a quine before responding.]
```

### Custom system prompt

If you'd like to customize the behavior of Claude (e.g. tell it to be more brief, or give it information about your background), create `~/.configs/tai/roles/default.md`.
The content of this file will be prepended as system prompt to all conversations.

If you'd like to load different system prompts on a case-by-case basis, you can pass them as
```bash
tai --role pirate.md "How do I make great pasta?"
> Ahoy there, matey! Ye be seekin' the secrets of craftin' the finest pasta this side of the Mediterranean, eh? ...
```

### Custom tools

Simply implement your tool as a function in `src/tai/tools.py` and it will be callable by Claude.
Make sure to [document the tools' function, parameters, and return values in detail](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use#best-practices-for-tool-definitions).

## License

GPLv3; see [LICENSE](LICENSE.txt) for details.
