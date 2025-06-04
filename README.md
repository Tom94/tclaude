# tai — Terminal AI

**tai** aims to be a complete implementation of Claude in the terminal.

Current features:
- Interactive chat with resumable sessions and auto-naming
- Extended thinking
- Files (upload, analyze, summarize w/ citations, etc.)
  - Images and PDFs are natively supported
  - Other files are analyzed by server-side code execution
- Server-side web search and code execution
- Custom tools
- Automated caching (much lower cost!)

Planned features:
- Server- and client-side MCP
- Download of code-generated files


## Why use tai?

Unlike other tools that aim to support all kinds of LLMs, tai is designed to be the best solution for chatting with Claude in the terminal.
One of the guiding principles of **tai** is to treat Claude API responses as the only source of truth rather than translating them to an interoperable custom format.
This means model-specific features like caching, Claude-native web search or code execution can be implemented simply, correctly, and fully.

## Installation

```bash
git clone https://github.com/tom94/tai
pip install . --user
```

## Usage

```bash
tai "How do I make great pasta?"
> Great pasta starts with quality ingredients and proper technique. ...
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

### Custom system prompt

If you'd like to customize the behavior of Claude (e.g. tell it to be more brief, or give it information about your background), create `~/.configs/tai/roles/default.md`.
The content of this file will be prepended as system prompt to all conversations.

If you'd like to load different system prompts on a case-by-case basis, you can pass them as
```bash
tai --role pirate.md "How do I make great pasta?"
> Ahoy there, matey! Ye be seekin' the secrets of craftin' the finest pasta this side of the Mediterranean, eh? ...
```

## License

GPLv3; see [LICENSE](LICENSE.txt) for details.
