# tai — Terminal AI (Claude only)

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

Unplanned:
- Batch messages API

## Why use tai?

Unlike other tools that aim to support all kinds of LLMs, tai is designed to be the best solution for Claude.
One of the guiding principles of **tai** is to treat Claude API responses as the only source of truth, whereas other tools need to translate
API responses to a common format in order to remain interoperable. Usually, information is lost in this conversion and model-specific
features like Claude-native web search or code execution either take a while to be implemented or are not implemented at all.

## License

GPLv3; see [LICENSE](LICENSE.txt) for details.
