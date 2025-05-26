# tai — Terminal AI (Claude only)

**tai** is a complete implementation of Claude in the terminal.
- Interactive chat with resumable sessions and auto-naming
- Extended thinking
- Files (upload, analyze, read, write, etc.)
- Server-side web search, code execution, and MCP
- Custom tools
- Automated caching (much lower cost!)

## Why use tai?

Unlike other tools that aim to support all kinds of LLMs, tai is designed to be the best solution for Claude.
One of the guiding principles of **tai** is to treat Claude API responses as the only source of truth, whereas other tools need to translate
API responses to a common format in order to remain interoperable. Usually, information is lost in this conversion and model-specific
features like Claude-native web search or code execution either take a while to be implemented or are not implemented at all.

## License

GPLv3; see [LICENSE](LICENSE.txt) for details.
