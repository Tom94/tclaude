#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import sys

from anthropic import Anthropic
from io import StringIO
from partial_json_parser import loads as partial_loads

from common import prompt
from print import history_to_pretty_string

# Web search tool configuration
MAX_SEARCH_USES = 5
ALLOWED_DOMAINS = None  # Example: ["example.com", "trusteddomain.org"]
BLOCKED_DOMAINS = None  # Example: ["untrustedsource.com"]

# Initialize the Anthropic client
CLIENT = Anthropic()

IS_ATTY = sys.stdout.isatty()


class TokenCounter:
    def __init__(self, cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=0, output_tokens=0):
        self.cache_creation = cache_creation_input_tokens
        self.cache_read = cache_read_input_tokens
        self.input = input_tokens
        self.output = output_tokens

    def __add__(self, other):
        result = TokenCounter()
        result.cache_creation = self.cache_creation + other.cache_creation
        result.cache_read = self.cache_read + other.cache_read
        result.input = self.input + other.input
        result.output = self.output + other.output
        return result

    def cost(self, model: str):
        cost_factor = 1.0
        if "opus" in model:
            cost_factor = 5.0
        elif "haiku" in model:
            cost_factor = 1.0 / 3.75

        # See https://docs.anthropic.com/en/docs/about-claude/models/overview#model-pricing
        price_per_minput_cache_creation = 3.75 * cost_factor
        price_per_minput_cache_read = 0.3 * cost_factor
        price_per_minput = 3.0 * cost_factor
        price_per_moutput = 15.0 * cost_factor

        cache_creation_cost = (self.cache_creation / 1000000) * price_per_minput_cache_creation
        cache_read_cost = (self.cache_read / 1000000) * price_per_minput_cache_read
        input_cost = (self.input / 1000000) * price_per_minput
        output_cost = (self.output / 1000000) * price_per_moutput

        return cache_creation_cost, cache_read_cost, input_cost, output_cost

    def total_cost(self, model: str) -> float:
        cache_creation_cost, cache_read_cost, input_cost, output_cost = self.cost(model)
        return cache_creation_cost + cache_read_cost + input_cost + output_cost

    def print_tokens(self):
        print(f"Tokens: cache_creation={self.cache_creation} cache_read={self.cache_read} input={self.input} output={self.output}")

    def print_cost(self, model: str):
        cache_creation_cost, cache_read_cost, input_cost, output_cost = self.cost(model)
        print(
            f"Cost: cache_creation=${cache_creation_cost:.2f} cache_read=${cache_read_cost:.2f} input=${input_cost:.2f} output=${output_cost:.2f}"
        )


def get_anthropic_response(
    user_input,
    model,
    history=[],
    max_tokens=16384,
    enable_web_search=True,
    enable_code_exec=True,
    system_prompt=None,
    enable_thinking=False,
    thinking_budget=None,
    enable_printing=True,
    is_repl=False,
    write_cache=False,
):
    """
    Send user input to Anthropic API and get the response using the Anthropic Python client.
    Uses streaming for incremental output.
    """
    history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

    if write_cache:
        # See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#how-many-cache-breakpoints-can-i-use
        MAX_NUM_CACHE_BREAKPOINTS = 4

        # First remove all but the last max-1 cache_control entries
        num_cache_breakpoints = 0
        for message in reversed(history):
            if not "content" in message or not isinstance(message["content"], list):
                continue

            for content in message["content"]:
                if "cache_control" in content:
                    # num_cache_breakpoints += 1
                    # if num_cache_breakpoints >= MAX_NUM_CACHE_BREAKPOINTS - 1:
                    del content["cache_control"]

        # Then set a new cache breakpoint for the last message
        history[-1]["content"][0]["cache_control"] = {"type": "ephemeral"}

    # Prepare request parameters
    params = {"model": model, "max_tokens": max_tokens, "messages": history}

    # Add system prompt if provided. Always cache it.
    if system_prompt:
        params["system"] = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

    # Add extended thinking if enabled
    if enable_thinking:
        thinking_config = {"type": "enabled", "budget_tokens": thinking_budget if thinking_budget else max(1024, max_tokens // 2)}
        params["thinking"] = thinking_config

    # Add web search tool if enabled
    tools = []
    if enable_web_search:
        web_search_tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_SEARCH_USES}

        # Add domain filters if specified
        if ALLOWED_DOMAINS:
            web_search_tool["allowed_domains"] = ALLOWED_DOMAINS
        elif BLOCKED_DOMAINS:
            web_search_tool["blocked_domains"] = BLOCKED_DOMAINS

        tools.append(web_search_tool)

    if enable_code_exec:
        code_exec_tool = {"type": "code_execution_20250522", "name": "code_execution"}
        tools.append(code_exec_tool)

    params["tools"] = tools
    params["tool_choice"] = {"type": "auto"}
    params["extra_headers"] = {"anthropic-beta": "interleaved-thinking-2025-05-14,code-execution-2025-05-22,files-api-2025-04-14"}

    text_response = ""
    with CLIENT.messages.stream(**params) as stream:
        # Track if we're currently in a thinking block
        in_thinking_section = False

        def print_stream(text):
            nonlocal text_response
            text_response += text
            if enable_printing:
                print(text, end="", flush=True)

        # Process each event in the stream
        current_content_block = None
        current_server_tool = None
        tool_use_json = StringIO()
        last_tool_input_length = 0

        for event in stream:
            # Handle different event types
            if event.type == "content_block_start":
                if current_content_block is not None and current_content_block != event.content_block.type:
                    print_stream("\n\n")

                current_content_block = event.content_block.type

                # Print the start of a new content block
                if event.content_block.type == "thinking":
                    print_stream("# Thought process\n\n")
                    in_thinking_section = True
                elif event.content_block.type == "text":
                    if in_thinking_section:
                        print_stream("# Thoughtful response\n\n")
                        in_thinking_section = True
                elif event.content_block.type == "server_tool_use":
                    tool_use_json = StringIO()
                    last_tool_input_length = 0

                    current_server_tool = event.content_block.name
                    if current_server_tool == "web_search":
                        print_stream(f"# Searching the web for: ")
                    elif current_server_tool == "code_execution":
                        print_stream(f"# Executing code...\n\n")
                        print_stream(f"```python\n")
                elif event.content_block.type == "web_search_tool_result":
                    print_stream(f"## Web search results\n\n")
                    for c in event.content_block.content:
                        print_stream(f"{c.title} - {c.url}\n")
                elif event.content_block.type == "code_execution_tool_result":
                    stdout = event.content_block.content.get("stdout", "")
                    stderr = event.content_block.content.get("stderr", "")
                    if stdout:
                        print_stream(f"## stdout\n\n{event.content_block.content['stdout']}")
                    if stderr:
                        print_stream(f"## stderr\n\n{event.content_block.content['stderr']}")

            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    print_stream(event.delta.thinking)
                elif event.delta.type == "text_delta":
                    print_stream(event.delta.text)
                elif event.delta.type == "input_json_delta":
                    tool_use_json.write(event.delta.partial_json)
                    if tool_use_json.tell() > 0:
                        tool_use: dict = partial_loads(tool_use_json.getvalue())  # type: ignore

                        tool_input = ""
                        if current_server_tool == "web_search":
                            tool_input = tool_use.get("query", "")
                        elif current_server_tool == "code_execution":
                            tool_input = tool_use.get("code", "")

                        if len(tool_input) > last_tool_input_length:
                            print_stream(tool_input[last_tool_input_length:])
                            last_tool_input_length = len(tool_input)

            elif event.type == "content_block_stop":
                if event.content_block.type == "thinking":
                    pass
                elif event.content_block.type == "text":
                    pass
                elif event.content_block.type == "server_tool_use":
                    tool_use = json.loads(tool_use_json.getvalue())
                    if event.content_block.name == "web_search":
                        pass
                    elif event.content_block.name == "code_execution":
                        print_stream(f"\n```")

        # Get the final message with all content
        final_message = stream.get_final_message()

        citations = []
        if enable_web_search:
            # Extract citations from the response
            for content_block in final_message.content:
                if content_block.type == "text" and hasattr(content_block, "citations") and content_block.citations:
                    for citation in content_block.citations:
                        if hasattr(citation, "type") and citation.type == "web_search_result_location":
                            citations.append({"url": citation.url, "title": citation.title, "cited_text": citation.cited_text})

        if citations:
            print_stream("\n\nSources:")
            for i, citation in enumerate(citations, 1):
                print_stream(f"\n{i}. {citation['title']} - {citation['url']}")

        history.append(
            {"role": "assistant", "content": [block.model_dump(exclude_unset=True, exclude_none=True) for block in final_message.content]}
        )

        tokens = TokenCounter(
            cache_creation_input_tokens=final_message.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=final_message.usage.cache_read_input_tokens or 0,
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
        )
        return text_response, tokens


def main():
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    parser = argparse.ArgumentParser(description="Chat with Anthropic's Claude API")
    parser.add_argument("input", nargs="*", help="Input text to send to Claude")
    parser.add_argument("-s", "--session", help="Path to session file for conversation history")
    parser.add_argument("-r", "--role", help="Path to a markdown file containing a system prompt")
    parser.add_argument("-m", "--model", default="claude-sonnet-4-0", help="Anthropic model to use (default: claude-3.7-sonnet)")
    parser.add_argument("--max-tokens", type=int, default=2**14, help="Maximum number of tokens in the response (default: 16384)")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search capability (enabled by default)")
    parser.add_argument("--thinking", action="store_true", help="Enable Claude's extended thinking process")
    parser.add_argument("--thinking-budget", type=int, help="Number of tokens to allocate for thinking (min 1024)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Get user input from arguments or stdin
    user_input = ""
    is_repl = False
    if args.input:
        user_input = " ".join(args.input)
    elif not sys.stdin.isatty() and not sys.stdin.closed:
        user_input = sys.stdin.read().strip()
    else:
        is_repl = sys.stdin.isatty()

    if not user_input and not is_repl:
        print("No input provided.")
        return

    # Read system prompt from file if provided
    system_prompt = None
    if args.role:
        try:
            with open(args.role, "r") as f:
                system_prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading system prompt file: {e}")
            return

    # Initialize or load messages history
    history = []
    if args.session and os.path.exists(args.session):
        try:
            with open(args.session, "r") as f:
                history = json.load(f)
                print(history_to_pretty_string(prompt(True), history), "\n")
        except json.JSONDecodeError:
            print(f"Error: Could not parse session file {args.session}. Starting new session.")
            return

    total_tokens = TokenCounter()

    received_response = False

    # Initially, don't cache anything. The system prompt is always cached.
    write_cache = False

    try:
        while True:
            if is_repl:
                user_input = input(prompt(True)).strip()
                if not user_input:
                    continue

            # The response is already printed during streaming, so we don't need to print it again
            _, tokens = get_anthropic_response(
                user_input,
                model=args.model,
                history=history,
                max_tokens=args.max_tokens,
                enable_web_search=not args.no_web_search,  # Web search is enabled by default
                system_prompt=system_prompt,
                enable_thinking=args.thinking,
                thinking_budget=args.thinking_budget,
                is_repl=is_repl,
                write_cache=write_cache,
            )

            total_tokens += tokens

            # We heuristically set a new cache breakpoint when our next prompt (if short ~0 tokens) causes the cost of input to be larger
            # than that of cache reads.
            # TODO: If we just finished a web search, apparently something messy happens to the cache... should investigate
            tokens_if_short_follow_up = TokenCounter(
                cache_creation_input_tokens=0,
                cache_read_input_tokens=tokens.cache_read + tokens.cache_creation,
                input_tokens=tokens.input + tokens.output,
                output_tokens=0,
            )
            _, cache_read_cost, input_cost, _ = tokens_if_short_follow_up.cost(args.model)
            write_cache = cache_read_cost < input_cost

            if is_repl:
                # An empty line between each prompt
                print()
                print()

                if args.verbose:
                    tokens.print_tokens()
                    tokens.print_cost(args.model)
                    if write_cache:
                        print("Next prompt will be cached.")
                    print()

            received_response = True

            if not is_repl:
                break
    except KeyboardInterrupt:
        pass
    except EOFError:
        pass

    # Print stats and save session if in REPL mode
    if is_repl and received_response:
        # Save updated history if session file is specified
        session_name = args.session
        if session_name is None:
            print("Auto-naming session file...")
            session_name, tokens = get_anthropic_response(
                "Title this conversation with less than 30 characters. Respond with just the title and nothing else. Thank you.",
                model=args.model,
                history=history.copy(),  # Using a copy ensures we don't modify the original history
                max_tokens=30,
                enable_web_search=False,
                system_prompt=system_prompt,
                enable_thinking=False,
                enable_printing=False,
                is_repl=False,
            )

            total_tokens += tokens

            session_name = session_name.replace("\n", "-").replace(" ", "-").replace(":", "-").replace("/", "-").strip()
            session_name = "-".join(filter(None, session_name.split("-")))  # remove duplicate -

            date = datetime.datetime.now().strftime("%Y-%m-%d")
            session_name = f"{date}-{session_name}.json"

        print(f"Saving session as {session_name}...")
        with open(session_name, "w") as f:
            json.dump(history, f, indent=2)

        if args.verbose:
            total_tokens.print_tokens()
            total_tokens.print_cost(args.model)

        print(f"Total cost: ${total_tokens.total_cost(args.model):.2f}")


if __name__ == "__main__":
    main()
