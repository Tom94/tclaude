#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import sys
import aiohttp

from io import StringIO
from partial_json_parser import loads as partial_loads

from common import prompt
from print import history_to_pretty_string, history_to_string

# Web search tool configuration
MAX_SEARCH_USES = 5
ALLOWED_DOMAINS = None  # Example: ["example.com", "trusteddomain.org"]
BLOCKED_DOMAINS = None  # Example: ["untrustedsource.com"]

# Anthropic API configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

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


async def get_anthropic_response(
    user_input,
    model,
    history=[],
    max_tokens=16384,
    enable_web_search=True,
    enable_code_exec=True,
    system_prompt=None,
    enable_thinking=False,
    thinking_budget=None,
    write_cache=False,
    on_response_update=None,
):
    """
    Send user input to Anthropic API and get the response using aiohttp.
    Uses async streaming for incremental output.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

    if write_cache:
        # See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#how-many-cache-breakpoints-can-i-use
        # We set the maximum to the docs-specified 4 minus one for the system prompt.
        MAX_NUM_CACHE_BREAKPOINTS = 4 - 1

        # First remove all but the last max-1 cache_control entries
        num_cache_breakpoints = 0
        for message in reversed(history):
            if not "content" in message or not isinstance(message["content"], list):
                continue

            for content in message["content"]:
                if "cache_control" in content:
                    num_cache_breakpoints += 1
                    if num_cache_breakpoints >= MAX_NUM_CACHE_BREAKPOINTS - 1:
                        del content["cache_control"]

        # Then set a new cache breakpoint for the last message
        history[-1]["content"][0]["cache_control"] = {"type": "ephemeral"}

    # Prepare request parameters
    params = {"model": model, "max_tokens": max_tokens, "messages": history, "stream": True}

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

    if tools:
        params["tools"] = tools
        params["tool_choice"] = {"type": "auto"}

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "interleaved-thinking-2025-05-14,code-execution-2025-05-22,files-api-2025-04-14",
    }

    # Make the async streaming request
    async with aiohttp.ClientSession() as session:
        async with session.post(ANTHROPIC_API_URL, headers=headers, json=params) as response:
            response.raise_for_status()

            # Track if we're currently in a thinking block
            tool_use_json = StringIO()

            message = {"role": "assistant", "content": []}
            content = message["content"]
            message_info = {}

            tokens = TokenCounter()

            # Parse SSE stream
            while True:
                try:
                    line = await response.content.readline()
                except aiohttp.ClientError as e:
                    print(f"Error reading response: {e}")
                    break

                if not line:
                    break

                if not line.startswith(b"data: "):
                    continue

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                kind = data.get("type")

                # Handle different types of events. During event handling *only* accumulate data. Don't print anything yet.
                if kind == "message_start":
                    message_info = data.get("message", {})

                elif kind == "message_delta":
                    delta = data.get("delta", {})
                    stop_reason = delta.get("stop_reason")
                    if stop_reason is not None:
                        message_info["stop_reason"] = stop_reason

                    usage = data.get("usage")
                    if usage:
                        turn_tokens = TokenCounter(
                            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
                            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                        )

                        tokens += turn_tokens

                elif kind == "message_end":
                    continue

                elif kind == "content_block_start":
                    index = data.get("index", 0)
                    content_block = data.get("content_block", {})

                    if index >= len(content):
                        content.extend([None] * (index - len(content) + 1))

                    content[index] = content_block

                elif kind == "content_block_delta":
                    index = data.get("index", 0)

                    if index >= len(content):
                        raise ValueError(f"Index {index} out of range for content list")

                    delta = data.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "thinking_delta":
                        thinking_text = delta.get("thinking", "")
                        if "thinking" not in content[index]:
                            content[index]["thinking"] = ""
                        content[index]["thinking"] += thinking_text

                    elif delta_type == "text_delta":
                        text = delta.get("text", "")
                        if "text" not in content[index]:
                            content[index]["text"] = ""
                        content[index]["text"] += text

                    elif delta_type == "citations_delta":
                        citation = delta.get("citation", {})
                        if "citations" not in content[index]:
                            content[index]["citations"] = []
                        content[index]["citations"].append(citation)

                    elif delta_type == "input_json_delta":
                        partial_json = delta.get("partial_json", "")
                        tool_use_json.write(partial_json)

                        if tool_use_json.tell() > 0:
                            try:
                                content[index]["input"] = partial_loads(tool_use_json.getvalue())
                            except:
                                pass

                if on_response_update is not None:
                    on_response_update(message, tokens)

    history.append(message)
    return message_info, message, tokens


async def main():
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
    parser.add_argument("--no-code-execution", action="store_true", help="Disable code execution capability (enabled by default)")
    parser.add_argument("--thinking", action="store_true", help="Enable Claude's extended thinking process")
    parser.add_argument("--thinking-budget", type=int, help="Number of tokens to allocate for thinking (min 1024)")

    args = parser.parse_args()

    # Get user input from arguments or stdin
    user_input = ""
    if args.input:
        user_input = " ".join(args.input)
    elif not sys.stdin.isatty() and not sys.stdin.closed:
        user_input = sys.stdin.read().strip()

    if not user_input:
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

    # The response is already printed during streaming, so we don't need to print it again
    _, message, _ = await get_anthropic_response(
        user_input,
        model=args.model,
        history=[],
        max_tokens=args.max_tokens,
        enable_web_search=not args.no_web_search,  # Web search is enabled by default
        enable_code_exec=not args.no_code_execution,  # Code execution is enabled by default
        system_prompt=system_prompt,
        enable_thinking=args.thinking,
        thinking_budget=args.thinking_budget,
    )

    print(history_to_string(prompt(False), [message]), end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
