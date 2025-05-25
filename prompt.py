#!/usr/bin/env python3

import json
import os
import sys
import requests
import subprocess
from typing import Callable, Iterator

from io import StringIO
from partial_json_parser import loads as partial_loads

import common
from print import history_to_string

# Web search tool configuration
MAX_SEARCH_USES = 5
ALLOWED_DOMAINS = None  # Example: ["example.com", "trusteddomain.org"]
BLOCKED_DOMAINS = None  # Example: ["untrustedsource.com"]

# Anthropic API configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

VERTEX_API_KEY = os.getenv("VERTEX_API_KEY")
VERTEX_API_URL = "https://api.vertex.ai/v1/messages"
VERTEX_API_PROJECT = os.getenv("VERTEX_API_PROJECT")

IS_ATTY = sys.stdout.isatty()


def use_tools(messages: list[dict]) -> dict:
    """
    Use the tools specified in the messages to perform actions.
    This function is called when the model indicates that it wants to use a tool.
    """
    # TODO: implement
    return {}


class TokenCounter:
    def __init__(
        self, cache_creation_input_tokens: int = 0, cache_read_input_tokens: int = 0, input_tokens: int = 0, output_tokens: int = 0
    ):
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

    def cost(self, model: str) -> tuple[float, float, float, float]:
        cost_factor = 1.0
        if "opus" in model:
            cost_factor = 5.0
        elif "haiku" in model:
            cost_factor = 1.0 / 3.75
            if "3-haiku" in model:
                cost_factor *= 0.3

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


def get_gcp_access_token() -> str:
    cmd = ["gcloud", "auth", "print-access-token"]
    token = subprocess.check_output(cmd).decode("utf-8").strip()
    return token


def get_endpoint_vertex(model: str) -> tuple[str, dict, dict]:
    if not VERTEX_API_PROJECT:
        raise ValueError("VERTEX_API_PROJECT environment variables are required")

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {get_gcp_access_token()}",
        "Content-Type": "application/json",
    }

    url = f"https://aiplatform.googleapis.com/v1/projects/{VERTEX_API_PROJECT}/locations/global/publishers/anthropic/models/{model}:streamRawPredict"
    params = {
        "anthropic_version": "vertex-2023-10-16",
    }

    return url, headers, params


def get_endpoint_anthropic(model: str) -> tuple[str, dict, dict]:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "interleaved-thinking-2025-05-14,code-execution-2025-05-22,files-api-2025-04-14",
    }

    url = ANTHROPIC_API_URL
    params = {
        "model": model,
    }

    return url, headers, params


def stream_events(url: str, headers: dict, params: dict) -> Iterator[dict]:
    """
    Stream events using requests.
    """
    response = requests.post(url, headers=headers, json=params, stream=True)
    response.raise_for_status()
    for line in response.iter_lines():
        if line.startswith(b"data: "):
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line[6:]}")
                continue


def stream_response(
    model: str,
    history: list[dict] = [],
    max_tokens: int = 16384,
    enable_web_search: bool = True,
    enable_code_exec: bool = True,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    write_cache: bool = False,
    on_response_update: Callable[[list[dict], TokenCounter], None] | None = None,
) -> tuple[list[dict], TokenCounter, bool]:
    """
    Send user input to Anthropic API and get the response by streaming for incremental output.
    """

    if not history or history[-1].get("role", "") != "user":
        raise ValueError("The last message in history must be the user prompt.")

    if "3-5" in model:
        # Disable features not supported by the 3.5 models
        enable_web_search = False
        enable_code_exec = False
        enable_thinking = False
        max_tokens = min(max_tokens, 8192)

    url, headers, params = get_endpoint_anthropic(model)
    # url, headers, params = get_endpoint_vertex("claude-sonnet-4@20250514")

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

    # Make a copy is history in which messages don't contain anything but role and content. The online APIs aren't happy if they get more
    # data than that.
    history_to_submit = [{"role": m["role"], "content": m["content"]} for m in history]

    # Prepare request parameters
    params["max_tokens"] = max_tokens
    params["messages"] = history_to_submit
    params["stream"] = True

    # Add system prompt if provided. Always cache it.
    if system_prompt is not None:
        params["system"] = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

    # Add extended thinking if enabled
    if enable_thinking:
        thinking_config = {
            "type": "enabled",
            "budget_tokens": thinking_budget if thinking_budget is not None else max(1024, max_tokens // 2),
        }

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

    tool_use_json = {}
    messages = []

    tokens = TokenCounter()
    for data in stream_events(url, headers, params):
        kind = data.get("type", "")

        if "message" in kind:
            # Handle different types of events. During event handling *only* accumulate data. Don't print anything yet.
            if kind == "message_start":
                messages.append(data.get("message", {}))
                messages[-1]["role"] = "assistant"
                messages[-1]["content"] = []
                tool_use_json = {}  # TODO: handle this more gracefully

            elif kind == "message_delta":
                delta = data.get("delta", {})
                stop_reason = delta.get("stop_reason")
                if stop_reason is not None:
                    messages[-1]["stop_reason"] = stop_reason

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

            continue

        if not messages:
            raise ValueError("Content block before message in the response")

        content = messages[-1]["content"]
        if kind == "content_block_start":
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

                if index not in tool_use_json:
                    tool_use_json[index] = StringIO()

                tuj = tool_use_json[index]
                tuj.write(partial_json)

                if tuj.tell() > 0:
                    try:
                        content[index]["input"] = partial_loads(tuj.getvalue())
                    except:
                        pass

        if on_response_update is not None:
            on_response_update(messages, tokens)

    stop_reason = "unknown" if not messages else messages[-1].get("stop_reason")
    if stop_reason == "pause_turn":
        call_again = True
    elif stop_reason == "tool_use":
        call_again = True
        messages.append(use_tools(messages))
    else:
        call_again = False

    return messages, tokens, call_again


def main():
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    args = common.parse_args()

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
    history = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]

    call_again = True
    while call_again:
        messages, _, call_again = stream_response(
            model=args.model,
            history=history,
            max_tokens=args.max_tokens,
            enable_web_search=not args.no_web_search,  # Web search is enabled by default
            enable_code_exec=not args.no_code_execution,  # Code execution is enabled by default
            system_prompt=system_prompt,
            enable_thinking=args.thinking,
            thinking_budget=args.thinking_budget,
        )
        history.extend(messages)

    print(history_to_string(history[1:], pretty=False), end="", flush=True)


if __name__ == "__main__":
    main()
