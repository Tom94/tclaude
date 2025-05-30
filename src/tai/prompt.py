# tai -- Terminal AI
#
# Copyright (C) 2025 Thomas Müller <contact@tom94.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import asyncio
import contextlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from collections.abc import AsyncGenerator
from io import StringIO
from typing import Callable, cast

import aiohttp
from partial_json_parser import loads as partial_loads

from . import common
from .common import History, pplain
from .json import JSON, get, get_or, get_or_default
from .print import history_to_string

# Web search tool configuration
MAX_SEARCH_USES = 5
ALLOWED_DOMAINS = None  # Example: ["example.com", "trusteddomain.org"]
BLOCKED_DOMAINS = None  # Example: ["untrustedsource.com"]

# Anthropic API configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

VERTEX_API_KEY = os.getenv("VERTEX_API_KEY")
VERTEX_API_PROJECT = os.getenv("VERTEX_API_PROJECT")


def get_available_tools() -> dict[str, Callable[[object], object]]:
    """
    Dynamically import tools.py and extract all callable functions.
    Returns a dictionary mapping function names to their callable objects.
    """
    try:
        tools_module = importlib.import_module(".tools", package="tai")
        available_tools: dict[str, Callable[[object], object]] = {}

        for name, obj in inspect.getmembers(tools_module):
            if inspect.isfunction(obj) and not name.startswith("_"):
                available_tools[name] = obj

        return available_tools
    except ImportError:
        return {}


def get_tool_definitions() -> list[JSON]:
    """
    Generate tool definitions for Claude based on functions in tools.py.
    """
    available_tools = get_available_tools()
    tool_definitions: list[JSON] = []

    for name, func in available_tools.items():
        # Get function signature and docstring
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        # Parse parameters
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type
            param_desc = f"Parameter {param_name}"

            # Try to extract type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_type = "string"
                elif param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif hasattr(param.annotation, "__origin__"):
                    # Handle generic types like List[str]
                    if param.annotation.__origin__ == list:
                        param_type = "array"
                    elif param.annotation.__origin__ == dict:
                        param_type = "object"

            properties[param_name] = {"type": param_type, "description": param_desc}

            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        tool_def: JSON = {
            "name": name,
            "description": doc,
            "input_schema": {"type": "object", "properties": properties, "required": required},
        }

        tool_definitions.append(tool_def)

    return tool_definitions


async def use_tools(messages: History) -> dict[str, JSON]:
    """
    Use the tools specified in the messages to perform actions.
    This function is called when the model indicates that it wants to use a tool.
    """
    available_tools = get_available_tools()
    tool_results: list[dict[str, JSON]] = []
    tool_tasks: list[asyncio.Task[JSON] | None] = []

    # Find the last assistant message with tool use
    last_message = messages[-1] if messages else None
    if not last_message or last_message.get("role") != "assistant":
        return {"role": "user", "content": tool_results}

    # Process each content block that contains tool use. Tools are run in parallel.
    for content_block in get_or_default(last_message, "content", list[JSON]):
        if get(content_block, "type", str) == "tool_use":
            tool_name = get(content_block, "name", str)
            tool_input = get_or(content_block, "input", {})
            tool_use_id = get(content_block, "id", str)

            if tool_name and tool_name in available_tools:
                # Call the tool function with the provided input
                tool_use_task: asyncio.Task[JSON] = asyncio.create_task(available_tools[tool_name](**tool_input))
                tool_results.append({"type": "tool_result", "tool_use_id": tool_use_id})
                tool_tasks.append(tool_use_task)
            else:
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": f"Tool {tool_name} not found", "is_error": True}
                )
                tool_tasks.append(None)

    # Wait for all async tool calls to complete
    for tool_result, task in zip(tool_results, tool_tasks):
        if task is None:
            continue

        try:
            tool_result["content"] = str(await task)
        except (KeyboardInterrupt, asyncio.CancelledError, Exception) as e:
            if isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
                tool_result["content"] = "Tool execution was cancelled."
            else:
                tool_result["content"] = f"Error executing tool: {str(e)}"
            tool_result["is_error"] = True

    return {"role": "user", "content": tool_results}


class TokenCounter:
    def __init__(
        self, cache_creation_input_tokens: int = 0, cache_read_input_tokens: int = 0, input_tokens: int = 0, output_tokens: int = 0
    ):
        self.cache_creation: int = cache_creation_input_tokens
        self.cache_read: int = cache_read_input_tokens
        self.input: int = input_tokens
        self.output: int = output_tokens

    def __add__(self, other: "TokenCounter") -> "TokenCounter":
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
        pplain(f"Tokens: cache_creation={self.cache_creation} cache_read={self.cache_read} input={self.input} output={self.output}")

    def print_cost(self, model: str):
        cache_creation_cost, cache_read_cost, input_cost, output_cost = self.cost(model)
        pplain(
            f"Cost: cache_creation=${cache_creation_cost:.2f} cache_read=${cache_read_cost:.2f} input=${input_cost:.2f} output=${output_cost:.2f}"
        )


def get_gcp_access_token() -> str:
    cmd = ["gcloud", "auth", "print-access-token"]
    token = subprocess.check_output(cmd).decode("utf-8").strip()
    return token


def get_endpoint_vertex(model: str) -> tuple[str, dict[str, str], dict[str, JSON]]:
    if not VERTEX_API_PROJECT:
        raise ValueError("VERTEX_API_PROJECT environment variable must be set")

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {get_gcp_access_token()}",
        "Content-Type": "application/json",
    }

    url = f"https://aiplatform.googleapis.com/v1/projects/{VERTEX_API_PROJECT}/locations/global/publishers/anthropic/models/{model}:streamRawPredict"
    params: dict[str, JSON] = {
        "anthropic_version": "vertex-2023-10-16",
    }

    return url, headers, params


def get_endpoint_anthropic(model: str) -> tuple[str, dict[str, str], dict[str, JSON]]:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "interleaved-thinking-2025-05-14,code-execution-2025-05-22,files-api-2025-04-14",
    }

    url = "https://api.anthropic.com/v1/messages"
    params: dict[str, JSON] = {
        "model": model,
    }

    return url, headers, params


async def stream_events(url: str, headers: dict[str, str], params: JSON) -> AsyncGenerator[JSON]:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=params) as response:
            response.raise_for_status()
            async for line in response.content:
                if line.startswith(b"data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {line[6:]}")
                        continue


async def stream_response(
    model: str,
    history: History,
    max_tokens: int = 16384,
    enable_web_search: bool = True,
    enable_code_exec: bool = True,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    write_cache: bool = False,
    on_response_update: Callable[[History, TokenCounter], None] | None = None,
) -> tuple[History, TokenCounter, bool]:
    """
    Send user input to Anthropic API and get the response by streaming for incremental output.
    """
    if not history or get(history[-1], "role", str) != "user":
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
            for content_block in get_or_default(message, "content", list[dict[str, JSON]]):
                if "cache_control" in content_block:
                    num_cache_breakpoints += 1
                    if num_cache_breakpoints >= MAX_NUM_CACHE_BREAKPOINTS - 1:
                        del content_block["cache_control"]

        # Then set a new cache breakpoint for the last message
        last_message = get_or_default(history[-1], "content", list[dict[str, JSON]])
        if last_message:
            last_message[0]["cache_control"] = {"type": "ephemeral"}

    # Make a copy is history in which messages don't contain anything but role and content. The online APIs aren't happy if they get more
    # data than that.
    history_to_submit: list[JSON] = [{"role": get_or(m, "role", ""), "content": get_or_default(m, "content", list[JSON])} for m in history]

    # Prepare request parameters
    params["max_tokens"] = max_tokens
    params["messages"] = history_to_submit
    params["stream"] = True

    # Add system prompt if provided. Always cache it.
    if system_prompt is not None:
        params["system"] = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]

    # Add extended thinking if enabled
    if enable_thinking:
        thinking_config: JSON = {
            "type": "enabled",
            "budget_tokens": thinking_budget if thinking_budget is not None else max(1024, max_tokens // 2),
        }

        params["thinking"] = thinking_config

    # Add web search tool if enabled
    tools: list[JSON] = []

    if enable_web_search:
        web_search_tool: JSON = {"type": "web_search_20250305", "name": "web_search", "max_uses": MAX_SEARCH_USES}

        # Add domain filters if specified
        if ALLOWED_DOMAINS:
            web_search_tool["allowed_domains"] = ALLOWED_DOMAINS
        elif BLOCKED_DOMAINS:
            web_search_tool["blocked_domains"] = BLOCKED_DOMAINS

        tools.append(web_search_tool)

    if enable_code_exec:
        code_exec_tool: JSON = {"type": "code_execution_20250522", "name": "code_execution"}
        tools.append(code_exec_tool)

    # Add dynamically loaded tools from tools.py
    custom_tools = get_tool_definitions()
    tools.extend(custom_tools)

    if tools:
        params["tools"] = tools
        params["tool_choice"] = {"type": "auto"}

    tool_use_json: dict[int, StringIO] = {}
    messages: History = []
    tokens = TokenCounter()

    # Async generator cleanup: https://www.youtube.com/watch?v=N56Jrqc7SBk
    async with contextlib.aclosing(stream_events(url, headers, params)) as events:
        async for data in events:
            kind = get_or(data, "type", "")

            if "message" in kind:
                # Handle different types of events. During event handling *only* accumulate data. Don't print anything yet.
                if kind == "message_start":
                    messages.append(get_or(data, "message", {}))
                    messages[-1]["role"] = "assistant"
                    messages[-1]["content"] = []
                    tool_use_json = {}  # TODO: handle this more gracefully

                elif kind == "message_delta":
                    delta = get_or(data, "delta", {})
                    stop_reason = get(delta, "stop_reason", str)
                    if stop_reason is not None:
                        messages[-1]["stop_reason"] = stop_reason

                    usage = get(data, "usage", dict[str, JSON])
                    if usage:
                        turn_tokens = TokenCounter(
                            cache_creation_input_tokens=get_or(usage, "cache_creation_input_tokens", 0),
                            cache_read_input_tokens=get_or(usage, "cache_read_input_tokens", 0),
                            input_tokens=get_or(usage, "input_tokens", 0),
                            output_tokens=get_or(usage, "output_tokens", 0),
                        )

                        tokens += turn_tokens

                elif kind == "message_end":
                    continue

                continue

            if not messages:
                raise ValueError("Content block before message in the response")

            content: list[dict[str, JSON]] = get_or(messages[-1], "content", [])
            if kind == "content_block_start":
                index = get_or(data, "index", 0)
                content_block = get_or(data, "content_block", {})

                while index >= len(content):
                    content.append({})

                content[index] = content_block

            elif kind == "content_block_delta":
                index = get_or(data, "index", 0)

                if index >= len(content):
                    raise ValueError(f"Index {index} out of range for content list")

                delta = get_or(data, "delta", {})
                delta_type = delta.get("type")

                if delta_type == "thinking_delta":
                    thinking_delta = get_or(delta, "thinking", "")
                    if "thinking" not in content[index] or not isinstance(content[index]["thinking"], str):
                        content[index]["thinking"] = ""
                    thinking = cast(str, content[index]["thinking"])
                    content[index]["thinking"] = thinking + thinking_delta

                if delta_type == "signature_delta":
                    signature_delta = get_or(delta, "signature", "")
                    if "signature" not in content[index] or not isinstance(content[index]["signature"], str):
                        content[index]["signature"] = ""
                    signature = cast(str, content[index]["signature"])
                    content[index]["signature"] = signature + signature_delta

                elif delta_type == "text_delta":
                    text_delta = get_or(delta, "text", "")
                    if "text" not in content[index] or not isinstance(content[index]["text"], str):
                        content[index]["text"] = ""
                    text = cast(str, content[index]["text"])
                    content[index]["text"] = text + text_delta

                elif delta_type == "citations_delta":
                    citation = get_or(delta, "citation", {})
                    if "citations" not in content[index] or not isinstance(content[index]["citations"], list):
                        content[index]["citations"] = []
                    citations = cast(list[JSON], content[index]["citations"])
                    citations.append(citation)

                elif delta_type == "input_json_delta":
                    partial_json = get_or(delta, "partial_json", "")

                    if index not in tool_use_json:
                        tool_use_json[index] = StringIO()

                    tuj = tool_use_json[index]
                    _ = tuj.write(partial_json)

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
        messages.append(await use_tools(messages))
    else:
        call_again = False

    return messages, tokens, call_again


async def async_main():
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    args = common.parse_tai_args()

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
        system_prompt = common.load_system_prompt(args.role)

    # The response is already printed during streaming, so we don't need to print it again
    history: History = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]

    call_again = True
    while call_again:
        messages, _, call_again = await stream_response(
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


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
