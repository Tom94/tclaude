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
import sys
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from io import StringIO
from typing import Callable, cast

import aiohttp
from partial_json_parser import loads as partial_loads

from . import common, endpoints
from .common import History, pplain, pwarning
from .json import JSON, get, get_or, get_or_default
from .print import history_to_string

# Web search tool configuration
MAX_SEARCH_USES = 5
ALLOWED_DOMAINS = None  # Example: ["example.com", "trusteddomain.org"]
BLOCKED_DOMAINS = None  # Example: ["untrustedsource.com"]


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


async def stream_events(session: aiohttp.ClientSession, url: str, headers: dict[str, str], params: JSON) -> AsyncGenerator[JSON]:
    async with session.post(url, headers=headers, json=params) as response:
        response.raise_for_status()
        async for line in response.content:
            if line.startswith(b"data: "):
                try:
                    yield json.loads(line[6:])
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line[6:]}")
                    continue


@dataclass
class Response:
    messages: History
    tokens: TokenCounter
    call_again: bool


async def stream_response(
    session: aiohttp.ClientSession,
    model: str,
    history: History,
    max_tokens: int = 16384,
    enable_web_search: bool = True,
    enable_code_exec: bool = True,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    write_cache: bool = False,
    on_response_update: Callable[[Response], None] | None = None,
) -> Response:
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

    url, headers, params = endpoints.get_messages_endpoint_anthropic(model)
    # url, headers, params = endpoints.get_messages_endpoint_vertex("claude-sonnet-4@20250514")

    # Use the latest container if available
    container = common.get_latest_container(history)
    if container is not None:
        params["container"] = container.id

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
    async with contextlib.aclosing(stream_events(session, url, headers, params)) as events:
        async for data in events:
            match data:
                # Ping messages are just keep-alive signals, ignore them.
                case {"type": "ping"}:
                    continue

                # Message block types
                case {"type": "message_start", "message": dict(message)}:
                    messages.append(message)
                    messages[-1]["role"] = "assistant"
                    messages[-1]["content"] = []
                    tool_use_json = {}
                case {"type": "message_delta", "delta": dict(delta), **rest}:
                    usage = get(rest, "usage", dict[str, JSON])
                    if usage:
                        turn_tokens = TokenCounter(
                            cache_creation_input_tokens=get_or(usage, "cache_creation_input_tokens", 0),
                            cache_read_input_tokens=get_or(usage, "cache_read_input_tokens", 0),
                            input_tokens=get_or(usage, "input_tokens", 0),
                            output_tokens=get_or(usage, "output_tokens", 0),
                        )

                        tokens += turn_tokens

                    stop_reason = get(delta, "stop_reason", str)
                    if stop_reason is not None:
                        messages[-1]["stop_reason"] = stop_reason

                    container = get(delta, "container", dict[str, JSON])
                    if container:
                        messages[-1]["container"] = container
                case {"type": "message_end"}:
                    continue

                # Content block types
                case {"type": "content_block_start", "index": int(index), "content_block": dict(new_content_block)}:
                    content_blocks = get_or_default(messages[-1], "content", list[dict[str, JSON]])
                    while index >= len(content_blocks):
                        content_blocks.append({})
                    content_blocks[index] = new_content_block
                case {"type": "content_block_delta", "index": int(index), "delta": dict(delta)}:
                    content_block = get_or_default(messages[-1], "content", list[dict[str, JSON]])[index]
                    match delta:
                        case {"type": "thinking_delta", "thinking": str(thinking_delta)}:
                            thinking = cast(str, content_block.setdefault("thinking", ""))
                            content_block["thinking"] = thinking + thinking_delta
                        case {"type": "signature_delta", "signature": str(signature_delta)}:
                            signature = cast(str, content_block.setdefault("signature", ""))
                            content_block["signature"] = signature + signature_delta
                        case {"type": "text_delta", "text": str(text_delta)}:
                            text = cast(str, content_block.setdefault("text", ""))
                            content_block["text"] = text + text_delta
                        case {"type": "citations_delta", "citation": dict(citation)}:
                            citations = cast(list[JSON], content_block.setdefault("citations", []))
                            citations.append(citation)
                        case {"type": "input_json_delta", "partial_json": str(partial_json)}:
                            if index not in tool_use_json:
                                tool_use_json[index] = StringIO()

                            tuj = tool_use_json[index]
                            _ = tuj.write(partial_json)

                            if tuj.tell() > 0:
                                try:
                                    content_block["input"] = partial_loads(tuj.getvalue())
                                except:
                                    pass
                        case _:
                            pwarning(f"Unknown content block delta type: {delta}")
                case {"type": "content_block_stop", "index": int(index)}:
                    content_block = get_or_default(messages[-1], "content", list[dict[str, JSON]])[index]
                    pass  # Content block stop is just a signal that the content block is complete.

                # Something unexpected
                case _:
                    if not "message" in get_or(data, "type", ""):
                        pwarning(f"Unknown message type: {data}")

            if on_response_update is not None:
                on_response_update(Response(messages=messages, tokens=tokens, call_again=False))

    # Strangely, Anthropic's API sometimes returns empty text blocks (usually at the beginning of a message right before its first
    # citation). Returning these blocks to the API causes bad request errors, so we filter them out. Non-thinking empty blocks are filtered
    # just in case; I've never seen them in practice.
    def is_content_block_valid(content_block: JSON) -> bool:
        match content_block:
            case (
                {"type": "thinking", "thinking": ""}
                | {"type": "signature", "signature": ""}
                | {"type": "text", "text": ""}
                | {"type": "citations", "citations": []}
            ):
                pwarning(f"Content block {content_block} is empty, removing it.")
                return False
            case _:
                return True

    for message in messages:
        content_blocks = get_or_default(message, "content", list[dict[str, JSON]])
        if not content_blocks:
            pwarning("Message has no content blocks.")
        message["content"] = [cb for cb in content_blocks if is_content_block_valid(cb)]

    stop_reason = "unknown" if not messages else messages[-1].get("stop_reason")
    if stop_reason == "pause_turn":
        call_again = True
    elif stop_reason == "tool_use":
        call_again = True
        messages.append(await use_tools(messages))
    else:
        call_again = False

    return Response(
        messages=messages,
        tokens=tokens,
        call_again=call_again,
    )


async def async_main():
    """
    Main function to parse arguments, get user input, and print Anthropic's response.
    """
    args = common.parse_tai_args()
    user_input = common.read_user_input(args.input)
    if not user_input:
        print("No input provided.")
        return

    # Read system prompt from file if provided
    system_prompt = None
    if args.role:
        system_prompt = common.load_system_prompt(args.role)

    history: History = [{"role": "user", "content": [{"type": "text", "text": user_input}]}]

    async with aiohttp.ClientSession() as session:
        call_again = True
        while call_again:
            response = await stream_response(
                session=session,
                model=args.model,
                history=history,
                max_tokens=args.max_tokens,
                enable_web_search=not args.no_web_search,  # Web search is enabled by default
                enable_code_exec=not args.no_code_execution,  # Code execution is enabled by default
                system_prompt=system_prompt,
                enable_thinking=args.thinking,
                thinking_budget=args.thinking_budget,
            )

            history.extend(response.messages)
            call_again = response.call_again

    print(history_to_string(history[1:], pretty=False), end="", flush=True)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
