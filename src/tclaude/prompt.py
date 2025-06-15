# tclaude -- Claude in the terminal
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

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from io import StringIO
from typing import Callable, cast

import aiohttp
from partial_json_parser import loads as ploads  # pyright: ignore

from . import common, endpoints, files, tool_use
from .common import History
from .json import JSON, get, get_or, get_or_default
from .token_counter import TokenCounter
from .tool_use import AvailableTools

logger = logging.getLogger(__package__)

# Web search tool configuration
MAX_SEARCH_USES = 5
ALLOWED_DOMAINS = None  # Example: ["example.com", "trusteddomain.org"]
BLOCKED_DOMAINS = None  # Example: ["untrustedsource.com"]


async def stream_events(session: aiohttp.ClientSession, url: str, headers: dict[str, str], params: JSON) -> AsyncGenerator[JSON]:
    async with session.post(url, headers=headers, json=params) as response:
        # Anthropic API errors are returned as JSON objects with a specific structure. Translate into human-readable errors.
        if response.status >= 300 or response.status < 200:
            text = await response.text()
            try:
                json_data = cast(JSON, json.loads(text))
                match json_data:
                    case {"type": "error", "error": {"type": str(type), "message": str(message)}}:
                        raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status, message=f"{type}: {message}")
                    case _:
                        raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status, message=text)
            except json.JSONDecodeError:
                raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status, message=text)

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
    external_tools_available: AvailableTools | None = None,
    external_tool_definitions: list[JSON] | None = None,
    mcp_remote_servers: list[dict[str, JSON]] | None = None,
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

    if mcp_remote_servers:
        params["mcp_servers"] = mcp_remote_servers

    if enable_code_exec:
        code_exec_tool: JSON = {"type": "code_execution_20250522", "name": "code_execution"}
        tools.append(code_exec_tool)

    if external_tool_definitions is not None:
        tools.extend(external_tool_definitions)

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
                                    content_block["input"] = ploads(tuj.getvalue())
                                except Exception:
                                    pass
                        case _:
                            logger.warning(f"Unknown content block delta type: {delta}")
                case {"type": "content_block_stop", "index": int(index)}:
                    content_block = get_or_default(messages[-1], "content", list[dict[str, JSON]])[index]
                    pass  # Content block stop is just a signal that the content block is complete.

                # Something unexpected
                case _:
                    if "message" not in get_or(data, "type", ""):
                        logger.warning(f"Unknown message type: {data}")

            if on_response_update is not None:
                on_response_update(Response(messages=messages, tokens=tokens, call_again=False))

    # Strangely, Anthropic's API sometimes returns empty text blocks (usually at the beginning of a message right before its first
    # citation). Returning these blocks to the API causes bad request errors, so we filter them out. Non-thinking empty blocks are filtered
    # just in case; I've never seen them in practice.
    messages = filter_invalid_messages(messages)

    stop_reason = "unknown" if not messages else messages[-1].get("stop_reason")
    if stop_reason == "pause_turn":
        call_again = True
    elif stop_reason == "tool_use":
        call_again = True
        messages.append(await tool_use.use_tools(session, external_tools_available or {}, messages))
    else:
        call_again = False

    if on_response_update is not None:
        on_response_update(Response(messages=messages, tokens=tokens, call_again=False))

    return Response(
        messages=messages,
        tokens=tokens,
        call_again=call_again,
    )


def filter_invalid_messages(messages: History) -> History:
    def is_content_block_valid(content_block: JSON) -> bool:
        match content_block:
            case (
                {"type": "thinking", "thinking": ""}
                | {"type": "signature", "signature": ""}
                | {"type": "text", "text": ""}
                | {"type": "citations", "citations": []}
            ):
                logger.warning(f"Content block {content_block} is empty, removing it.")
                return False
            case _:
                return True

    result: History = []

    for message in messages:
        content_blocks = get_or_default(message, "content", list[dict[str, JSON]])
        if not content_blocks:
            logger.warning("Message has no content. Removing it.")
            continue

        message["content"] = [cb for cb in content_blocks if is_content_block_valid(cb)]
        result.append(message)

    return result


def file_metadata_to_content(metadata: JSON) -> list[JSON]:
    """
    Convert a file metadata JSON object to a list of content blocks that can be added to the history.
    """
    content: list[JSON] = []

    type = files.mime_type_to_content_block_type(get_or(metadata, "mime_type", ""))
    id = get(metadata, "id", str)
    if id is None:
        return content

    # Even if the type is invalid, the code execution tool might still be able to handle the file. Always put valid file IDs
    # into the code execution container.
    content.append({"type": "container_upload", "file_id": id})
    if type is None:
        return content

    info: dict[str, JSON] = {"type": type, "source": {"type": "file", "file_id": id}}
    if type == "document":
        info["context"] = "This document was uploaded by the user."
        info["citations"] = {"enabled": True}
        info["title"] = get_or(metadata, "filename", id)

    content.append(info)
    return content
