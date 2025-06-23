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

# This file deals with runtime reflection and, as such, has to use the Any type in some places.
# pyright: reportAny=false, reportExplicitAny=false

import asyncio
import importlib
import inspect
import logging
from collections.abc import Awaitable, Coroutine
from types import UnionType
from typing import Any, Callable, Literal, get_args, get_origin

import aiohttp
import docstring_parser

from .common import History
from .files import upload_file_base64
from .json import JSON, get, get_or, get_or_default
from .tools import ToolContentBase64Image, ToolContentText, ToolResult

logger = logging.getLogger(__package__)

type AvailableTools = dict[str, Callable[..., Coroutine[None, None, ToolResult]]]


def get_available_tools() -> AvailableTools:
    """
    Dynamically import tools.py and extract all callable functions.
    Returns a dictionary mapping function names to their callable objects.
    """
    try:
        tools_module = importlib.import_module(".tools", package=__package__)
        available_tools: AvailableTools = {}

        for name, obj in inspect.getmembers(tools_module):
            if inspect.isfunction(obj) and not name.startswith("_"):
                available_tools[name] = obj

        return available_tools
    except ImportError:
        return {}


def python_type_to_json_schema(python_type: Any) -> dict[str, JSON]:
    """Convert Python type annotation to JSON schema."""

    if python_type is type(None):
        return {"type": "null"}

    basic_types = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        bytes: "string",
        dict: "object",
        list: "array",
    }

    if python_type in basic_types:
        return {"type": basic_types[python_type]}

    origin = get_origin(python_type)
    args = get_args(python_type)

    if origin is UnionType:
        return {"anyOf": [python_type_to_json_schema(arg) for arg in args]}

    if origin is list:
        if args:
            return {"type": "array", "items": python_type_to_json_schema(args[0])}
        return {"type": "array"}

    if origin is dict:
        dict_schema: dict[str, JSON] = {"type": "object"}
        if len(args) != 2:
            logger.error(f"Dict type {python_type} must have exactly two type arguments (key and value).")
            return {}

        # dict[str, ValueType] - add pattern properties if key is string
        if args[0] is not str:
            logger.error(f"Unsupported dict key type: {args[0]}. Only string keys are supported for JSON Schema.")
            return {}

        dict_schema["additionalProperties"] = python_type_to_json_schema(args[1])
        return dict_schema

    if origin is Literal:
        return {"enum": list(args)}

    logger.error(f"Unsupported type: {python_type}. Please ensure it is a valid JSON schema type.")
    return {}


def replace_single_newlines(text: str) -> str:
    # Replace \n that is NOT followed by another \n
    return text.replace("\n", " ").replace("  ", "\n\n")


def extract_tool_definition_from_signature(name: str, func: Callable[..., object]) -> JSON:
    sig = inspect.signature(func)
    if sig.return_annotation == inspect.Signature.empty:
        logger.error(f"Tool `{name}` has no return annotation. Claude requires a return type annotation.")
        return None

    if sig.return_annotation is not ToolResult:
        logger.error(f"Tool `{name}` has a return type of {sig.return_annotation}, which is not supported by Claude. Must return ToolResult.")
        return None

    doc = docstring_parser.parse(inspect.getdoc(func) or "")
    param_descs = {param.arg_name: param.description for param in doc.params if param.description}

    description: str = doc.description or "No description provided."
    if doc.returns:
        description += f"\n\nReturns: {doc.returns.description or 'No return description provided.'}"
    description = replace_single_newlines(description)

    # Parse parameters
    properties: dict[str, dict[str, JSON]] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Try to extract type from annotation
        if param.annotation == inspect.Parameter.empty:
            logger.error(f"Tool `{name}` parameter `{param_name}` has no type annotation. Claude requires type annotations for all parameters.")
            return None

        schema = python_type_to_json_schema(param.annotation)
        if not schema:
            logger.error(f"Could not convert type `{param.annotation}` for tool `{name}` parameter `{param_name}` to JSON schema. Ensure it is a valid type.")
            return None

        schema["description"] = f"{param_name}"
        if param_name in param_descs:
            schema["description"] = replace_single_newlines(param_descs[param_name])
        else:
            logger.warning(f"Tool `{name}` parameter `{param_name}` has no description in the docstring. Claude may misunderstand the parameter's purpose.")

        properties[param_name] = schema

        # Add to required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


def get_python_tools() -> tuple[AvailableTools, list[JSON]]:
    """
    Generate tool definitions for Claude based on functions in tools.py.
    """
    available_tools = get_available_tools()
    tool_definitions: list[JSON] = [extract_tool_definition_from_signature(name, func) for name, func in available_tools.items()]
    tool_definitions = [td for td in tool_definitions if td is not None]

    return available_tools, tool_definitions


async def use_tool(client: aiohttp.ClientSession, tool_future: Awaitable[ToolResult]) -> tuple[list[dict[str, JSON]], bool]:
    result = await tool_future
    content: list[dict[str, JSON]] = []
    for c in result.content:
        if isinstance(c, ToolContentText):
            content.append({"type": "text", "text": c.text})
        elif isinstance(c, ToolContentBase64Image):  # pyright: ignore[reportUnnecessaryIsInstance]
            import mimetypes

            block: dict[str, JSON]
            try:
                extension = mimetypes.guess_extension(c.type) or ""
                if not extension:
                    raise RuntimeError(f"Could not determine file extension for media type {c.type}.")

                filename = f"{id}_{len(content)}{extension}"

                metadata = await upload_file_base64(client, filename, c.data, c.type, None)
                if "id" not in metadata:
                    raise RuntimeError(f"Tool result image upload failed: {metadata}")

                block = {"type": "image", "source": {"type": "file", "file_id": metadata["id"]}}
            except Exception as e:
                logger.error(f"Failed to upload tool result image. Passing image inline. {e}")
                block = {"type": "image", "source": {"type": "base64", "data": c.data, "media_type": c.type}}

            content.append(block)
        else:
            logger.warning(f"Unexpected content type in tool result: {c}")

    return content, result.is_error


async def use_tools(client: aiohttp.ClientSession, available_tools: AvailableTools, messages: History) -> dict[str, JSON]:
    """
    Use the tools specified in the messages to perform actions.
    This function is called when the model indicates that it wants to use a tool.
    """
    tool_results: list[dict[str, JSON]] = []

    # Find the last assistant message with tool use
    last_message = messages[-1] if messages else None
    if not last_message or last_message.get("role") != "assistant":
        return {"role": "user", "content": tool_results}

    # Process each content block that contains tool use. Tools are run in parallel.
    async with asyncio.TaskGroup() as tg:
        for content_block in get_or_default(last_message, "content", list[JSON]):
            if get(content_block, "type", str) == "tool_use":
                tool_name = get(content_block, "name", str)
                tool_input = get_or(content_block, "input", {})
                tool_use_id = get(content_block, "id", str)

                if tool_name and tool_name in available_tools:

                    async def tool_use_wrapper(name: str, input: dict[str, Any], result: dict[str, JSON]) -> None:
                        try:
                            tool_fun = available_tools[name]
                            result["content"], result["is_error"] = await use_tool(client, tool_fun(**input))
                        except (KeyboardInterrupt, asyncio.CancelledError, Exception) as e:
                            if isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)):
                                result["content"] = "Tool execution was cancelled."
                            else:
                                result["content"] = f"Error executing tool: {str(e)}"
                            result["is_error"] = True

                    tool_results.append({"type": "tool_result", "tool_use_id": tool_use_id})
                    _ = tg.create_task(tool_use_wrapper(tool_name, tool_input, tool_results[-1]))
                else:
                    tool_results.append({"type": "tool_result", "tool_use_id": tool_use_id, "content": f"Tool {tool_name} not found", "is_error": True})

    return {"role": "user", "content": tool_results}
